#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM → ROS 指令桥接节点（带闭环重试 + 动态修正偏移）
"""

import rospy
import threading
import json
import requests
import os
import re
import queue
import copy
from std_msgs.msg import String
from datetime import datetime

# ========== 可配置参数 ==========
DEVICE_IP = rospy.get_param('~llm_ip', "10.31.171.185")
PORT = rospy.get_param('~llm_port', "8910")
MODEL_NAME = rospy.get_param('~model_name', "Phi-3.5-mini")
OUTPUT_DIR = rospy.get_param('~output_dir', "/tmp/arm_commands")
TOPIC_NAME = rospy.get_param('~output_topic', "/arm/command")
INPUT_TOPIC = rospy.get_param('~input_topic', "/arm/user_command")
LLM_TIMEOUT = rospy.get_param('~llm_timeout', 30)
MAX_RETRY = rospy.get_param('~max_retry', 3)
LLM_FEEDBACK_TIMEOUT = rospy.get_param('~llm_feedback_timeout', 90)

REQUIRED_FIELDS = ["action", "target"]
VALID_ACTIONS = ["grasp", "move", "place", "wait", "open_gripper", "close_gripper"]
VALID_ACTIONS.append("clear_table")

# ========== 全局变量 ==========
pub = None
llm_url = None
command_queue = queue.Queue(maxsize=10)
offset_pub = None   # 发布偏移修正

last_user_command = None
retry_count = 0
retry_in_progress = False
current_color = None
last_command_dict = None
latest_yolo_detections = None

def extract_json_from_text(text):
    if not text:
        return None
    cleaned = text.replace("```json", "").replace("```", "")
    start = cleaned.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return cleaned[start:index + 1]
    return None

def extract_json_objects_from_text(text):
    if not text:
        return []
    cleaned = text.replace("```json", "").replace("```", "")
    objects = []
    start = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(cleaned):
        if start is None:
            if char == '{':
                start = index
                depth = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    objects.append(cleaned[start:index + 1])
                    start = None

    return objects

def merge_grasp_commands(cmd_list):
    if not cmd_list:
        return None
    if len(cmd_list) == 1:
        return cmd_list[0]

    if any(not isinstance(cmd, dict) for cmd in cmd_list):
        return cmd_list[0]

    actions = {cmd.get("action") for cmd in cmd_list}
    if actions != {"grasp"}:
        return cmd_list[0]

    targets = [cmd.get("target", {}) for cmd in cmd_list]
    colors = {(target.get("color") or "").lower() for target in targets}
    if len(colors) != 1:
        return cmd_list[0]

    merged_target = copy.deepcopy(targets[0])
    exact_classes = []
    for target in targets:
        exact_class = target.get("exact_class")
        if isinstance(exact_class, list):
            for item in exact_class:
                if item and item not in exact_classes:
                    exact_classes.append(item)
        elif isinstance(exact_class, str) and exact_class and exact_class not in exact_classes:
            exact_classes.append(exact_class)

    if exact_classes:
        merged_target["exact_class"] = exact_classes

    merged_cmd = copy.deepcopy(cmd_list[0])
    merged_cmd["target"] = merged_target
    return merged_cmd

def validate_command(cmd_dict):
    if not isinstance(cmd_dict, dict):
        return False, "Not a dict"
    for field in REQUIRED_FIELDS:
        if field not in cmd_dict:
            if cmd_dict.get("action") == "clear_table" and field == "target":
                continue
            return False, f"Missing field: {field}"
    action = cmd_dict.get("action")
    if action and action not in VALID_ACTIONS:
        return False, f"Invalid action: {action}"
    target = cmd_dict.get("target")
    if target and not isinstance(target, dict):
        return False, "target must be an object"
    if isinstance(target, dict):
        exact_class = target.get("exact_class")
        if isinstance(exact_class, list):
            if not exact_class or not all(isinstance(item, str) and item for item in exact_class):
                return False, "exact_class list must contain non-empty strings"
        elif exact_class is not None and not isinstance(exact_class, str):
            return False, "exact_class must be a string or a list of strings"
    return True, "OK"

def chat_with_model(prompt, callback=None, timeout_sec=None, temperature=0.8, top_p=1.0):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temp": temperature,
        "top_p": top_p
    }
    headers = {"Content-Type": "application/json"}
    full_response = ""
    request_timeout = timeout_sec if timeout_sec is not None else LLM_TIMEOUT
    try:
        response = requests.post(llm_url, json=payload, headers=headers, stream=True, timeout=(10, request_timeout))
        if response.status_code != 200:
            rospy.logerr(f"LLM HTTP {response.status_code}: {response.text[:200]}")
            return None
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        full_response += content
                        if callback:
                            callback(content)
                except:
                    continue
        return full_response
    except Exception as e:
        rospy.logerr(f"LLM call failed: {e}")
        return None

def extract_color_from_command(cmd_dict):
    target = cmd_dict.get("target", {})
    return target.get("color", "").lower()

def publish_command(cmd_dict, user_text, retry_note=None):
    payload = copy.deepcopy(cmd_dict)
    payload["timestamp"] = rospy.get_time()
    payload["node"] = rospy.get_name()
    payload["raw_prompt"] = user_text
    if retry_note:
        payload["retry_note"] = retry_note
    try:
        json_msg = String(data=json.dumps(payload, ensure_ascii=False))
        pub.publish(json_msg)
        rospy.loginfo(f"📤 已发布到 {TOPIC_NAME}")
    except Exception as e:
        rospy.logerr(f"发布话题失败: {e}")
        return None
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = os.path.join(OUTPUT_DIR, f"cmd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        rospy.loginfo(f"💾 已保存: {filename}")
    except Exception as e:
        rospy.logerr(f"保存文件失败: {e}")
    return payload

def process_user_command(user_text, is_retry=False):
    global last_user_command, retry_count, current_color, last_command_dict

    if not is_retry:
        last_user_command = user_text
        retry_count = 0
        current_color = None
        publish_offset(0, 0)
        rospy.loginfo("新一轮指令开始，已清空抓取偏移")

    rospy.loginfo(f"🔍 处理指令: {user_text[:100]}")

    system_hint = """你是一个机械臂控制指令生成器。请将用户指令转换为严格JSON格式，不要额外解释或Markdown。
可用动作: grasp, move, place, wait, open_gripper, close_gripper
新增动作: clear_table
输出格式示例:
    {"action":"grasp","target":{"color":"red","shape":"cube","exact_class":"redcube","position":{"x":0.3,"y":-0.1,"z":0.45}},"gripper":{"action":"close","force":0.8}}
    {"action":"grasp","target":{"color":"red","shape":"object","exact_class":["redcube","cola"]},"gripper":{"action":"close","force":0.8}}
识别规则:
- 你只需要认识 4 个标签：greencube、redcube、bluecube、cola。
- 用户说“绿色方块”“绿色cube”时，必须输出 color=green 且 exact_class=greencube。
- 用户说“可乐”“cola”时，必须输出 color=red 且 exact_class=cola，不能输出 redcube。
- 用户说“红色易拉罐”“红色可乐罐”“红罐”“红色罐子”时，必须输出 color=red 且 exact_class=cola，不能输出 redcube。
- 用户说“红色方块”“红方块”“红色cube”“红cube”时，必须输出 color=red 且 exact_class=redcube，不能把 cola 当成同一目标。
- 用户说“红色物体”“红色东西”“红物体”“红色目标”时，必须输出 color=red 且 exact_class=["redcube","cola"]，表示需要按顺序同时处理这两个目标。
- 如果用户说“抓取全部红色物体”或同类表达，请只输出一个 JSON，并把 exact_class 直接写成 ["redcube","cola"]，不要拆成多个 JSON。
- 用户说“方块”“cube”时，优先理解为 cube 类目标，不要把 cola 当成方块。
- 用户说“清空桌面”时，输出 action=clear_table，让下游持续抓取直到当前检测中没有任何目标。
"""
    prompt = f"{system_hint}\n\n用户指令: {user_text}\n\n请只根据上述标签和比较规则输出JSON，不要引入视觉检测结果。\nJSON输出:"
    raw_text = chat_with_model(prompt)
    if not raw_text:
        rospy.logerr("❌ LLM 无有效响应")
        return
    json_objects = extract_json_objects_from_text(raw_text)
    if not json_objects:
        rospy.logerr(f"❌ 未提取到JSON | 原始输出: {raw_text[:300]}")
        return
    parsed_cmds = []
    for json_str in json_objects:
        try:
            parsed_cmds.append(json.loads(json_str))
        except json.JSONDecodeError as e:
            rospy.logwarn(f"⚠️ 单个JSON解析失败: {e}")
    if not parsed_cmds:
        rospy.logerr(f"❌ JSON解析失败 | 原始输出: {raw_text[:300]}")
        return
    cmd_dict = merge_grasp_commands(parsed_cmds)
    valid, msg = validate_command(cmd_dict)
    if not valid:
        rospy.logwarn(f"⚠️ 指令校验失败: {msg}")
        return
    rospy.loginfo(f"✅ 指令校验通过: action={cmd_dict['action']}")
    color = extract_color_from_command(cmd_dict)
    if color and not is_retry:
        current_color = color
    last_command_dict = publish_command(cmd_dict, user_text)

def publish_offset(offset_x, offset_y):
    """发布修正偏移到 /grasp_offset"""
    msg = json.dumps({"offset_x": offset_x, "offset_y": offset_y})
    offset_pub.publish(String(data=msg))
    rospy.loginfo(f"📤 发布偏移修正: ox={offset_x}, oy={offset_y}")

def compute_fallback_offset(current_ox, current_oy, center_dx, center_dy, edge_distance, retry_num):
    """当 LLM 超时或输出异常时，给出一个保底的几何修正。"""
    if edge_distance is None or edge_distance < 0:
        edge_boost = 8
    elif edge_distance < 20:
        edge_boost = 24
    elif edge_distance < 40:
        edge_boost = 20
    elif edge_distance < 80:
        edge_boost = 14
    else:
        edge_boost = 10

    retry_boost = 12 + max(0, retry_num - 1) * 4
    magnitude = max(50, edge_boost + retry_boost)

    def toward_center(value):
        if value > 0:
            return -1
        if value < 0:
            return 1
        return 0

    delta_x = toward_center(center_dx) * magnitude
    delta_y = toward_center(center_dy) * magnitude
    fallback_x = int(max(-200, min(200, current_ox + delta_x)))
    fallback_y = int(max(-200, min(200, current_oy + delta_y)))
    return fallback_x, fallback_y

def compute_reference_offset(current_ox, current_oy, center_dx, center_dy, edge_distance, retry_num):
    """给 LLM 的参考起点，用于稳定提示并鼓励更激进的修正。"""
    if edge_distance is None or edge_distance < 0:
        base = 12
    elif edge_distance < 20:
        base = 30
    elif edge_distance < 40:
        base = 24
    elif edge_distance < 80:
        base = 18
    else:
        base = 12

    retry_bonus = 8 + max(0, retry_num - 1) * 5
    magnitude = max(50, base + retry_bonus)

    def outward_bias(value):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    ref_x = int(max(-200, min(200, current_ox + outward_bias(center_dx) * magnitude)))
    ref_y = int(max(-200, min(200, current_oy + outward_bias(center_dy) * magnitude)))
    return ref_x, ref_y

def feedback_callback(msg):
    global retry_count, retry_in_progress, current_color, last_user_command, last_command_dict
    if retry_in_progress:
        return
    data = msg.data.strip()
    if data.startswith("failed:"):
        parts = data.split(":")
        if len(parts) >= 3:
            color = parts[1]
            try:
                pre_x = int(parts[2])
                pre_y = int(parts[3])
                post_x = int(parts[4]) if len(parts) > 4 else -1
                post_y = int(parts[5]) if len(parts) > 5 else -1
                delta_x = int(parts[6]) if len(parts) > 6 else (post_x - pre_x if post_x >= 0 else 0)
                delta_y = int(parts[7]) if len(parts) > 7 else (post_y - pre_y if post_y >= 0 else 0)
                current_ox = int(parts[8]) if len(parts) > 8 else 0
                current_oy = int(parts[9]) if len(parts) > 9 else 0
                img_w = int(parts[10]) if len(parts) > 10 else 0
                img_h = int(parts[11]) if len(parts) > 11 else 0
                center_dx = int(parts[12]) if len(parts) > 12 else 0
                center_dy = int(parts[13]) if len(parts) > 13 else 0
                edge_distance = int(parts[14]) if len(parts) > 14 else -1
            except:
                pre_x = pre_y = post_x = post_y = -1
                delta_x = delta_y = 0
                current_ox = current_oy = 0
                img_w = img_h = 0
                center_dx = center_dy = 0
                edge_distance = -1
        else:
            color = "unknown"
            pre_x = pre_y = post_x = post_y = -1
            delta_x = delta_y = 0
            current_ox = current_oy = 0
            img_w = img_h = 0
            center_dx = center_dy = 0
            edge_distance = -1

        if retry_count >= MAX_RETRY:
            rospy.logerr(f"已达到最大重试次数 ({MAX_RETRY})，放弃重试")
            retry_count = 0
            return

        retry_in_progress = True
        retry_count += 1
        rospy.logwarn(f"抓取失败（颜色: {color}），第 {retry_count} 次重试，请求 LLM 提供修正偏移")

        # 构造 prompt 让 LLM 分析失败原因并给出偏移修正
        ref_x, ref_y = compute_reference_offset(current_ox, current_oy, center_dx, center_dy, edge_distance, retry_count)
        prompt = f"""你是一个机器人抓取修正专家。抓取 {color} 物体失败。
    当前是第 {retry_count} 次重试，请从第一次重试开始就给出激进补偿，不要保守，不要重复模板值。
    重试策略：
    - 第 1 次重试：直接给出激进修正，幅度至少 50 像素，明显大于常规小偏移。
    - 第 2 次重试：在第 1 次基础上继续加大，优先向中心方向更强补偿。
    - 第 3 次及以后：给出更强修正，避免继续停留在小幅度范围。

    抓取前的像素坐标: ({pre_x}, {pre_y})
    抓取后的检测坐标: ({post_x}, {post_y})（-1表示未检测到）
    像素移动量: ({delta_x}, {delta_y})
    图像尺寸: ({img_w}, {img_h})
    目标相对图像中心偏移: ({center_dx}, {center_dy})
    目标距离图像边缘的最近距离: {edge_distance}
    当前应用中的修正: offset_x={current_ox}, offset_y={current_oy}
    几何参考偏移（仅作参考，不要照抄；你的结果应当比它更像真实分析后的结论）: offset_x={ref_x}, offset_y={ref_y}

    请综合考虑以下因素：
    1. 前后位置不一致时，不要把它当作随机噪声；优先把它解释成机械臂接触后把目标推偏，但没有真正抓牢。
    2. 如果 post 位置比 pre 位置更靠近图像中心，说明你的抓取点偏内，下一轮应当把抓取点更向外推；如果 post 更靠近边缘，则相反。
    3. 人工标定存在误差，但不要因此保守，标定误差应作为额外补偿的理由，而不是缩小修正。
    4. 靠近视野边缘时，要朝图像中心方向更大幅度补偿，边缘越近补偿越大；通常抓取点都偏内，所以最终抓取点应比当前预计位置更向外一些。
    5. 你的输出应该显著偏离当前 offset，尤其第 1 次重试就要直接进入激进区间，绝对值尽量不小于 50 像素，不要输出小模板值。

    只输出单个 JSON 对象，不要解释，不要 Markdown，不要复述示例句子。
    输出格式必须严格为：{{"offset_x": 数值, "offset_y": 数值}}
    偏移含义是“在当前检测像素基础上额外加的修正”，如果你判断目标偏左就让抓取点向右补偿，偏上就向下补偿，反之亦然。
    如果你认为不需要修正，就输出 0。"""
        response = chat_with_model(prompt, timeout_sec=LLM_FEEDBACK_TIMEOUT, temperature=0.0, top_p=0.3)
        if response:
            # 提取 JSON
            json_str = extract_json_from_text(response)
            if json_str:
                try:
                    offset = json.loads(json_str)
                    ox = offset.get("offset_x", 0)
                    oy = offset.get("offset_y", 0)
                    # 限制偏移范围避免过大
                    ox = max(-200, min(200, ox))
                    oy = max(-200, min(200, oy))
                    publish_offset(ox, oy)
                except:
                    rospy.logerr("解析LLM偏移响应失败，使用默认偏移0")
                    fallback_x, fallback_y = compute_fallback_offset(current_ox, current_oy, center_dx, center_dy, edge_distance, retry_count)
                    publish_offset(fallback_x, fallback_y)
            else:
                rospy.logwarn("LLM未输出有效JSON，使用几何保底偏移")
                fallback_x, fallback_y = compute_fallback_offset(current_ox, current_oy, center_dx, center_dy, edge_distance, retry_count)
                publish_offset(fallback_x, fallback_y)
        else:
            rospy.logwarn("LLM无响应，使用几何保底偏移")
            fallback_x, fallback_y = compute_fallback_offset(current_ox, current_oy, center_dx, center_dy, edge_distance, retry_count)
            publish_offset(fallback_x, fallback_y)

        # 重新发送上一次已经生成并发布过的指令，不再额外请求一次 LLM
        if last_command_dict is not None:
            retry_note = f"retry_after_failed_grasp:{color}"
            publish_command(last_command_dict, last_user_command or "", retry_note=retry_note)
        retry_in_progress = False

    elif data.startswith("success:"):
        color = data.split(":")[1] if ":" in data else "unknown"
        rospy.loginfo(f"抓取成功（颜色: {color}），重试计数已重置，偏移保持当前值")
        retry_count = 0

def command_callback(msg):
    user_text = msg.data.strip()
    if not user_text:
        return
    try:
        command_queue.put_nowait(user_text)
    except queue.Full:
        rospy.logwarn("指令队列已满")

def queue_processor_timer(event):
    while not command_queue.empty():
        try:
            user_text = command_queue.get_nowait()
            process_user_command(user_text, is_retry=False)
        except queue.Empty:
            break
        except Exception as e:
            rospy.logerr(f"处理指令异常: {e}")

def main():
    global pub, llm_url, offset_pub
    rospy.init_node('llm_arm_bridge', anonymous=True)
    llm_url = f"http://{DEVICE_IP}:{PORT}/v1/chat/completions"
    rospy.loginfo(f"🔗 连接LLM: {llm_url} | 模型: {MODEL_NAME}")

    pub = rospy.Publisher(TOPIC_NAME, String, queue_size=10)
    offset_pub = rospy.Publisher('/grasp_offset', String, queue_size=10)
    rospy.loginfo(f"📤 发布话题: {TOPIC_NAME}")
    rospy.loginfo("📤 发布偏移修正话题: /grasp_offset")

    rospy.Subscriber(INPUT_TOPIC, String, command_callback)
    rospy.loginfo(f"👂 监听指令: {INPUT_TOPIC}")

    rospy.Subscriber("/grasp_feedback", String, feedback_callback)
    rospy.loginfo("👂 监听抓取反馈: /grasp_feedback")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rospy.Timer(rospy.Duration(0.1), queue_processor_timer)
    rospy.loginfo("🚀 LLM 桥接节点启动成功（支持闭环重试+动态偏移修正）")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点退出")
    except KeyboardInterrupt:
        rospy.loginfo("用户中断")