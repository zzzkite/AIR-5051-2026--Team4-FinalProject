#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import json
import threading
from std_msgs.msg import String

class GraspExecutor:
    def __init__(self):
        rospy.init_node('grasp_executor', anonymous=True)
        self.sub = rospy.Subscriber('/arm/command', String, self.callback)
        self.grasp_pub = rospy.Publisher('/grasp_target', String, queue_size=10)
        self.feedback_sub = rospy.Subscriber('/grasp_feedback', String, self.feedback_callback)
        self.detections_sub = rospy.Subscriber('/yolo_detections', String, self.detections_callback)
        self.latest_detections = []
        self.clear_mode = False
        self.clear_busy = False
        self.last_clear_target = None
        self.sequence_mode = False
        self.pending_grasp_targets = []
        self.sequence_busy = False
        self.sequence_current_target = None
        self.sequence_red_clear_mode = False
        rospy.loginfo("GraspExecutor 已启动，等待 /arm/command 消息...")

    def detections_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            self.latest_detections = payload.get('detections', [])
        except Exception as e:
            rospy.logerr("YOLO检测结果解析失败: %s", e)

    def feedback_callback(self, msg):
        data = msg.data.strip()
        if not self.clear_mode and not self.sequence_mode:
            return

        if data.startswith('success:') or data.startswith('failed:'):
            if self.clear_mode:
                self.clear_busy = False
                threading.Timer(0.8, self.maybe_continue_clear).start()
            if self.sequence_mode:
                if data.startswith('failed:') and self.sequence_current_target is not None:
                    self.pending_grasp_targets.insert(0, self.sequence_current_target)
                    rospy.loginfo("顺序抓取失败：重新加入队首 %s", self.sequence_current_target.get('exact_class'))
                self.sequence_current_target = None
                self.sequence_busy = False
                threading.Timer(0.8, self.maybe_continue_sequence).start()

    def coarse_color_from_class(self, cls_name):
        name = (cls_name or '').lower()
        if 'green' in name:
            return 'green'
        if 'blue' in name:
            return 'blue'
        if 'red' in name or 'cola' in name:
            return 'red'
        return None

    def pick_next_clear_target(self):
        best_det = None
        best_conf = -1.0
        for det in self.latest_detections:
            coarse_color = det.get('coarse_color') or self.coarse_color_from_class(det.get('class'))
            if coarse_color not in ['red', 'green', 'blue']:
                continue
            conf = float(det.get('confidence', 0.0))
            if conf > best_conf:
                best_conf = conf
                best_det = coarse_color
        return best_det

    def maybe_continue_clear(self):
        if not self.clear_mode or self.clear_busy:
            return
        next_color = self.pick_next_clear_target()
        if next_color is None:
            rospy.loginfo("清空桌面完成：当前未检测到任何目标")
            self.clear_mode = False
            self.last_clear_target = None
            return

        self.last_clear_target = next_color
        self.clear_busy = True
        rospy.loginfo("清空桌面：继续抓取 %s 类目标", next_color)
        self.grasp_pub.publish(next_color)

    def build_grasp_payload(self, target):
        payload = {
            'color': (target.get('color') or '').lower(),
        }
        shape = target.get('shape')
        if shape:
            payload['shape'] = shape
        exact_class = target.get('exact_class')
        if exact_class:
            payload['exact_class'] = exact_class
        return payload

    def pick_next_red_target(self):
        best_det = None
        best_conf = -1.0
        for det in self.latest_detections:
            coarse_color = det.get('coarse_color') or self.coarse_color_from_class(det.get('class'))
            if coarse_color != 'red':
                continue
            cls_name = (det.get('class') or '').lower()
            if cls_name not in ['redcube', 'cola']:
                continue
            conf = float(det.get('confidence', 0.0))
            if conf > best_conf:
                best_conf = conf
                best_det = cls_name
        return best_det

    def maybe_continue_sequence(self):
        if not self.sequence_mode or self.sequence_busy:
            return

        if self.sequence_red_clear_mode:
            next_class = self.pick_next_red_target()
            if next_class is None:
                rospy.loginfo("红色物体清理完成：当前未检测到 redcube 或 cola")
                self.sequence_mode = False
                self.sequence_red_clear_mode = False
                self.sequence_busy = False
                self.sequence_current_target = None
                return

            self.sequence_current_target = {'color': 'red', 'exact_class': next_class}
            self.sequence_busy = True
            rospy.loginfo("红色连续清理：继续处理 %s", next_class)
            self.grasp_pub.publish(json.dumps(self.sequence_current_target, ensure_ascii=False))
            return

        if not self.pending_grasp_targets:
            rospy.loginfo("顺序抓取完成：所有目标已处理")
            self.sequence_mode = False
            self.sequence_busy = False
            return

        next_target = self.pending_grasp_targets.pop(0)
        self.sequence_current_target = next_target
        self.sequence_busy = True
        rospy.loginfo("顺序抓取：继续处理 %s", next_target.get('exact_class'))
        self.grasp_pub.publish(json.dumps(self.build_grasp_payload(next_target), ensure_ascii=False))

    def callback(self, msg):
        # 打印原始 JSON 输出
        rospy.loginfo("收到 JSON: %s", msg.data)

        try:
            cmd = json.loads(msg.data)
        except Exception as e:
            rospy.logerr("JSON 解析失败: %s", e)
            return

        # 只处理 grasp 动作
        action = cmd.get('action')
        if action == 'clear_table':
            self.clear_mode = True
            self.clear_busy = False
            rospy.loginfo("进入清空桌面模式")
            self.maybe_continue_clear()
            return

        if action != 'grasp':
            rospy.logwarn("收到非 grasp 动作: %s", action)
            return

        target = cmd.get('target', {})
        color = target.get('color')
        shape = target.get('shape')
        exact_class = target.get('exact_class')
        rospy.loginfo("目标物体: 颜色=%s, 形状=%s", color, shape)

        if color and color.lower() in ['red', 'green', 'blue']:
            rospy.loginfo("✅ %s 物体，触发抓取", color)
            if isinstance(exact_class, list):
                self.sequence_mode = True
                self.sequence_busy = False
                self.pending_grasp_targets = []
                self.sequence_red_clear_mode = color.lower() == 'red' and set([item for item in exact_class if item]) <= set(['redcube', 'cola'])
                if not self.sequence_red_clear_mode:
                    for item in exact_class:
                        if item:
                            self.pending_grasp_targets.append({
                                'color': color.lower(),
                                'shape': shape,
                                'exact_class': item,
                            })
                self.maybe_continue_sequence()
            else:
                payload = {'color': color.lower()}
                if exact_class:
                    payload['exact_class'] = exact_class
                self.grasp_pub.publish(json.dumps(payload, ensure_ascii=False))
        else:
            rospy.logwarn("❌ 不支持的目标颜色，不执行抓取")

if __name__ == '__main__':
    try:
        executor = GraspExecutor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass