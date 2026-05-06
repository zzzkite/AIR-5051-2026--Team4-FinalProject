#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import cv2
import numpy as np
import yaml
import json
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO
from sagittarius_object_color_detector.msg import SGRCtrlAction, SGRCtrlGoal, SGRCtrlResult

class YOLOBasedGrasp:
    def __init__(self):
        rospy.init_node('yolo_based_grasp', anonymous=True)

        # ----- 参数读取 -----
        self.arm_name = rospy.get_param("~arm_name", "sgr532")
        self.search_x = rospy.get_param("~search_x", 0.20)
        self.search_y = rospy.get_param("~search_y", 0.00)
        self.search_z = rospy.get_param("~search_z", 0.16)
        self.search_pitch = rospy.get_param("~search_pitch", 1.55)
        self.pick_z = rospy.get_param("~pick_z", 0.02)
        self.pick_pitch = rospy.get_param("~pick_pitch", 1.57)
        self.put_z = rospy.get_param("~put_z", 0.2)

        model_path = rospy.get_param("~model_path", "")
        self.conf_thres = rospy.get_param("~conf_thres", 0.6)
        self.fail_conf_thres = rospy.get_param("~fail_conf_thres", 0.4)
        self.stable_frames = rospy.get_param("~stable_frames", 5)
        self.position_threshold = rospy.get_param("~position_threshold", 15)

        # 标定参数
        vision_config = rospy.get_param("~vision_config", "")
        if not vision_config:
            rospy.logerr("未提供 vision_config 参数")
            rospy.signal_shutdown("No vision_config")
            return
        try:
            with open(vision_config, 'r') as f:
                cfg = yaml.safe_load(f)
            self.k1 = cfg['LinearRegression']['k1']
            self.b1 = cfg['LinearRegression']['b1']
            self.k2 = cfg['LinearRegression']['k2']
            self.b2 = cfg['LinearRegression']['b2']
            rospy.loginfo("标定参数: k1=%.5f, b1=%.5f, k2=%.5f, b2=%.5f",
                          self.k1, self.b1, self.k2, self.b2)
        except Exception as e:
            rospy.logerr("加载标定参数失败: %s", e)
            rospy.signal_shutdown("Calib error")
            return

        self.box_dst = {
            'red':   {'x': 0.34, 'y': 0.18},
            'green': {'x': 0.18, 'y': 0.18},
            'blue':  {'x': 0.26, 'y': 0.18}
        }

        # 加载 YOLO
        if not model_path:
            rospy.logerr("未提供模型路径")
            rospy.signal_shutdown("No model")
            return
        try:
            self.model = YOLO(model_path)
            rospy.loginfo("YOLO 模型加载成功: %s", model_path)
        except Exception as e:
            rospy.logerr("模型加载失败: %s", e)
            return

        # 机械臂 Action
        action_topic = self.arm_name + '/sgr_ctrl'
        rospy.loginfo("等待 Action Server: %s ...", action_topic)
        self.client = actionlib.SimpleActionClient(action_topic, SGRCtrlAction)
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("无法连接到机械臂控制服务器")
            rospy.signal_shutdown("No action server")
            return
        rospy.loginfo("已连接到机械臂控制服务器")

        # 图像订阅和状态
        self.bridge = CvBridge()
        self.detection_enabled = False
        self.is_grasping = False
        self.target_color = None
        self.target_class = None
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.stable_counter = 0
        self.stable_target = None
        self.last_positions = []
        self.empty_counter = 0
        self.last_grasp_pixel = None
        self.stable_target_key = None

        # 图像缓存
        self.latest_image = None
        self.image_seq = 0
        self.latest_frame_id = 0
        self.grasp_task_running = False
        self.latest_image_size = (0, 0)

        self.feedback_pub = rospy.Publisher('/grasp_feedback', String, queue_size=10)
        self.detections_pub = rospy.Publisher('/yolo_detections', String, queue_size=10)
        self.offset_pub = rospy.Publisher('/grasp_offset', String, queue_size=10)
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        self.target_sub = rospy.Subscriber("/grasp_target", String, self.target_callback)
        self.offset_sub = rospy.Subscriber("/grasp_offset", String, self.offset_callback)
        rospy.loginfo("已订阅图像话题: /usb_cam/image_raw")
        rospy.loginfo("已订阅目标颜色话题: /grasp_target")
        rospy.loginfo("已订阅偏移修正话题: /grasp_offset")
        rospy.loginfo("发布抓取反馈到: /grasp_feedback")
        rospy.loginfo("发布YOLO检测结果到: /yolo_detections")

        # Action 目标
        self.goal_search = SGRCtrlGoal()
        self.goal_search.action_type = self.goal_search.ACTION_TYPE_XYZ_RPY
        self.goal_search.grasp_type = self.goal_search.GRASP_OPEN
        self.goal_search.pos_x = self.search_x
        self.goal_search.pos_y = self.search_y
        self.goal_search.pos_z = self.search_z
        self.goal_search.pos_pitch = self.search_pitch

        self.goal_pick = SGRCtrlGoal()
        self.goal_pick.action_type = self.goal_pick.ACTION_TYPE_PICK_XYZ
        self.goal_pick.grasp_type = self.goal_pick.GRASP_OPEN
        self.goal_pick.pos_z = self.pick_z
        self.goal_pick.pos_pitch = self.pick_pitch

        self.goal_put = SGRCtrlGoal()
        self.goal_put.action_type = self.goal_put.ACTION_TYPE_PUT_XYZ
        self.goal_put.grasp_type = self.goal_put.GRASP_CLOSE
        self.goal_put.pos_z = self.put_z
        self.goal_put.pos_pitch = self.pick_pitch

    def coarse_color_from_class(self, cls_name):
        name = cls_name.lower()
        if 'green' in name:
            return 'green'
        if 'blue' in name:
            return 'blue'
        if 'red' in name or 'cola' in name:
            return 'red'
        return None

    def select_best_detection(self, boxes, image_width, image_height, target_color=None, target_class=None):
        best_det = None
        best_conf = -1.0
        detections = []

        if boxes is None:
            return best_det, detections

        for box in boxes:
            cls_name = self.model.names[int(box.cls[0])]
            coarse_color = self.coarse_color_from_class(cls_name)
            conf = float(box.conf[0])
            center_x = int(box.xywhn[0, 0] * image_width)
            center_y = int(box.xywhn[0, 1] * image_height)

            detections.append({
                'class': cls_name,
                'coarse_color': coarse_color,
                'confidence': conf,
                'center_x': center_x,
                'center_y': center_y,
            })

            exact_class = cls_name.lower()
            class_match = target_class is None or exact_class == target_class.lower()
            color_match = target_color is None or coarse_color == target_color
            if class_match and color_match:
                if conf > best_conf:
                    best_conf = conf
                    best_det = (cls_name, (center_x, center_y), coarse_color)

        return best_det, detections

    def publish_detections(self, detections, image_width, image_height):
        payload = {
            'timestamp': rospy.get_time(),
            'image_width': image_width,
            'image_height': image_height,
            'detections': detections,
        }
        try:
            self.detections_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        except Exception as e:
            rospy.logerr("发布YOLO检测结果失败: %s", e)

    def publish_offset(self, offset_x, offset_y):
        payload = json.dumps({"offset_x": offset_x, "offset_y": offset_y})
        try:
            self.offset_pub.publish(String(data=payload))
        except Exception as e:
            rospy.logerr("发布偏移修正失败: %s", e)

    def publish_scene_snapshot(self):
        if self.latest_image is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        except Exception as e:
            rospy.logerr("抓取后场景快照图像转换失败: %s", e)
            return

        h, w = cv_image.shape[:2]
        results = self.model(cv_image, conf=self.conf_thres, verbose=False)
        boxes = results[0].boxes
        _, detections = self.select_best_detection(boxes, w, h, target_color=None, target_class=None)
        self.publish_detections(detections, w, h)

    def offset_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.offset_x = data.get("offset_x", 0.0)
            self.offset_y = data.get("offset_y", 0.0)
            rospy.loginfo("更新偏移修正: offset_x=%.1f, offset_y=%.1f", self.offset_x, self.offset_y)
        except Exception as e:
            rospy.logerr("解析偏移修正失败: %s", e)

    def target_callback(self, msg):
        raw_text = msg.data.strip()
        color = None
        target_class = None
        try:
            payload = json.loads(raw_text)
            color = (payload.get('color') or '').strip().lower()
            target_class = (payload.get('exact_class') or '').strip().lower() or None
        except Exception:
            color = raw_text.lower()

        if color not in ['red', 'green', 'blue']:
            rospy.logwarn("无效目标颜色: %s", color)
            return
        if self.target_color == color and self.target_class == target_class:
            rospy.loginfo("已处于目标颜色 %s，忽略重复指令", color)
            return
        rospy.loginfo("收到抓取目标: %s%s，开始执行", color, f"/{target_class}" if target_class else "")
        self.target_color = color
        self.target_class = target_class
        self.stable_counter = 0
        self.stable_target = None
        self.last_positions.clear()
        self.empty_counter = 0
        if not self.is_grasping and not self.detection_enabled:
            self.client.send_goal_and_wait(self.goal_search, rospy.Duration(30.0))
            rospy.loginfo("机械臂已到达拍照位置")
        self.detection_enabled = True

    def force_reset(self):
        self.stable_counter = 0
        self.stable_target = None
        self.stable_target_key = None
        self.last_positions.clear()
        self.empty_counter = 0

    def get_stable_object_position(self, color, timeout=6.0):
        """使用最新缓存图像获取稳定位置，不依赖 wait_for_message"""
        start = rospy.get_time()
        stable_counter_local = 0
        stable_target_local = None
        last_positions_local = []
        rate = rospy.Rate(10)
        rospy.loginfo("开始获取 %s 物体的稳定位置（阈值=%.2f）...", color, self.fail_conf_thres)

        last_frame_id = self.latest_frame_id
        while (rospy.get_time() - start) < timeout:
            if self.latest_image is None:
                rate.sleep()
                continue
            if self.latest_frame_id == last_frame_id:
                rate.sleep()
                continue
            last_frame_id = self.latest_frame_id
            try:
                cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            except Exception as e:
                rospy.logerr("图像转换失败: %s", e)
                rate.sleep()
                continue

            h, w = cv_img.shape[:2]
            results = self.model(cv_img, conf=self.fail_conf_thres, verbose=False)
            boxes = results[0].boxes
            best, detections = self.select_best_detection(boxes, w, h, target_color=color)
            self.publish_detections(detections, w, h)
            if best:
                cls_name, center_px, matched_color = best
                if matched_color != color:
                    rospy.logdebug("检测到非目标颜色，忽略")
                    rate.sleep()
                    continue
                if stable_target_local is None:
                    stable_target_local = center_px
                    stable_counter_local = 1
                    last_positions_local = [center_px]
                    rospy.logdebug("首帧检测到: (%d,%d)", center_px[0], center_px[1])
                else:
                    prev_center = stable_target_local
                    dist = np.hypot(center_px[0] - prev_center[0], center_px[1] - prev_center[1])
                    if dist <= self.position_threshold:
                        stable_counter_local += 1
                        last_positions_local.append(center_px)
                        if len(last_positions_local) > self.stable_frames:
                            last_positions_local.pop(0)
                        avg_x = int(np.mean([p[0] for p in last_positions_local]))
                        avg_y = int(np.mean([p[1] for p in last_positions_local]))
                        stable_target_local = (avg_x, avg_y)
                        rospy.logdebug("稳定计数 %d/%d, 平均中心=(%d,%d)",
                                       stable_counter_local, self.stable_frames, avg_x, avg_y)
                        if stable_counter_local >= self.stable_frames:
                            rospy.loginfo("获取稳定坐标成功: (%d, %d)", avg_x, avg_y)
                            return (avg_x, avg_y)
                    else:
                        stable_counter_local = 1
                        stable_target_local = center_px
                        last_positions_local = [center_px]
                        rospy.logdebug("位置跳动，重置")
            else:
                rospy.logdebug("未检测到物体")
            rate.sleep()

        rospy.logwarn("超时未获取到 %s 物体的稳定位置", color)
        return None

    def image_callback(self, msg):
        """图像回调：更新缓存，同时进行正常检测"""
        self.latest_image = msg
        self.image_seq = msg.header.seq
        self.latest_frame_id += 1

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("图像转换失败: %s", e)
            return

        h, w = cv_image.shape[:2]
        self.latest_image_size = (w, h)

        results = self.model(cv_image, conf=self.conf_thres, verbose=False)
        boxes = results[0].boxes
        target_color_for_detection = self.target_color if (self.detection_enabled and self.target_color is not None) else None
        target_class_for_detection = self.target_class if (self.detection_enabled and self.target_class is not None) else None
        best_det, detections = self.select_best_detection(
            boxes,
            w,
            h,
            target_color=target_color_for_detection,
            target_class=target_class_for_detection,
        )

        self.publish_detections(detections, w, h)

        # 没有进入抓取流程时，只发布当前检测结果供 LLM/上游决策使用
        if self.is_grasping or not self.detection_enabled or self.target_color is None:
            return

        if best_det is None:
            self.empty_counter += 1
            if self.empty_counter >= 3:
                self.force_reset()
            return

        self.empty_counter = 0
        cls_name, center_px, color = best_det
        if color is None or color != self.target_color:
            self.force_reset()
            return

        stable_key = self.target_class if self.target_class else color
        if self.stable_target is None or self.stable_target[0] != stable_key:
            self.force_reset()
            self.stable_target = (stable_key, center_px)
            self.stable_target_key = stable_key
            self.stable_counter = 1
            self.last_positions = [center_px]
            rospy.loginfo("新目标跟踪: %s, 起始像素=(%d,%d)", stable_key, center_px[0], center_px[1])
            return

        prev_center = self.stable_target[1]
        dist = np.hypot(center_px[0] - prev_center[0], center_px[1] - prev_center[1])
        if dist <= self.position_threshold:
            self.stable_counter += 1
            self.last_positions.append(center_px)
            if len(self.last_positions) > self.stable_frames:
                self.last_positions.pop(0)
            avg_x = int(np.mean([p[0] for p in self.last_positions]))
            avg_y = int(np.mean([p[1] for p in self.last_positions]))
            self.stable_target = (self.stable_target_key or stable_key, (avg_x, avg_y))

            if self.stable_counter >= self.stable_frames:
                self.last_grasp_pixel = (avg_x, avg_y)
                rospy.loginfo("✅ 目标稳定: %s, 平均像素中心=(%d,%d)", color, avg_x, avg_y)
                self.is_grasping = True
                self.detection_enabled = False
                self.force_reset()
                if not self.grasp_task_running:
                    self.grasp_task_running = True
                    grasp_thread = threading.Thread(
                        target=self.trigger_grasp,
                        args=(color, avg_x, avg_y),
                        daemon=True,
                    )
                    grasp_thread.start()
        else:
            self.force_reset()
            self.stable_target = (stable_key, center_px)
            self.stable_target_key = stable_key
            self.stable_counter = 1
            self.last_positions = [center_px]

    def trigger_grasp(self, color, pixel_x, pixel_y):
        corrected_x = pixel_x + self.offset_x
        corrected_y = pixel_y + self.offset_y
        rospy.loginfo("触发抓取: %s, 原始像素=(%d,%d), 修正后=(%d,%d), 偏移=(%.1f,%.1f)",
                      color, pixel_x, pixel_y, corrected_x, corrected_y, self.offset_x, self.offset_y)

        pick_x = self.k1 * corrected_y + self.b1
        pick_y = self.k2 * corrected_x + self.b2
        rospy.loginfo("转换后抓取坐标: (%.4f, %.4f, %.4f)", pick_x, pick_y, self.pick_z)

        self.goal_pick.pos_x = pick_x
        self.goal_pick.pos_y = pick_y

        self.client.send_goal_and_wait(self.goal_pick, rospy.Duration(30.0))
        ret = self.client.get_result()
        success = (ret.result != SGRCtrlResult.PLAN_NOT_FOUND and ret.result != SGRCtrlResult.GRASP_FAILD)

        if success:
            rospy.loginfo("抓取成功")
            if color in self.box_dst:
                # 使用与参考脚本一致的 PUT_XYZ 语义到放置点，由动作服务器在到位后完成释放
                self.goal_put.pos_x = self.box_dst[color]['x']
                self.goal_put.pos_y = self.box_dst[color]['y']
                self.client.send_goal_and_wait(self.goal_put, rospy.Duration(30.0))
                rospy.loginfo("已到达放置点")
                rospy.loginfo("放置完成")

            # 放置结束后返回固定拍照点，为下一轮抓取恢复视觉位姿
            rospy.loginfo("返回拍照位置")
            self.client.send_goal_and_wait(self.goal_search, rospy.Duration(30.0))
            rospy.sleep(1.5)  # 等待稳定
            self.offset_x = 0.0
            self.offset_y = 0.0
            self.publish_offset(0, 0)
            rospy.loginfo("已重置抓取偏移，下一件物体将重新独立计算修正")
            self.publish_scene_snapshot()
            self.feedback_pub.publish(f"success:{color}")
        else:
            rospy.logwarn("抓取失败，获取失败后物体稳定坐标")
            # 失败后先回到固定拍照点，再确认物体位置，保持原有正确流程
            rospy.loginfo("失败后返回拍照位置")
            self.client.send_goal_and_wait(self.goal_search, rospy.Duration(30.0))
            rospy.sleep(1.5)  # 等待相机视野稳定
            self.publish_scene_snapshot()

            # 使用缓存图像获取稳定位置
            post_pixel = self.get_stable_object_position(color, timeout=6.0)
            if post_pixel is None:
                rospy.logwarn("抓取失败后未检测到物体，可能已消失")
                img_w, img_h = self.latest_image_size
                feedback_msg = f"failed:{color}:disappeared:-1:-1:0:0:{int(self.offset_x)}:{int(self.offset_y)}:{img_w}:{img_h}:0:0:unknown"
            else:
                delta_x = post_pixel[0] - pixel_x
                delta_y = post_pixel[1] - pixel_y
                img_w, img_h = self.latest_image_size
                center_x = img_w // 2 if img_w > 0 else -1
                center_y = img_h // 2 if img_h > 0 else -1
                if img_w > 0 and img_h > 0:
                    dist_to_center_x = post_pixel[0] - center_x
                    dist_to_center_y = post_pixel[1] - center_y
                    edge_distance = min(post_pixel[0], post_pixel[1], img_w - post_pixel[0], img_h - post_pixel[1])
                else:
                    dist_to_center_x = 0
                    dist_to_center_y = 0
                    edge_distance = -1
                feedback_msg = (
                    f"failed:{color}:{pixel_x}:{pixel_y}:{post_pixel[0]}:{post_pixel[1]}"
                    f":{delta_x}:{delta_y}:{int(self.offset_x)}:{int(self.offset_y)}"
                    f":{img_w}:{img_h}:{dist_to_center_x}:{dist_to_center_y}:{edge_distance}"
                )
                rospy.loginfo("获取到失败后物体稳定坐标: (%d, %d)", post_pixel[0], post_pixel[1])
            self.feedback_pub.publish(feedback_msg)

        # 清除状态，等待下一次指令
        self.target_color = None
        self.detection_enabled = False
        self.is_grasping = False
        self.force_reset()
        self.grasp_task_running = False
        rospy.loginfo("单次抓取周期结束")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = YOLOBasedGrasp()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被关闭")
    except KeyboardInterrupt:
        rospy.loginfo("用户中断")