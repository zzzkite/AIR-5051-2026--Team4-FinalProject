#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import actionlib
import cv2
import numpy as np
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO
from sagittarius_object_color_detector.msg import SGRCtrlAction, SGRCtrlGoal

class YOLODetectWithArm:
    def __init__(self):
        rospy.init_node('yolo_detect_with_arm', anonymous=True)
        rospy.loginfo("节点启动")

        # ----- 参数读取 -----
        self.arm_name = rospy.get_param("~arm_name", "sgr532")
        self.photo_x = rospy.get_param("~photo_x", 0.20)
        self.photo_y = rospy.get_param("~photo_y", 0.00)
        self.photo_z = rospy.get_param("~photo_z", 0.16)
        self.photo_pitch = rospy.get_param("~photo_pitch", 1.55)
        self.photo_roll = rospy.get_param("~photo_roll", 0.0)
        self.photo_yaw = rospy.get_param("~photo_yaw", 0.0)

        # YOLO 模型参数
        model_path = rospy.get_param("~model_path", "/home/robotics/team_4/runs/train/yolov8_custom/weights/best.pt")
        self.conf_thres = rospy.get_param("~conf_thres", 0.5)

        # 图像话题
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.output_image_topic = rospy.get_param("~output_image_topic", "/yolo_detection/image")
        self.show_window = rospy.get_param("~show_window", True)

        # ----- 标志位：机械臂是否已就位，允许开始检测 -----
        self.detection_enabled = False

        # ----- 加载 YOLO 模型 -----
        try:
            self.model = YOLO(model_path)
            rospy.loginfo("YOLO 模型加载成功: %s", model_path)
        except Exception as e:
            rospy.logerr("模型加载失败: %s", e)
            rospy.signal_shutdown("Model load error")
            return

        # ----- 连接机械臂控制 Action -----
        action_topic = self.arm_name + '/sgr_ctrl'
        rospy.loginfo("等待 Action Server: %s ...", action_topic)
        self.client = actionlib.SimpleActionClient(action_topic, SGRCtrlAction)
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("无法连接到机械臂控制服务器！退出")
            rospy.signal_shutdown("No action server")
            return
        rospy.loginfo("已连接到机械臂控制服务器")

        # ----- 图像订阅与发布 -----
        self.bridge = CvBridge()
        self.image_received = None
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher(self.output_image_topic, Image, queue_size=1)
        rospy.loginfo("已订阅图像话题: %s", self.image_topic)
        rospy.loginfo("将发布标注图像到: %s", self.output_image_topic)

        # ----- 用于输出节流 -----
        self.last_print_time = 0
        self.last_detections = []

    def image_callback(self, msg):
        """接收图像；仅在检测启用时才运行 YOLO（避免机械臂未就绪时进行检测）"""
        # 如果机械臂尚未就位，直接返回，不做任何处理
        if not self.detection_enabled:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("图像转换失败: %s", e)
            return

        # 运行检测，verbose=False 抑制 YOLO 自带的打印信息
        results = self.model(cv_image, conf=self.conf_thres, verbose=False)

        # 获取图像尺寸（用于转换为像素坐标）
        h, w = cv_image.shape[:2]

        # 解析检测结果
        boxes = results[0].boxes
        detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id]

                # 获取归一化的边界框中心点坐标和宽高 (xywhn)
                x_center_norm, y_center_norm, width_norm, height_norm = box.xywhn[0].tolist()
                # 像素坐标
                x_center_px = int(x_center_norm * w)
                y_center_px = int(y_center_norm * h)
                width_px = int(width_norm * w)
                height_px = int(height_norm * h)
                # 左上角和右下角像素坐标
                x1 = int((x_center_norm - width_norm/2) * w)
                y1 = int((y_center_norm - height_norm/2) * h)
                x2 = int((x_center_norm + width_norm/2) * w)
                y2 = int((y_center_norm + height_norm/2) * h)

                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'norm_center': (x_center_norm, y_center_norm),
                    'norm_size': (width_norm, height_norm),
                    'pixel_center': (x_center_px, y_center_px),
                    'pixel_box': (x1, y1, x2, y2)
                })

        self.last_detections = detections

        # 每秒打印一次检测结果
        now = time.time()
        if now - self.last_print_time >= 1.0:
            self.last_print_time = now
            self.print_detections()

        # 获取带标注的图像并发布
        annotated = results[0].plot()
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            self.image_pub.publish(out_msg)
        except Exception as e:
            rospy.logerr("发布标注图像失败: %s", e)

        if self.show_window:
            cv2.imshow("YOLO Detection", annotated)
            cv2.waitKey(1)

    def print_detections(self):
        """格式化打印检测结果（含坐标信息）"""
        if not self.last_detections:
            rospy.loginfo("当前未检测到任何目标")
            return

        rospy.loginfo("========== YOLO 检测结果 ==========")
        for i, det in enumerate(self.last_detections, 1):
            rospy.loginfo("目标 %d: 类别=%s, 置信度=%.2f", i, det['class'], det['confidence'])
            rospy.loginfo("  归一化坐标 (中心点, 宽高): (%.3f, %.3f, %.3f, %.3f)",
                          det['norm_center'][0], det['norm_center'][1],
                          det['norm_size'][0], det['norm_size'][1])
            rospy.loginfo("  像素坐标  (中心点): (%d, %d)", det['pixel_center'][0], det['pixel_center'][1])
            rospy.loginfo("  边界框    (左上,右下): (%d, %d) -> (%d, %d)",
                          det['pixel_box'][0], det['pixel_box'][1],
                          det['pixel_box'][2], det['pixel_box'][3])
        rospy.loginfo("====================================")

    def move_to_photo_pose(self):
        """控制机械臂移动到预设拍照位置（垂直向下）"""
        goal = SGRCtrlGoal()
        goal.action_type = 1   # ACTION_TYPE_XYZ_RPY
        goal.pos_x = self.photo_x
        goal.pos_y = self.photo_y
        goal.pos_z = self.photo_z
        goal.pos_roll = self.photo_roll
        goal.pos_pitch = self.photo_pitch
        goal.pos_yaw = self.photo_yaw
        goal.grasp_type = 0

        rospy.loginfo("发送目标位姿: (%.3f, %.3f, %.3f) 俯仰=%.3f",
                      goal.pos_x, goal.pos_y, goal.pos_z, goal.pos_pitch)
        self.client.send_goal_and_wait(goal, rospy.Duration(30.0))
        state = self.client.get_state()
        rospy.loginfo("Action 状态码: %d (3=SUCCEEDED)", state)
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("机械臂已到达目标位姿")
            return True
        else:
            rospy.logerr("移动失败，状态码: %d", state)
            return False

    def run(self):
        """主流程：先移动机械臂，然后启用检测"""
        rospy.sleep(1.0)
        if not self.move_to_photo_pose():
            rospy.logerr("机械臂未到达拍照位置，退出")
            return

        # 机械臂就位，允许开始检测
        self.detection_enabled = True
        rospy.loginfo("机械臂已就位，开始 YOLO 实时检测（终端输出每1秒刷新一次）...")
        rospy.spin()

        if self.show_window:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = YOLODetectWithArm()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被关闭")
    except KeyboardInterrupt:
        rospy.loginfo("用户中断")