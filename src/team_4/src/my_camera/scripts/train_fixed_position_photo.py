#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import actionlib
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sagittarius_object_color_detector.msg import SGRCtrlAction, SGRCtrlGoal

class FixedPhotoCapture:
    def __init__(self):
        rospy.init_node('fixed_photo_capture', anonymous=True)
        rospy.loginfo("节点启动")

        # ----- 参数读取 -----
        self.arm_name = rospy.get_param("~arm_name", "sgr532")
        self.photo_x = rospy.get_param("~photo_x", 0.20)
        self.photo_y = rospy.get_param("~photo_y", 0.00)
        self.photo_z = rospy.get_param("~photo_z", 0.16)
        self.photo_pitch = rospy.get_param("~photo_pitch", 1.55)   # 手动测试可行的俯仰角
        self.photo_roll = rospy.get_param("~photo_roll", 0.0)
        self.photo_yaw = rospy.get_param("~photo_yaw", 0.0)

        # ----- 固定保存路径（YOLO 训练目录）-----
        self.save_dir = rospy.get_param("~save_dir", "/home/robotics/team_4/dataset/train/images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            rospy.loginfo("创建目录: %s", self.save_dir)
        else:
            rospy.loginfo("使用已有目录: %s", self.save_dir)

        # 拍照间隔（秒）
        self.interval = rospy.get_param("~interval", 5.0)

        # ----- 连接机械臂控制 Action -----
        action_topic = self.arm_name + '/sgr_ctrl'
        rospy.loginfo("等待 Action Server: %s ...", action_topic)
        self.client = actionlib.SimpleActionClient(action_topic, SGRCtrlAction)
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("无法连接到机械臂控制服务器！退出")
            rospy.signal_shutdown("No action server")
            return
        rospy.loginfo("已连接到机械臂控制服务器")

        # ----- 订阅图像 -----
        self.bridge = CvBridge()
        self.image_received = None
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        rospy.loginfo("已订阅 /usb_cam/image_raw")

    def image_callback(self, msg):
        try:
            self.image_received = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("图像转换失败: %s", e)

    def move_to_photo_pose(self):
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

    def get_next_filename(self):
        """返回下一个可用的文件名（自动递增，避免覆盖）"""
        existing = [f for f in os.listdir(self.save_dir) if f.endswith('.jpg')]
        if not existing:
            return "0001.jpg"
        numbers = []
        for f in existing:
            base = f.split('.')[0]
            if base.isdigit():
                numbers.append(int(base))
        if not numbers:
            return "0001.jpg"
        next_num = max(numbers) + 1
        return f"{next_num:04d}.jpg"

    def capture_and_save(self):
        """捕获最新图像并保存，返回是否成功"""
        if self.image_received is None:
            rospy.logwarn("当前无图像数据，跳过本次拍照")
            return False

        filename = self.get_next_filename()
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, self.image_received)
        rospy.loginfo("照片已保存: %s", filepath)
        return True

    def run(self):
        rospy.sleep(1.0)   # 等待订阅建立
        if not self.move_to_photo_pose():
            rospy.logerr("机械臂未到达拍照位置，退出")
            return

        rospy.loginfo("开始自动拍照，间隔 %.1f 秒（按 Ctrl+C 停止）", self.interval)
        rate = rospy.Rate(1.0 / self.interval)
        while not rospy.is_shutdown():
            self.capture_and_save()
            rate.sleep()

if __name__ == '__main__':
    try:
        node = FixedPhotoCapture()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("用户中断，退出拍照")
    except KeyboardInterrupt:
        rospy.loginfo("用户中断，退出拍照")