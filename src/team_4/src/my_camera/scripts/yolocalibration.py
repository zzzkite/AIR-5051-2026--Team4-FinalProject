#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO 手眼标定程序（交互式终端版）
==================================
功能：
  1. 控制机械臂依次移动到 5 个预设位置。
  2. 在每个位置等待用户将标定物（如蓝色方块）放在机械臂末端正下方的桌面上。
  3. 用户按 Enter 确认后，机械臂移开，YOLO 自动检测物体中心像素坐标并记录。
  4. 自动拟合线性回归，将像素坐标 (u, v) 映射到机械臂基座坐标 (x, y)。
  5. 将拟合参数保存到 vision_config.yaml 中，供 yolo_based_grasp.py 使用。

使用方法：
  1. 启动此节点（rosrun 或 roslaunch）。
  2. 观察终端提示，按 Enter 继续每一步。
  3. 完成后参数自动保存，节点退出。
"""

import rospy
import actionlib
import cv2
import numpy as np
np.float = float   # 解决新版 NumPy 兼容问题
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO
from sagittarius_object_color_detector.msg import SGRCtrlAction, SGRCtrlGoal

class YoloCalibration:
    def __init__(self):
        rospy.init_node('yolo_calibration', anonymous=True)
        rospy.loginfo("=== YOLO 手眼标定程序启动 ===")

        # ------------------- 参数读取 -------------------
        self.arm_name = rospy.get_param("~arm_name", "sgr532")
        self.model_path = rospy.get_param("~model_path", "")
        self.conf_thres = rospy.get_param("~conf_thres", 0.6)
        self.save_path = rospy.get_param("~vision_config", "")

        # 5 个标定点的机械臂坐标 (x, y)  [z 固定 0.06]
        self.calib_points = [
            (0.25, 0.00),
            (0.225, 0.025),
            (0.275, 0.025),
            (0.275, -0.025),
            (0.225, -0.025)
        ]

        # 存储采集到的像素坐标 (u, v)
        self.pixel_u = []   # 对应图像中的列坐标 (x)
        self.pixel_v = []   # 对应图像中的行坐标 (y)
        self.robot_x = []   # 机械臂基座 X
        self.robot_y = []   # 机械臂基座 Y

        # YOLO 检测相关
        self.bridge = CvBridge()
        self.latest_center = None   # (u, v) 像素坐标
        self.detection_ready = False
        self.model = None

        # 机械臂 Action 客户端
        action_topic = self.arm_name + '/sgr_ctrl'
        rospy.loginfo("等待机械臂控制服务器: %s", action_topic)
        self.client = actionlib.SimpleActionClient(action_topic, SGRCtrlAction)
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("无法连接到机械臂控制服务器，请检查 bringup 是否启动")
            rospy.signal_shutdown("No action server")
            return
        rospy.loginfo("机械臂控制服务器已连接")

        # ---------- 加载 YOLO 模型 ----------
        if not self.model_path:
            rospy.logerr("未提供 YOLO 模型路径，请设置参数 ~model_path")
            rospy.signal_shutdown("No model")
            return
        try:
            self.model = YOLO(self.model_path)
            rospy.loginfo("YOLO 模型加载成功: %s", self.model_path)
        except Exception as e:
            rospy.logerr("YOLO 模型加载失败: %s", e)
            rospy.signal_shutdown("Model error")
            return

        # ---------- 图像订阅 ----------
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        rospy.loginfo("已订阅相机话题 /usb_cam/image_raw")

        # ---------- 控制发布（用于通知标定流程）----------
        self.arm_cmd_pub = rospy.Publisher('cali_arm_cmd_topic', String, queue_size=5)

        # ---------- 开始交互 ----------
        rospy.sleep(1.0)   # 等待订阅建立

    def image_callback(self, msg):
        """YOLO 检测图像回调，更新 latest_center"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("图像转换失败: %s", e)
            return

        # 运行检测（抑制控制台输出）
        results = self.model(cv_image, conf=self.conf_thres, verbose=False)
        h, w = cv_image.shape[:2]
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            # 取置信度最高的一个
            best_conf = 0
            best_center = None
            for box in boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    xc_norm, yc_norm = box.xywhn[0][:2].tolist()
                    xc_px = int(xc_norm * w)
                    yc_px = int(yc_norm * h)
                    best_center = (xc_px, yc_px)
                    best_conf = conf
            if best_center:
                self.latest_center = best_center
                self.detection_ready = True
        else:
            self.detection_ready = False

    def wait_key(self, message):
        """打印提示并等待用户按 Enter 继续"""
        input("\n👉 " + message + " 按 Enter 继续...")

    def move_to_point(self, idx):
        """控制机械臂移动到第 idx 个标定点（位置）"""
        x, y = self.calib_points[idx]
        goal = SGRCtrlGoal()
        goal.action_type = goal.ACTION_TYPE_XYZ_RPY
        goal.pos_x = x
        goal.pos_y = y
        goal.pos_z = 0.06          # 固定标定高度
        goal.pos_pitch = 1.57      # 垂直向下
        goal.grasp_type = goal.GRASP_OPEN
        rospy.loginfo("移动机械臂到标定点 %d: (%.3f, %.3f)", idx+1, x, y)
        self.client.send_goal_and_wait(goal, rospy.Duration(30.0))
        if self.client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.logerr("机械臂移动失败！")
            return False
        return True

    def move_to_clear_pose(self):
        """机械臂移动到不遮挡相机的位置（侧上方）"""
        goal = SGRCtrlGoal()
        goal.action_type = goal.ACTION_TYPE_XYZ_RPY
        goal.pos_x = 0.2
        goal.pos_y = 0.0
        goal.pos_z = 0.15
        goal.pos_pitch = 1.57
        goal.grasp_type = goal.GRASP_OPEN
        rospy.loginfo("机械臂移开，准备拍照...")
        self.client.send_goal_and_wait(goal, rospy.Duration(30.0))
        return self.client.get_state() == actionlib.GoalStatus.SUCCEEDED

    def capture_pixel_center(self):
        """触发一次检测，等待获取有效的像素中心（超时 5 秒）"""
        rospy.loginfo("等待 YOLO 检测物体...")
        # 短暂等待，让相机捕捉稳定画面
        rospy.sleep(1.0)
        self.detection_ready = False
        start_time = rospy.get_time()
        timeout = 5.0
        while (rospy.get_time() - start_time) < timeout:
            if self.detection_ready and self.latest_center is not None:
                u, v = self.latest_center
                rospy.loginfo("检测成功！像素中心 = (u=%d, v=%d)", u, v)
                return (u, v)
            rospy.sleep(0.1)
        rospy.logerr("超时未检测到物体，请检查 YOLO 模型或物体是否在视野内")
        return None

    def run_calibration(self):
        """执行完整的标定流程"""
        rospy.loginfo("\n========== 开始 YOLO 手眼标定 ==========")
        self.wait_key("请确保标定物（如蓝色方块）已准备好，机械臂将开始移动。")

        for i in range(len(self.calib_points)):
            rospy.loginfo("\n--- 第 %d / %d 个标定点 ---", i+1, len(self.calib_points))

            # 1. 移动机械臂到标定点
            if not self.move_to_point(i):
                rospy.logerr("标定失败：无法移动到点 %d", i+1)
                return

            # 2. 等待用户放置物体并按 Enter
            self.wait_key(f"请将标定物放在机械臂末端正下方的桌面上，然后")

            # 3. 机械臂移开，露出相机视野
            if not self.move_to_clear_pose():
                rospy.logerr("机械臂移开失败")
                return

            # 4. 触发相机检测，获取像素中心
            pixel = self.capture_pixel_center()
            if pixel is None:
                rospy.logerr("标定点 %d 检测失败，请检查物体位置或光照，按 Enter 重试此点", i+1)
                input("按 Enter 重试当前点...")
                # 重试当前点（回退索引）
                continue

            # 5. 记录数据
            x, y = self.calib_points[i]
            self.robot_x.append(x)
            self.robot_y.append(y)
            self.pixel_u.append(pixel[0])   # u = 列坐标
            self.pixel_v.append(pixel[1])   # v = 行坐标
            rospy.loginfo("已记录: 机械臂(%.3f,%.3f) <-> 像素(%d,%d)", x, y, pixel[0], pixel[1])

            # 如果不是最后一个点，等待用户确认继续下一个点
            if i < len(self.calib_points) - 1:
                self.wait_key("数据已记录，请移走物体，然后")
        # 所有点采集完成
        rospy.loginfo("\n========== 数据采集完成，开始计算线性回归 ==========")
        self.compute_and_save()

    def compute_and_save(self):
        """拟合线性回归并保存到 yaml 文件"""
        if len(self.robot_x) < 3:
            rospy.logerr("有效数据点不足，需要至少 3 个点")
            return

        # 数据准备
        robot_x = np.array(self.robot_x).reshape(-1, 1)
        robot_y = np.array(self.robot_y).reshape(-1, 1)
        pixel_v = np.array(self.pixel_v).reshape(-1, 1)   # 机械臂 X 对应像素的 Y (行)
        pixel_u = np.array(self.pixel_u).reshape(-1, 1)   # 机械臂 Y 对应像素的 X (列)

        # 线性回归
        from sklearn.linear_model import LinearRegression
        reg_x = LinearRegression().fit(pixel_v, robot_x)   # x = k1 * v + b1
        reg_y = LinearRegression().fit(pixel_u, robot_y)   # y = k2 * u + b2

        k1 = reg_x.coef_[0][0]
        b1 = reg_x.intercept_[0]
        k2 = reg_y.coef_[0][0]
        b2 = reg_y.intercept_[0]

        rospy.loginfo("\n========== 线性回归结果 ==========")
        rospy.loginfo("机械臂 X = %.5f * 像素 V + %.5f", k1, b1)
        rospy.loginfo("机械臂 Y = %.5f * 像素 U + %.5f", k2, b2)
        rospy.loginfo("=================================")

        # 保存到配置文件
        if not self.save_path:
            rospy.logwarn("未提供 vision_config 保存路径，参数将打印但不会保存")
            return

        try:
            # 尝试读取现有配置，如果没有则创建
            try:
                with open(self.save_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
            except:
                cfg = {}
            if 'LinearRegression' not in cfg:
                cfg['LinearRegression'] = {}
            cfg['LinearRegression']['k1'] = float(k1)
            cfg['LinearRegression']['b1'] = float(b1)
            cfg['LinearRegression']['k2'] = float(k2)
            cfg['LinearRegression']['b2'] = float(b2)
            with open(self.save_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
            rospy.loginfo("标定参数已成功保存至: %s", self.save_path)
        except Exception as e:
            rospy.logerr("保存配置文件失败: %s", e)

        rospy.loginfo("\n🎉 标定完成！现在可以运行 yolo_based_grasp.py 进行抓取。")

    def run(self):
        self.run_calibration()
        rospy.loginfo("节点将在 2 秒后退出")
        rospy.sleep(2)
        rospy.signal_shutdown("Calibration finished")

if __name__ == '__main__':
    try:
        node = YoloCalibration()
        node.run()
    except rospy.ROSInterruptException:
        pass