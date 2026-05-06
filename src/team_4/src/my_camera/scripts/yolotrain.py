#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import rospy
from ultralytics import YOLO

class YOLOTrainNode:
    def __init__(self):
        rospy.init_node('yolo_train_node', anonymous=True)
        rospy.loginfo("YOLO 训练节点启动")

        # ----- 训练参数（可通过 rosparam 配置）-----
        self.data_yaml = rospy.get_param("~data_yaml", "/home/robotics/team_4/dataset/data.yaml")
        self.epochs = rospy.get_param("~epochs", 100)
        self.imgsz = rospy.get_param("~imgsz", 640)
        self.batch = rospy.get_param("~batch", 16)
        self.device = rospy.get_param("~device", "cpu")
        self.workers = rospy.get_param("~workers", 8)
        self.lr0 = rospy.get_param("~lr0", 0.01)

        # ----- 修改点：模型保存路径设置为 ~/team_4/runs/train -----
        default_project = os.path.expanduser("~/team_4/runs/train")
        self.project = rospy.get_param("~project", default_project)
        self.name = rospy.get_param("~name", "yolov8_custom")

        # 确保保存目录存在（YOLO 会自动创建，这里只是提前检查父目录）
        os.makedirs(self.project, exist_ok=True)

        # ----- 检查数据集配置文件 -----
        if not os.path.exists(self.data_yaml):
            rospy.logerr("数据集配置文件不存在: %s", self.data_yaml)
            rospy.signal_shutdown("Missing data.yaml")
            return

        rospy.loginfo("数据集配置文件: %s", self.data_yaml)
        rospy.loginfo("训练参数: epochs=%d, imgsz=%d, batch=%d, device=%s",
                      self.epochs, self.imgsz, self.batch, self.device)
        rospy.loginfo("模型将保存到: %s/%s/weights", self.project, self.name)

        # ----- 加载预训练模型 -----
        model = YOLO('yolov8n.pt')
        rospy.loginfo("已加载预训练模型 yolov8n.pt")

        # ----- 开始训练 -----
        rospy.loginfo("开始训练... (这可能需要几小时，请耐心等待)")
        try:
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                device=self.device,
                workers=self.workers,
                lr0=self.lr0,
                project=self.project,   # 使用新路径
                name=self.name,
                exist_ok=True,
                verbose=True
            )
            rospy.loginfo("训练完成！模型保存在: %s/%s/weights", self.project, self.name)
        except Exception as e:
            rospy.logerr("训练过程中出错: %s", str(e))
            sys.exit(1)

if __name__ == '__main__':
    try:
        node = YOLOTrainNode()
        rospy.signal_shutdown("Training finished")
    except rospy.ROSInterruptException:
        rospy.loginfo("训练被用户中断")