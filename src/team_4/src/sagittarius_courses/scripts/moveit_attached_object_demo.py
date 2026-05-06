#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys
import copy
import moveit_commander
from moveit_commander import RobotCommander, MoveGroupCommander, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject, PlanningScene
from math import radians
from copy import deepcopy

class MoveAttachedObjectDemo:
    def __init__(self):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)

        # # 初始化ROS节点
        # rospy.init_node('moveit_attached_object_demo', anonymous=True)


        # 修改：先初始化节点，再处理参数重映射
        rospy.init_node('moveit_attached_object_demo', anonymous=True)
        
                # ===== 重要修改：设置命名空间 =====
        # 设置move_group的命名空间
        move_group_namespace = '/sgr532'
        
        # 复制robot_description参数
        if not rospy.get_param('robot_description', None):
            robot_desc = rospy.get_param(f'{move_group_namespace}/robot_description', None)
            if robot_desc:
                rospy.set_param('robot_description', robot_desc)
                rospy.loginfo("Copied robot_description to root namespace")
        
        # 创建MoveGroupCommander，指定正确的命名空间
        arm = MoveGroupCommander(
            'sagittarius_arm', 
            robot_description=f'{move_group_namespace}/robot_description',
            ns=move_group_namespace  # 指定命名空间
        )

        
        # 初始化场景对象
        scene = PlanningSceneInterface()
        rospy.sleep(1)
                                
        # # 初始化需要使用move group控制的机械臂中的arm group
        # arm = MoveGroupCommander('sagittarius_arm')
        
        # 获取终端link的名称
        end_effector_link = arm.get_end_effector_link()
        
        # 设置机械臂运动的允许误差值
        arm.set_goal_joint_tolerance(0.001)

        # 设置允许的最大速度和加速度
        arm.set_max_acceleration_scaling_factor(0.5)
        arm.set_max_velocity_scaling_factor(0.5)

        # 控制机械臂回到初始化位置
        arm.set_named_target('home')
        arm.go()
        rospy.sleep(1)

        # 设置每次运动规划的时间限制：10s
        arm.set_planning_time(10)
        
        # 移除场景中之前运行残留的物体
        scene.remove_attached_object(end_effector_link, 'tool')
        scene.remove_world_object('table') 

        # 设置桌面的高度
        table_ground = 0.45
        
        # 设置table和tool的三维尺寸
        table_size = [0.1, 0.3, 0.02]
        tool_size = [0.2, 0.02, 0.02]
        
        # 设置tool的位姿
        p = PoseStamped()
        p.header.frame_id = end_effector_link
        
        p.pose.position.x = tool_size[0] - 0.03
        p.pose.position.y = 0
        p.pose.position.z = -0.015
        p.pose.orientation.w = 1
        
        # 将tool附着到机器人的终端
        scene.attach_box(end_effector_link, 'tool', p, tool_size)

        # 将table加入场景当中
        table_pose = PoseStamped()
        table_pose.header.frame_id = 'world'
        table_pose.pose.position.x = 0.4
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = table_ground + table_size[2] / 2.0
        table_pose.pose.orientation.w = 1.0
        scene.add_box('table', table_pose, table_size)
        
        rospy.sleep(2)

        # 更新当前的位姿
        arm.set_start_state_to_current_state()

        # 设置机械臂的目标位置，使用六轴的位置数据进行描述（单位：弧度）
        joint_positions = [-0.05061454830783556, 0.8534660042252271, 0.609119908946021,
                    0.20245819323134223, -0.9285151620609834, -0.03839724354387525]
        arm.set_joint_value_target(joint_positions)
        
        # 控制机械臂完成运动
        arm.go()
        rospy.sleep(1)
        
        # 控制机械臂回到初始化位置
        arm.set_named_target('sleep')
        arm.go()
        rospy.sleep(1)

        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

if __name__ == "__main__":
    MoveAttachedObjectDemo()
