#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose

class MoveItIkDemo:
    def __init__(self):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)
        
        # # 初始化ROS节点
        # rospy.init_node('moveit_ik_demo')
                
        # # 初始化需要使用move group控制的机械臂中的arm group
        # arm = moveit_commander.MoveGroupCommander("sagittarius_arm")

        # 修改：先初始化节点，再处理参数重映射
        rospy.init_node('moveit_ik_demo', anonymous=True)
        
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
        arm = moveit_commander.MoveGroupCommander(
            'sagittarius_arm', 
            robot_description=f'{move_group_namespace}/robot_description',
            ns=move_group_namespace  # 指定命名空间
        )
                
        # 获取终端link的名称
        end_effector_link = arm.get_end_effector_link()
                        
        # 设置目标位置所使用的参考坐标系
        reference_frame = 'world'
        arm.set_pose_reference_frame(reference_frame)
                
        # 当运动规划失败后，允许重新规划
        arm.allow_replanning(True)
        
        # 设置位置(单位：米)和姿态（单位：弧度）的允许误差
        arm.set_goal_position_tolerance(0.001)
        arm.set_goal_orientation_tolerance(0.001)
       
        # 设置允许的最大速度和加速度
        arm.set_max_acceleration_scaling_factor(0.5)
        arm.set_max_velocity_scaling_factor(0.5)

        # 控制机械臂先回到初始化位置
        arm.set_named_target('home')
        arm.go()
        rospy.sleep(1)
               
        # 设置机械臂工作空间中的目标位姿，位置使用x、y、z坐标描述，
        # 姿态使用四元数描述，基于base_link坐标系
        target_pose = PoseStamped()
        target_pose.header.frame_id = reference_frame
        target_pose.header.stamp = rospy.Time.now() 
        target_pose.pose.position.x = 0.2472990396168796
        target_pose.pose.position.y = 0.0006590926103004068
        target_pose.pose.position.z = 0.3456034504080325
        target_pose.pose.orientation.w = 1.0

        # 设置机器臂当前的状态作为运动初始状态
        arm.set_start_state_to_current_state()
        
        # 设置机械臂终端运动的目标位姿
        arm.set_pose_target(target_pose, end_effector_link)
        
        # 规划运动路径
        #traj = arm.plan()
        plan_success, traj, planning_time, error_code = arm.plan()

        # 按照规划的运动路径控制机械臂运动
        arm.execute(traj)
        rospy.sleep(1)

        # 控制机械臂回到初始化位置
        arm.set_named_target('sleep')
        arm.go()

        # 关闭并退出moveit
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

if __name__ == "__main__":
    MoveItIkDemo()
