#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys
import moveit_commander

class MoveItGripperDemo:
    def __init__(self):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)

        # # 初始化ROS节点
        # rospy.init_node('moveit_gripper_demo', anonymous=True)

        # 修改：先初始化节点，再处理参数重映射
        rospy.init_node('moveit_gripper_demo', anonymous=True)
        
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
        gripper = moveit_commander.MoveGroupCommander(
            'sagittarius_gripper', 
            robot_description=f'{move_group_namespace}/robot_description',
            ns=move_group_namespace  # 指定命名空间
        )

        # # 初始化需要使用move group控制的夹爪的group
        # gripper = moveit_commander.MoveGroupCommander("sagittarius_gripper")
        
        # 设置夹爪运动的允许误差值
        gripper.set_goal_joint_tolerance(0.001)

        # 设置允许的最大速度和加速度
        gripper.set_max_acceleration_scaling_factor(0.5)
        gripper.set_max_velocity_scaling_factor(0.5)

        # 控制夹爪打开
        gripper.set_named_target('open')
        gripper.go()
        rospy.sleep(2)

        # 控制夹爪闭合
        gripper.set_named_target('close')
        gripper.go()
        rospy.sleep(2)

        # 设置夹爪的目标位置，使用两个关节的位置数据进行描述（单位：弧度）
        joint_positions = [-0.022, -0.022]
        gripper.set_joint_value_target(joint_positions)

        # 控制夹爪完成运动
        gripper.go()
        rospy.sleep(2)

        # 控制夹爪先回到初始化位置
        gripper.set_named_target('open')
        gripper.go()
        rospy.sleep(1)
        
        # 关闭并退出moveit
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

if __name__ == "__main__":
    try:
        MoveItGripperDemo()
    except rospy.ROSInterruptException:
        pass
