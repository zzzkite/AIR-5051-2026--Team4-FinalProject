#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sys
import moveit_commander
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import Pose
from copy import deepcopy
import math

class MoveItCircleDemo:
    def __init__(self):
        # 1. 初始化 move_group 接口
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('moveit_circle_demo', anonymous=True)
        
        # 2. 初始化机械臂组 (请根据实际名称修改 'sagittarius_arm')
        self.arm = MoveGroupCommander('sagittarius_arm')
        
        # 允许重规划
        self.arm.allow_replanning(True)
        self.arm.set_pose_reference_frame('world')
        
        # 设置精度
        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.001)
        
        # 设置速度/加速度限制 (画圆建议稍微慢一点以保证平滑)
        self.arm.set_max_velocity_scaling_factor(0.3)
        self.arm.set_max_acceleration_scaling_factor(0.3)
        
        self.end_effector_link = self.arm.get_end_effector_link()

        # 3. 移动到安全起始点
        rospy.loginfo(">>> Step 1: Moving to safe start position (home)...")
        self.arm.set_named_target('home')
        if not self.arm.go():
            rospy.logerr("Failed to move to home position.")
            self.shutdown()
            return
        rospy.sleep(1.0)

        # 4. 获取当前位姿作为圆的中心参考
        current_pose = self.arm.get_current_pose(self.end_effector_link).pose
        rospy.loginfo(f"Current pose acquired at: x={current_pose.position.x:.3f}, y={current_pose.position.y:.3f}, z={current_pose.position.z:.3f}")

        # --- 配置画圆参数 ---
        circle_center = deepcopy(current_pose)
        # 可选：如果你想让圆心不在当前点，而是在当前点前方/上方，可以在此修改
        # circle_center.position.x += 0.1 
        
        radius = 0.15  # 圆的半径 (米)，请根据机械臂工作空间调整，不要太大
        num_points = 40 # 圆被分成多少个点，越多越平滑，但规划时间越长
        
        rospy.loginfo(f">>> Step 2: Generating {num_points} waypoints for a circle with radius {radius}m...")

        # 5. 计算路点列表
        waypoints = []
        
        # 保存初始姿态 (画圆时通常保持姿态不变)
        target_orientation = deepcopy(current_pose.orientation)
        
        for i in range(num_points + 1): # +1 是为了让终点和起点重合，形成闭环
            angle = 2 * math.pi * i / num_points
            
            wpose = deepcopy(circle_center)
            
            # 使用三角函数计算圆上的点 (在 XY 平面画圆)
            # 如果需要在这个平面画圆，修改下面的 x, y 计算公式
            wpose.position.x = circle_center.position.x + radius * math.cos(angle)
            wpose.position.y = circle_center.position.y + radius * math.sin(angle)
            wpose.position.z = circle_center.position.z # 保持高度不变
            
            # 保持姿态不变 (非常重要，否则机械臂会在画圆时乱转)
            wpose.orientation = target_orientation
            
            waypoints.append(deepcopy(wpose))

        rospy.loginfo(f"Waypoints generated. Total points: {len(waypoints)}")

        # 6. 调用路径规划 API
        rospy.loginfo(">>> Step 3: Computing Cartesian path...")
        
        fraction = 0.0
        max_tries = 50
        attempts = 0
        
        # 设置起始状态为当前状态
        self.arm.set_start_state_to_current_state()

        while fraction < 1.0 and attempts < max_tries:
            (plan, fraction) = self.arm.compute_cartesian_path(
                waypoints,
                0.01,       # eef_step: 终端步进值 (米)，越小越精确但计算越慢
                0.0,        # jump_threshold: 跳跃阈值，0.0 表示不允许跳跃
                avoid_collisions=True
            )
            attempts += 1
            if attempts % 10 == 0:
                rospy.loginfo(f"Still trying to plan circle... ({attempts}/{max_tries})")

        # 7. 执行运动
        if fraction == 1.0:
            rospy.loginfo(">>> Path computed successfully! Starting circular motion.")
            result = self.arm.execute(plan)
            if result:
                rospy.loginfo(">>> Circle drawing complete!")
            else:
                rospy.logerr("Execution failed during motion.")
        else:
            rospy.logerr(f"Path planning failed! Only achieved {fraction*100:.1f}% of the path after {attempts} attempts.")
            rospy.logerr("Possible reasons: Radius too large, singularities, or collision.")

        rospy.sleep(2.0)

        # 8. 结束清理
        self.shutdown()

    def shutdown(self):
        rospy.loginfo("Stopping demonstration and shutting down...")
        # 可以选择回到 sleep 姿态
        try:
            self.arm.set_named_target('sleep')
            self.arm.go()
        except:
            pass
        
        moveit_commander.roscpp_shutdown()
        rospy.signal_shutdown("Demo finished")
        # moveit_commander.os._exit(0)

if __name__ == "__main__":
    try:
        MoveItCircleDemo()
        # 保持节点运行直到完成
        rospy.spin()
    except rospy.ROSInterruptException:
        pass