#!/usr/bin/env python3
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
        # 1. 初始化
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("moveit_circle_demo", anonymous=True)

        # ===== 关键修改 1：指定命名空间和 robot_description =====
        self.ns = "/sgr532"
        self.robot_description = "/sgr532/robot_description"
        self.group_name = "sagittarius_arm"

        # 可选：先初始化 RobotCommander / Scene，方便后面排查
        self.robot = moveit_commander.RobotCommander(
            robot_description=self.robot_description, ns=self.ns
        )

        self.scene = moveit_commander.PlanningSceneInterface(ns=self.ns)

        self.arm = MoveGroupCommander(
            self.group_name, robot_description=self.robot_description, ns=self.ns
        )

        rospy.loginfo("Robot groups: %s", self.robot.get_group_names())
        rospy.loginfo("End effector link: %s", self.arm.get_end_effector_link())

        # 允许重规划
        self.arm.allow_replanning(True)

        # ===== 关键修改 2：参考坐标系别盲写 world，直接取当前规划坐标系 =====
        planning_frame = self.arm.get_planning_frame()
        rospy.loginfo("Planning frame: %s", planning_frame)
        self.arm.set_pose_reference_frame(planning_frame)

        # 设置精度
        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.001)

        # 设置速度/加速度
        self.arm.set_max_velocity_scaling_factor(0.1)
        self.arm.set_max_acceleration_scaling_factor(0.1)

        self.end_effector_link = self.arm.get_end_effector_link()

        # 3. 移动到安全起始点
        rospy.loginfo(">>> Step 1: Moving to safe start position (home)...")
        self.arm.set_named_target("home")
        ok = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

        if not ok:
            rospy.logerr("Failed to move to home position.")
            self.shutdown()
            return

        rospy.sleep(1.0)

        # 4. 获取当前位姿作为圆心参考
        current_pose = self.arm.get_current_pose(self.end_effector_link).pose
        rospy.loginfo(
            "Current pose: x=%.3f, y=%.3f, z=%.3f",
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z,
        )

        # ===== 关键修改 3：半径先改小一点，0.15 对教学机械臂偏大 =====
        circle_center = deepcopy(current_pose)
        radius = 0.05
        num_points = 30

        rospy.loginfo(
            ">>> Step 2: Generating %d waypoints for a circle with radius %.3fm...",
            num_points,
            radius,
        )

        # 5. 生成路点
        waypoints = []
        target_orientation = deepcopy(current_pose.orientation)

        # 先从当前点平滑过渡到圆的起点
        start_point = deepcopy(circle_center)
        start_point.position.x = circle_center.position.x + radius
        start_point.position.y = circle_center.position.y
        start_point.position.z = circle_center.position.z
        start_point.orientation = target_orientation
        waypoints.append(deepcopy(start_point))

        # 圆轨迹
        for i in range(num_points + 1):
            angle = 2.0 * math.pi * i / num_points
            wpose = deepcopy(circle_center)
            wpose.position.x = circle_center.position.x + radius * math.cos(angle)
            wpose.position.y = circle_center.position.y + radius * math.sin(angle)
            wpose.position.z = circle_center.position.z
            wpose.orientation = target_orientation
            waypoints.append(deepcopy(wpose))

        rospy.loginfo("Waypoints generated: %d", len(waypoints))

        # 6. 规划笛卡尔路径
        rospy.loginfo(">>> Step 3: Computing Cartesian path...")
        self.arm.set_start_state_to_current_state()

        plan, fraction = self.arm.compute_cartesian_path(
            waypoints, 0.01, 0.0, avoid_collisions=True  # eef_step  # jump_threshold
        )

        rospy.loginfo("Cartesian path fraction: %.3f", fraction)

        # 7. 执行
        if fraction > 0.90:
            rospy.loginfo(">>> Path computed successfully! Starting circular motion.")
            ok = self.arm.execute(plan, wait=True)
            self.arm.stop()
            self.arm.clear_pose_targets()

            if ok:
                rospy.loginfo(">>> Circle drawing complete!")
            else:
                rospy.logerr("Execution failed during motion.")
        else:
            rospy.logerr("Path planning failed! fraction = %.3f", fraction)
            rospy.logerr("Try smaller radius or a different start pose.")

        rospy.sleep(1.0)
        self.shutdown()

    def shutdown(self):
        rospy.loginfo("Stopping demonstration and shutting down...")
        try:
            self.arm.set_named_target("sleep")
            self.arm.go(wait=True)
            self.arm.stop()
            self.arm.clear_pose_targets()
        except Exception as e:
            rospy.logwarn("Failed to move to sleep pose: %s", str(e))

        moveit_commander.roscpp_shutdown()
        rospy.signal_shutdown("Demo finished")


if __name__ == "__main__":
    try:
        MoveItCircleDemo()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
