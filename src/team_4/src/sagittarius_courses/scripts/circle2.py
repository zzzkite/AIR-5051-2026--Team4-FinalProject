#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import rospy
import moveit_commander
from moveit_commander import MoveGroupCommander
from copy import deepcopy
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_commander import MoveGroupCommander

class MoveItCircleDemo:
    def __init__(self):
        # 1. 初始化
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("moveit_circle_demo", anonymous=True)

        # 2. 命名空间和机器人描述参数
        self.ns = "/sgr532"
        self.robot_description = "/sgr532/robot_description"
        self.group_name = "sagittarius_arm"

        # 3. 初始化 MoveIt 接口
        self.robot = moveit_commander.RobotCommander(
            robot_description=self.robot_description, ns=self.ns
        )

        self.scene = moveit_commander.PlanningSceneInterface(ns=self.ns)

        self.arm = MoveGroupCommander(
            self.group_name, robot_description=self.robot_description, ns=self.ns
        )

        rospy.loginfo("Robot groups: %s", self.robot.get_group_names())
        rospy.loginfo("End effector link: %s", self.arm.get_end_effector_link())

        # 4. 基本参数
        self.arm.allow_replanning(True)

        planning_frame = self.arm.get_planning_frame()
        rospy.loginfo("Planning frame: %s", planning_frame)
        self.arm.set_pose_reference_frame(planning_frame)

        self.arm.set_goal_position_tolerance(0.002)
        self.arm.set_goal_orientation_tolerance(0.002)

        # 速度调慢，避免 CONTROL_FAILED
        self.arm.set_max_velocity_scaling_factor(0.03)
        self.arm.set_max_acceleration_scaling_factor(0.03)

        self.end_effector_link = self.arm.get_end_effector_link()

        # 5. 先同步当前状态
        rospy.loginfo(">>> Step 0: Sync current robot state...")
        self.arm.set_start_state_to_current_state()
        rospy.sleep(1.0)

        # 6. 尝试移动到 home，但失败也不中断
        rospy.loginfo(">>> Step 1: Try moving to safe start position (home)...")
        moved_home = self.move_to_named_target("home")

        if moved_home:
            rospy.loginfo("Moved to home successfully.")
            rospy.sleep(1.0)
        else:
            rospy.logwarn("Failed to move to home. Continue from current pose.")

        # 7. 获取当前位姿
        current_pose = self.arm.get_current_pose(self.end_effector_link).pose
        rospy.loginfo(
            "Current pose: x=%.3f, y=%.3f, z=%.3f",
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z,
        )

        # 8. 设置画圆参数
        circle_center = deepcopy(current_pose)
        radius = 0.05
        num_points = 20

        rospy.loginfo(
            ">>> Step 2: Generating %d waypoints for a circle with radius %.3fm...",
            num_points,
            radius,
        )

        # 9. 生成路点
        waypoints = []
        target_orientation = deepcopy(current_pose.orientation)

        # 先平滑到圆起点
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

        # 10. 规划笛卡尔路径
        rospy.loginfo(">>> Step 3: Computing Cartesian path...")
        self.arm.set_start_state_to_current_state()
        rospy.sleep(0.5)

        plan, fraction = self.arm.compute_cartesian_path(
            waypoints, 0.01, 0.0, avoid_collisions=True  # eef_step  # jump_threshold
        )

        rospy.loginfo("Cartesian path fraction: %.3f", fraction)

        # 11. 执行轨迹
        if fraction > 0.90:
            rospy.loginfo(">>> Step 4: Retiming trajectory...")

            try:
                plan = self.arm.retime_trajectory(
                    self.robot.get_current_state(),
                    plan,
                    velocity_scaling_factor=0.03,
                    acceleration_scaling_factor=0.03,
                )
            except Exception as e:
                rospy.logwarn("retime_trajectory failed: %s", str(e))
                rospy.logwarn("Will try executing original plan.")

            rospy.sleep(0.5)
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

    def move_to_named_target(self, target_name):
        """
        尝试移动到命名姿态，成功返回 True，失败返回 False
        """
        try:
            self.arm.set_start_state_to_current_state()
            rospy.sleep(0.5)
            self.arm.set_named_target(target_name)
            ok = self.arm.go(wait=True)
            self.arm.stop()
            self.arm.clear_pose_targets()
            rospy.sleep(0.5)
            return ok
        except Exception as e:
            rospy.logwarn(
                "Failed to move to named target [%s]: %s", target_name, str(e)
            )
            self.arm.stop()
            self.arm.clear_pose_targets()
            return False

    def shutdown(self):
        rospy.loginfo("Stopping demonstration and shutting down...")

        # 尝试回 sleep，失败也不影响退出
        moved_sleep = self.move_to_named_target("sleep")
        if not moved_sleep:
            rospy.logwarn("Failed to move to sleep pose.")

        moveit_commander.roscpp_shutdown()
        rospy.signal_shutdown("Demo finished")


if __name__ == "__main__":
    try:
        MoveItCircleDemo()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
