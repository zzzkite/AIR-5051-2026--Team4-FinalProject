#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
import moveit_commander
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import Pose
from copy import deepcopy


class MoveItPickPlaceDemo:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('moveit_pick_place_demo', anonymous=True)

        self.ns = '/sgr532'
        self.robot_description = f'{self.ns}/robot_description'

        # 兼容有些环境只在命名空间下有 robot_description
        if not rospy.get_param('robot_description', None):
            robot_desc = rospy.get_param(self.robot_description, None)
            if robot_desc:
                rospy.set_param('robot_description', robot_desc)
                rospy.loginfo("Copied robot_description to root namespace")

        # Robot / Arm / Gripper
        self.robot = moveit_commander.RobotCommander(
            robot_description=self.robot_description,
            ns=self.ns
        )

        self.arm = MoveGroupCommander(
            'sagittarius_arm',
            robot_description=self.robot_description,
            ns=self.ns
        )

        self.gripper = MoveGroupCommander(
            'sagittarius_gripper',
            robot_description=self.robot_description,
            ns=self.ns
        )

        # 机械臂参数
        self.arm.allow_replanning(True)
        self.arm.set_pose_reference_frame('world')
        # self.arm.set_goal_position_tolerance(0.003)
        # self.arm.set_goal_orientation_tolerance(0.003)
        self.arm.set_goal_joint_tolerance(0.001)
        self.arm.set_max_velocity_scaling_factor(0.5)
        self.arm.set_max_acceleration_scaling_factor(0.5)

        # 夹爪参数
        self.gripper.set_goal_joint_tolerance(0.001)
        self.gripper.set_max_velocity_scaling_factor(0.5)
        self.gripper.set_max_acceleration_scaling_factor(0.5)

        self.end_effector_link = self.arm.get_end_effector_link()

        rospy.loginfo("Robot groups: %s", self.robot.get_group_names())
        rospy.loginfo("End effector link: %s", self.end_effector_link)

        self.run_demo()
        self.shutdown()

    # ---------- 基础动作 ----------
    def move_arm_named(self, target_name):
        rospy.loginfo("Move arm to named target: %s", target_name)
        self.arm.set_start_state_to_current_state()
        self.arm.set_named_target(target_name)
        ok = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        rospy.sleep(1.0)
        return ok

    def move_gripper_named(self, target_name):
        rospy.loginfo("Move gripper to named target: %s", target_name)
        self.gripper.set_named_target(target_name)
        ok = self.gripper.go(wait=True)
        self.gripper.stop()
        rospy.sleep(1.0)
        return ok

    def move_to_pose(self, target_pose):
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(target_pose, self.end_effector_link)
        ok = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        rospy.sleep(1.0)
        return ok

    # ---------- 主流程 ----------
    def run_demo(self):
        rospy.loginfo("========== Pick and Place Demo Start ==========")

        # 1. 回到初始位置
        if not self.move_arm_named('home'):
            rospy.logerr("Failed to move arm to home.")
            return

        # 2. 夹爪张开
        self.move_gripper_named('open')
        # if not self.move_gripper_named('open'):
        #     rospy.logerr("Failed to open gripper.")
        #     return

        # 3. 获取当前姿态，保留当前姿态方向，只改位置
        current_pose = self.arm.get_current_pose(self.end_effector_link).pose
        rospy.loginfo(
            "Current pose: x=%.3f, y=%.3f, z=%.3f",
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z
        )

        # ===== 下面这些坐标你可以按实际物体位置微调 =====
        # 假设物体放在机械臂前方一点点
        object_x = current_pose.position.x + 0.02
        object_y = current_pose.position.y
        object_z = current_pose.position.z - 0.08

        # 放置位置：放在右侧一点
        place_x = current_pose.position.x + 0.02
        place_y = current_pose.position.y - 0.05
        place_z = object_z

        # 抓取上方和放置上方的安全高度
        approach_height = 0.05

        # 4. 到物体上方
        above_pick_pose = deepcopy(current_pose)
        above_pick_pose.position.x = object_x
        above_pick_pose.position.y = object_y
        above_pick_pose.position.z = object_z + approach_height

        rospy.loginfo("Moving above object...")
        if not self.move_to_pose(above_pick_pose):
            rospy.logerr("Failed to move above object.")
            return

        # 5. 下移到抓取位置
        pick_pose = deepcopy(above_pick_pose)
        pick_pose.position.z = object_z

        rospy.loginfo("Moving down to pick position...")
        if not self.move_to_pose(pick_pose):
            rospy.logerr("Failed to move to pick position.")
            return

        # 6. 夹爪闭合
        rospy.loginfo("Closing gripper...")
        self.move_gripper_named('close')
        # if not self.move_gripper_named('close'):
        #     rospy.logerr("Failed to close gripper.")
        #     return

        # 7. 抬起到物体上方
        rospy.loginfo("Lifting object...")
        if not self.move_to_pose(above_pick_pose):
            rospy.logerr("Failed to lift object.")
            return

        # 8. 移动到放置位置上方
        above_place_pose = deepcopy(current_pose)
        above_place_pose.position.x = place_x
        above_place_pose.position.y = place_y
        above_place_pose.position.z = place_z + approach_height

        rospy.loginfo("Moving above place position...")
        if not self.move_to_pose(above_place_pose):
            rospy.logerr("Failed to move above place position.")
            return

        # 9. 下移到放置位置
        place_pose = deepcopy(above_place_pose)
        place_pose.position.z = place_z

        rospy.loginfo("Moving down to place position...")
        if not self.move_to_pose(place_pose):
            rospy.logerr("Failed to move to place position.")
            return

        # 10. 张开夹爪放下物体
        rospy.loginfo("Opening gripper to release object...")
        self.move_gripper_named('open')
        # if not self.move_gripper_named('open'):
        #     rospy.logerr("Failed to open gripper for release.")
        #     return

        # 11. 抬起
        rospy.loginfo("Lifting away from place position...")
        if not self.move_to_pose(above_place_pose):
            rospy.logerr("Failed to lift after release.")
            return

        # 12. 回到初始位置
        rospy.loginfo("Returning arm to home...")
        self.move_arm_named('home')

        # 13. 夹爪回到初始化位置
        rospy.loginfo("Returning gripper to open...")
        self.move_gripper_named('open')

        rospy.loginfo("========== Pick and Place Demo Finished ==========")

    def shutdown(self):
        rospy.loginfo("Shutting down MoveIt...")
        moveit_commander.roscpp_shutdown()
        rospy.signal_shutdown("Demo finished")


if __name__ == "__main__":
    try:
        MoveItPickPlaceDemo()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass