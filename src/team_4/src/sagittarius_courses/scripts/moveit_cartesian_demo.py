#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys
import moveit_commander
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import Pose
from copy import deepcopy
import time

class MoveItCartesianDemo:
    def __init__(self):
        # 初始化 move_group 的 API
        moveit_commander.roscpp_initialize(sys.argv)

        # 修改：先初始化节点，再处理参数重映射
        rospy.init_node('moveit_circle_demo', anonymous=True)
        
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
        self.arm = MoveGroupCommander(
            'sagittarius_arm', 
            robot_description=f'{move_group_namespace}/robot_description',
            ns=move_group_namespace  # 指定命名空间
        )
        
        # 当运动规划失败后，允许重新规划
        self.arm.allow_replanning(True)
        
        # 设置目标位置所使用的参考坐标系
        self.arm.set_pose_reference_frame('world')
                
        # 设置位置 (单位：米) 和姿态（单位：弧度）的允许误差
        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.001)
        
        # 设置允许的最大速度和加速度
        self.arm.set_max_acceleration_scaling_factor(0.5)
        self.arm.set_max_velocity_scaling_factor(0.5)
        
        # 获取终端 link 的名称
        self.end_effector_link = self.arm.get_end_effector_link()

        # 控制机械臂先回到初始化位置
        rospy.loginfo("Moving to home position...")
        self.arm.set_named_target('home')
        self.arm.go()
        rospy.sleep(2)  # 等待稳定
                                               
        # 获取当前位姿数据作为机械臂运动的起始位姿
        start_pose = self.arm.get_current_pose(self.end_effector_link).pose
        rospy.loginfo("Start pose acquired.")
        # print(start_pose) # 调试时可打开
                
        # 初始化路点列表
        self.waypoints = []
                
        # --- 构建路点列表 (基于你的原始逻辑) ---
        wpose = deepcopy(start_pose)

        # Point 1: A
        wpose.position.z += 0.06
        self.waypoints.append(deepcopy(wpose))

        # Point 2: B
        wpose.position.y -= 0.06
        self.waypoints.append(deepcopy(wpose))

        # Point 3: C
        wpose.position.y += 0.12
        self.waypoints.append(deepcopy(wpose))

        # Point 4: A
        wpose.position.y -= 0.06
        self.waypoints.append(deepcopy(wpose))


        
        # wujiaoxing
        # # Point 1: Z + 0.1
        # wpose.position.z += 0.1
        # self.waypoints.append(deepcopy(wpose))

        # # Point 2: Z - 0.0809, Y - 0.0588 (相对于 start_pose)
        # wpose.position.x = start_pose.position.x - 0.0588
        # wpose.position.y = start_pose.position.y - 0.0588
        # self.waypoints.append(deepcopy(wpose))

        # # Point 3: Z + 0.0309, Y + 0.0951
        # wpose.position.x = start_pose.position.x + 0.0951
        # wpose.position.y = start_pose.position.y + 0.0951
        # self.waypoints.append(deepcopy(wpose))

        # # Point 4: Z + 0.0309, Y - 0.0951
        # wpose.position.x = start_pose.position.x + 0.0951
        # wpose.position.y = start_pose.position.y - 0.0951
        # self.waypoints.append(deepcopy(wpose))

        # # Point 5: Z - 0.0809, Y + 0.0588
        # wpose.position.x = start_pose.position.x - 0.0588
        # wpose.position.y = start_pose.position.y + 0.0588
        # self.waypoints.append(deepcopy(wpose))
        # # --------------------------------------

        # for waypoint in self.waypoints:
        #     waypoint.position.x *= 0.5
        #     waypoint.position.x += 0.06
        #     waypoint.position.y *= 0.5


        rospy.loginfo(f"Total waypoints generated: {len(self.waypoints)}")

        # 状态变量
        self.current_point_index = 0
        self.is_moving = False  # 标记是否正在执行动作，防止重叠调用
        self.max_retries = 5    # 单个点的最大重试次数
        self.retry_count = 0    # 当前点的重试计数

        # 创建定时器：每 1.5 秒触发一次回调
        # 注意：时间间隔需要大于单次运动 + 规划的时间，或者依靠 is_moving 标志位来控制
        self.timer = rospy.Timer(rospy.Duration(1.5), self.timer_callback)
        
        rospy.loginfo("Timer started. Beginning sequential execution...")

    def timer_callback(self, event):
        """
        定时器回调函数：依次执行每一个路点
        """
        # 如果正在运动中，跳过本次回调，等待下一次
        if self.is_moving:
            return

        # 检查是否所有点都已执行完毕
        if self.current_point_index >= len(self.waypoints):
            rospy.loginfo("All waypoints executed successfully!")
            self.finish_execution()
            return

        target_pose = self.waypoints[self.current_point_index]
        rospy.loginfo(f"Attempting to plan for Waypoint {self.current_point_index + 1}/{len(self.waypoints)}")

        # 设置初始状态为当前状态（非常重要，确保基于最新位置规划）
        self.arm.set_start_state_to_current_state()

        # 构造单点路点列表（compute_cartesian_path 需要列表）
        single_waypoints = [target_pose]

        fraction = 0.0
        attempts = 0
        max_planning_tries = 20 # 单个点的规划尝试次数

        # 尝试规划
        while fraction < 1.0 and attempts < max_planning_tries:
            (plan, fraction) = self.arm.compute_cartesian_path(
                single_waypoints,
                0.01,        # eef_step
                0.0,         # jump_threshold
                avoid_collisions=True
            )
            attempts += 1
            
            if attempts % 5 == 0:
                rospy.logwarn(f"Planning attempt {attempts} for point {self.current_point_index + 1}...")

        # 判断规划结果
        if fraction == 1.0:
            rospy.loginfo(f"Path for point {self.current_point_index + 1} computed successfully. Executing...")
            self.is_moving = True
            
            # 执行运动 (execute 是阻塞的，但在回调中执行通常没问题，只要定时器频率不高)
            # 为了更安全，可以在另一个线程执行，但简单场景下直接 execute 即可
            result = self.arm.execute(plan)
            
            if result:
                rospy.loginfo(f"Point {self.current_point_index + 1} execution complete.")
                self.current_point_index += 1 # 移动到下一个点
                self.retry_count = 0          # 重置重试计数
            else:
                rospy.logerr(f"Execution failed for point {self.current_point_index + 1}.")
            
            self.is_moving = False

        else:
            rospy.logerr(f"Planning failed for point {self.current_point_index + 1} (Success rate: {fraction}).")
            self.retry_count += 1
            
            if self.retry_count >= self.max_retries:
                rospy.logerr(f"Max retries reached for point {self.current_point_index + 1}. Stopping sequence.")
                self.finish_execution(success=False)
            else:
                rospy.logwarn(f"Retrying point {self.current_point_index + 1} (Attempt {self.retry_count})...")

    def finish_execution(self, success=True):
        """
        结束流程：停止定时器，回到 sleep 姿态，退出
        """
        self.timer.shutdown() # 停止定时器
        
        if success:
            rospy.loginfo("Sequence finished. Moving to sleep position...")
            self.arm.set_named_target('sleep')
            self.arm.go()
            rospy.sleep(2)
        else:
            rospy.logerr("Sequence aborted due to errors.")

        rospy.loginfo("Shutting down MoveIt...")
        moveit_commander.roscpp_shutdown()
        # 使用 rospy.signal_shutdown 更安全地退出节点
        rospy.signal_shutdown("Demo complete")
        # 如果需要强制退出进程，保留下面这行，否则通常不需要
        # moveit_commander.os._exit(0)

if __name__ == "__main__":
    try:
        MoveItCartesianDemo()
        # 保持节点运行，直到 shutdown 被调用
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import rospy
# import sys
# import moveit_commander
# from moveit_commander import MoveGroupCommander
# from geometry_msgs.msg import Pose
# from copy import deepcopy
# import math

# class MoveItCircleDemo:
#     def __init__(self):
#         # 1. 初始化 move_group 接口
#         moveit_commander.roscpp_initialize(sys.argv)
        
#         # 修改：先初始化节点，再处理参数重映射
#         rospy.init_node('moveit_circle_demo', anonymous=True)
        
#                 # ===== 重要修改：设置命名空间 =====
#         # 设置move_group的命名空间
#         move_group_namespace = '/sgr532'
        
#         # 复制robot_description参数
#         if not rospy.get_param('robot_description', None):
#             robot_desc = rospy.get_param(f'{move_group_namespace}/robot_description', None)
#             if robot_desc:
#                 rospy.set_param('robot_description', robot_desc)
#                 rospy.loginfo("Copied robot_description to root namespace")
        
#         # 创建MoveGroupCommander，指定正确的命名空间
#         self.arm = MoveGroupCommander(
#             'sagittarius_arm', 
#             robot_description=f'{move_group_namespace}/robot_description',
#             ns=move_group_namespace  # 指定命名空间
#         )
        
#         # 2. 初始化机械臂组 (请根据实际名称修改 'sagittarius_arm')
#         # self.arm = MoveGroupCommander('sagittarius_arm')
        
#         # 允许重规划
#         self.arm.allow_replanning(True)
#         self.arm.set_pose_reference_frame('world')
        
#         # 设置精度
#         self.arm.set_goal_position_tolerance(0.001)
#         self.arm.set_goal_orientation_tolerance(0.001)
        
#         # 设置速度/加速度限制 (画圆建议稍微慢一点以保证平滑)
#         self.arm.set_max_velocity_scaling_factor(0.3)
#         self.arm.set_max_acceleration_scaling_factor(0.3)
        
#         self.end_effector_link = self.arm.get_end_effector_link()

#         # 3. 移动到安全起始点
#         rospy.loginfo(">>> Step 1: Moving to safe start position (home)...")
#         self.arm.set_named_target('home')
#         if not self.arm.go():
#             rospy.logerr("Failed to move to home position.")
#             self.shutdown()
#             return
#         rospy.sleep(1.0)

#         # 4. 获取当前位姿作为圆的中心参考
#         current_pose = self.arm.get_current_pose(self.end_effector_link).pose
#         rospy.loginfo(f"Current pose acquired at: x={current_pose.position.x:.3f}, y={current_pose.position.y:.3f}, z={current_pose.position.z:.3f}")

#         # --- 配置画圆参数 ---
#         circle_center = deepcopy(current_pose)
#         # 可选：如果你想让圆心不在当前点，而是在当前点前方/上方，可以在此修改
#         # circle_center.position.x += 0.1 
        
#         radius = 0.08  # 圆的半径 (米)，请根据机械臂工作空间调整，不要太大
#         num_points = 20 # 圆被分成多少个点，越多越平滑，但规划时间越长
        
#         rospy.loginfo(f">>> Step 2: Generating {num_points} waypoints for a circle with radius {radius}m...")

#         # 5. 计算路点列表
#         waypoints = []
        
#         # 保存初始姿态 (画圆时通常保持姿态不变)
#         target_orientation = deepcopy(current_pose.orientation)
        
#         for i in range(num_points + 1): # +1 是为了让终点和起点重合，形成闭环
#             angle = 2 * math.pi * i / num_points
            
#             wpose = deepcopy(circle_center)
            
#             # 使用三角函数计算圆上的点 (在 XY 平面画圆)
#             # 如果需要在这个平面画圆，修改下面的 x, y 计算公式
#             wpose.position.x = circle_center.position.x + radius * math.cos(angle)
#             wpose.position.y = circle_center.position.y + radius * math.sin(angle)
#             wpose.position.z = circle_center.position.z # 保持高度不变
            
#             # 保持姿态不变 (非常重要，否则机械臂会在画圆时乱转)
#             wpose.orientation = target_orientation
            
#             waypoints.append(deepcopy(wpose))

#         rospy.loginfo(f"Waypoints generated. Total points: {len(waypoints)}")

#         # 6. 调用路径规划 API
#         rospy.loginfo(">>> Step 3: Computing Cartesian path...")
        
#         fraction = 0.0
#         max_tries = 50
#         attempts = 0
        
#         # 设置起始状态为当前状态
#         self.arm.set_start_state_to_current_state()

#         while fraction < 1.0 and attempts < max_tries:
#             (plan, fraction) = self.arm.compute_cartesian_path(
#                 waypoints,
#                 0.01,       # eef_step: 终端步进值 (米)，越小越精确但计算越慢
#                 0.0,        # jump_threshold: 跳跃阈值，0.0 表示不允许跳跃
#                 avoid_collisions=True
#             )
#             attempts += 1
#             if attempts % 10 == 0:
#                 rospy.loginfo(f"Still trying to plan circle... ({attempts}/{max_tries})")

#         # 7. 执行运动
#         if fraction == 1.0:
#             rospy.loginfo(">>> Path computed successfully! Starting circular motion.")
#             result = self.arm.execute(plan)
#             if result:
#                 rospy.loginfo(">>> Circle drawing complete!")
#             else:
#                 rospy.logerr("Execution failed during motion.")
#         else:
#             rospy.logerr(f"Path planning failed! Only achieved {fraction*100:.1f}% of the path after {attempts} attempts.")
#             rospy.logerr("Possible reasons: Radius too large, singularities, or collision.")

#         rospy.sleep(2.0)

#         # 8. 结束清理
#         self.shutdown()

#     def shutdown(self):
#         rospy.loginfo("Stopping demonstration and shutting down...")
#         # 可以选择回到 sleep 姿态
#         try:
#             self.arm.set_named_target('sleep')
#             self.arm.go()
#         except:
#             pass
        
#         moveit_commander.roscpp_shutdown()
#         rospy.signal_shutdown("Demo finished")
#         # moveit_commander.os._exit(0)

# if __name__ == "__main__":
#     try:
#         MoveItCircleDemo()
#         # 保持节点运行直到完成
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass
