#!/usr/bin/env python3
import rospy
import actionlib
from sagittarius_object_color_detector.msg import SGRCtrlAction, SGRCtrlGoal

rospy.init_node('test_pose')
client = actionlib.SimpleActionClient('/sgr532/sgr_ctrl', SGRCtrlAction)
client.wait_for_server()
goal = SGRCtrlGoal()
goal.action_type = 1   # ACTION_TYPE_XYZ_RPY
goal.pos_x = 0.20
goal.pos_y = 0.00
goal.pos_z = 0.12
goal.pos_roll = 0.0
goal.pos_pitch = -1.0  # 或 -1.57
goal.pos_yaw = 0.0
goal.grasp_type = 0
client.send_goal_and_wait(goal, rospy.Duration(30))
rospy.loginfo("Done")