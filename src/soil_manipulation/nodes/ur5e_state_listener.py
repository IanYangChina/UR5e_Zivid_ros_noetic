#!/usr/bin/env python2

import sys
import rospy
import numpy as np
from sensor_msgs.msg import JointState
import moveit_commander
import tf


class StateListener:
    def __init__(self):
        rospy.init_node('listener_node', anonymous=True)
        rospy.Subscriber('/joint_states', JointState, callback=self.current_state_callback)
        self.current_joint_msg = JointState()
        self.current_xyz = np.array([0.0, 0.0, 0.0])
        self.current_quat = np.array([0.0, 0.0, 0.0, 0.0])
        self.current_header_seq = 0

        # self.tf_listener = tf.TransformListener()

        self.moveit_commander = moveit_commander.roscpp_initialize(sys.argv)
        self.moveit_robot = moveit_commander.RobotCommander(robot_description="robot_description")
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_group = moveit_commander.MoveGroupCommander("manipulator", robot_description="robot_description")

    def current_state_callback(self, msg):
        self.current_joint_msg = msg
        j0 = self.current_joint_msg.position[0]
        j1 = self.current_joint_msg.position[1]
        j2 = self.current_joint_msg.position[2]
        j3 = self.current_joint_msg.position[3]
        j4 = self.current_joint_msg.position[4]
        j5 = self.current_joint_msg.position[5]
        self.current_joint_msg.position = [j2, j1, j0, j3, j4, j5]
        pose = self.moveit_group.get_current_pose().pose
        self.current_xyz = np.array([pose.position.x, pose.position.y, pose.position.z])
        self.current_quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        # (trans, rot) = self.tf_listener.lookupTransform('base_link', 'base', rospy.Time(0))
        # print(trans, rot)

        print("New pose")
        print("X.pose.position.x = " + str(pose.position.x))
        print("X.pose.position.y = " + str(pose.position.y))
        print("X.pose.position.z = " + str(pose.position.z))
        print("X.pose.orientation.w = " + str(pose.orientation.w))
        print("X.pose.orientation.x = " + str(pose.orientation.x))
        print("X.pose.orientation.y = " + str(pose.orientation.y))
        print("X.pose.orientation.z = " + str(pose.orientation.z))
        euler = tf.transformations.euler_from_quaternion(self.current_quat)
        print("Euler angles: " + str(euler))
        print("Joint states")
        print(self.current_joint_msg)


if __name__ == "__main__":
    listener_node = StateListener()
    rospy.spin()
