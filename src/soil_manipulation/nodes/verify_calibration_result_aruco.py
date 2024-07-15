#!/usr/bin/env python3

import os, sys
import yaml
import rospy
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
import moveit_commander


def construct_homogeneous_transform_matrix(translation, orientation):
    translation = np.array(translation).reshape((3, 1))  # xyz
    if len(orientation) == 4:
        rotation = Rotation.from_quat(orientation).as_matrix()
    else:
        assert len(orientation) == 3, 'orientation should be a quaternion or 3 axis angles'
        rotation = Rotation.from_euler('xyz', orientation).as_matrix()
    transformation = np.append(rotation, translation, axis=1)
    transformation = np.append(transformation, np.array([[0, 0, 0, 1]]), axis=0)
    return transformation.copy()


script_path = os.path.dirname(os.path.realpath(__file__))
# Extrinsic calibration result
with open(os.path.join(script_path, '..', 'src', 'zivid_cam_extrinsic.yml'), 'r') as file:
    data = yaml.safe_load(file)
raw_transform_world_to_cam = data['matrix']
print('Loaded extrinsic calibration result: \n{}'.format(raw_transform_world_to_cam))
raw_transform_cam_to_world = np.linalg.inv(raw_transform_world_to_cam)


class ArucoPoseLisetener:
    """
    Listen to the aruco pose and convert it to the world frame.
    The program prints the transformation matrix from the world frame to the aruco frame and the end effector frame.
    Examine:
        1. If the relative position between the aruco and the effector is correct.
        2. if the aruco position correctly moves when the effector is moved.
    """
    def __init__(self):
        rospy.init_node("aruco_listener_node", anonymous=True)
        rospy.loginfo("Starting aruco_listener_node")
        self.aruco_pose_subscriber = rospy.Subscriber('/aruco_tracker/pose', PoseStamped, self.aruco_pose_callback)

        self.moveit_commander = moveit_commander.roscpp_initialize(sys.argv)
        self.moveit_robot = moveit_commander.RobotCommander(robot_description="robot_description")
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_group = moveit_commander.MoveGroupCommander("manipulator", robot_description="robot_description")

    def aruco_pose_callback(self, msg):
        rospy.loginfo('Aruco pose received, convert to world frame...')
        transform_cam_to_aruco_pose = construct_homogeneous_transform_matrix(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        )
        transform_world_to_aruco = np.dot(raw_transform_world_to_cam, transform_cam_to_aruco_pose)
        end_effector_pose = self.moveit_group.get_current_pose().pose
        transform_world_to_ee = construct_homogeneous_transform_matrix(
            [end_effector_pose.position.x, end_effector_pose.position.y, end_effector_pose.position.z],
            [end_effector_pose.orientation.x, end_effector_pose.orientation.y, end_effector_pose.orientation.z, end_effector_pose.orientation.w]
        )
        rospy.loginfo('transform_world_to_aruco: \n{}'.format(transform_world_to_aruco))
        rospy.loginfo('transform_world_to_ee: \n{}'.format(transform_world_to_ee))


if __name__ == '__main__':
    aruco_pose_listener = ArucoPoseLisetener()
    rospy.spin()
