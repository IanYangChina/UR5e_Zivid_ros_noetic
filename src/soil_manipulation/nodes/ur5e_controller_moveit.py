#!/usr/bin/env python3.8

import os
import copy
import sys
import math
import rospy, rosnode
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from utils.poses import *
import moveit_commander
import quaternion
from scipy.spatial.transform import Rotation
from soil_manipulation.srv import TargetPose, TargetPoseResponse, Reset, ResetResponse, MoveDistance, MoveDistanceResponse
import matplotlib.pyplot as plt

DISTANCE_THRESHOLD = 0.001


def qmul(q, r):
    terms = np.outer(r, q)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    out = np.array([w, x, y, z])
    return out / np.sqrt(out.dot(out))


class Controller:
    def __init__(self, translation_speed=0.05, rotation_speed=0.05*np.pi):
        rospy.init_node('controller_node', anonymous=True)

        self.current_pose_msg = PoseStamped()
        self.current_xyz = np.array([0.0, 0.0, 0.0])
        self.current_header_seq = 0

        self.moveit_commander = moveit_commander.roscpp_initialize(sys.argv)
        self.moveit_robot = moveit_commander.RobotCommander(robot_description="robot_description")
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_group = moveit_commander.MoveGroupCommander("manipulator", robot_description="robot_description")

        self.delta_position = 0.01  # meter per waypoint
        self.delta_angle = 5  # angle per waypoint

        self.init_robot()

        self.sample_service = rospy.Service('move_to_target', TargetPose, self.move_target)
        self.reset_service = rospy.Service('reset', Reset, self.reset)
        self.move_service = rospy.Service('move_distance', MoveDistance, self.move_distance)

        rospy.Subscriber('/keyboard', String, callback=self.keyboard_callback)
        self.translation_speed = translation_speed
        self.rotation_speed = rotation_speed

    def init_robot(self):
        rospy.loginfo("Initializing robot...")
        # plan = self.moveit_group.plan(joints=waiting_joint_state.position)
        # self.moveit_group.execute(plan[1], wait=True)
        # plan = self.moveit_group.plan(joints=pre_manipulation_joint_state.position)
        # self.moveit_group.execute(plan[1], wait=True)

    def keyboard_callback(self, data):
        key_pressed = data.data

        p = self.moveit_group.get_current_pose()
        target_pose = copy.deepcopy(p)
        if key_pressed == '1':
            target_pose.pose.position.x += self.translation_speed
        elif key_pressed == '2':
            target_pose.pose.position.x -= self.translation_speed
        elif key_pressed == '3':
            target_pose.pose.position.y += self.translation_speed
        elif key_pressed == '4':
            target_pose.pose.position.y -= self.translation_speed
        elif key_pressed == '5':
            target_pose.pose.position.z += self.translation_speed
        elif key_pressed == '6':
            target_pose.pose.position.z -= self.translation_speed
        else:
            quat = np.array([target_pose.pose.orientation.x,
                             target_pose.pose.orientation.y,
                             target_pose.pose.orientation.z,
                             target_pose.pose.orientation.w])
            rotation = Rotation.from_quat(quat).as_matrix()
            if key_pressed == '7':
                rotation_delta = Rotation.from_euler('xyz', np.array([0.0, 0.0, self.rotation_speed])).as_matrix()
            elif key_pressed == '8':
                rotation_delta = Rotation.from_euler('xyz', np.array([0.0, 0.0, -self.rotation_speed])).as_matrix()
            elif key_pressed == '9':
                rotation_delta = Rotation.from_euler('xyz', np.array([0.0, self.rotation_speed, 0.0])).as_matrix()
            elif key_pressed == '0':
                rotation_delta = Rotation.from_euler('xyz', np.array([0.0, -self.rotation_speed, 0.0])).as_matrix()
            elif key_pressed == '-':
                rotation_delta = Rotation.from_euler('xyz', np.array([self.rotation_speed, 0.0, 0.0])).as_matrix()
            elif key_pressed == '=':
                rotation_delta = Rotation.from_euler('xyz', np.array([-self.rotation_speed, 0.0, 0.0])).as_matrix()
            else:
                rotation_delta = np.eye(3)
            rotation_new = np.dot(rotation_delta, rotation)
            print(rotation_new)
            quat_new = quaternion.as_float_array(quaternion.from_rotation_matrix(rotation_new))  # w, x, y, z
            target_pose.pose.orientation.w = quat_new[0]
            target_pose.pose.orientation.x = quat_new[1]
            target_pose.pose.orientation.y = quat_new[2]
            target_pose.pose.orientation.z = quat_new[3]

        waypoints = self.compose_cartesian_waypoints(p.pose, target_pose.pose)
        self.plan_and_execute(waypoints)

    def move_distance(self, req):
        rospy.loginfo("Received moving request, generating plan...")

        p = self.moveit_group.get_current_pose().pose
        tar_p = copy.deepcopy(p)
        tar_p.position.x += req.x
        tar_p.position.y += req.y
        tar_p.position.z += req.z
        delta_quat_xyzw = Rotation.from_euler('xyz', np.array([req.a, req.b, req.c]), degrees=True).as_quat()
        delta_quat_wxyz = [delta_quat_xyzw[-1], delta_quat_xyzw[0], delta_quat_xyzw[1], delta_quat_xyzw[2]]
        tar_quat = qmul([p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z], delta_quat_wxyz)
        tar_p.orientation.w = tar_quat[0]
        tar_p.orientation.x = tar_quat[1]
        tar_p.orientation.y = tar_quat[2]
        tar_p.orientation.z = tar_quat[3]

        waypoints = self.compose_cartesian_waypoints(p, tar_p)
        self.plan_and_execute(waypoints)

        return MoveDistanceResponse()

    def reset(self, req):
        plan = self.moveit_group.plan(joints=resting_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        return ResetResponse()

    def move_target(self, req):
        cur_p = self.moveit_group.get_current_pose().pose
        tar_p = copy.deepcopy(cur_p)
        tar_p.pose.position.x = req.x
        tar_p.pose.position.y = req.y
        tar_p.pose.position.z = req.z
        delta_quat_xyzw = Rotation.from_euler('xyz', np.array([req.a, req.b, req.c]), degrees=True).as_quat()
        tar_p.pose.orientation.x = delta_quat_xyzw[0]
        tar_p.pose.orientation.y = delta_quat_xyzw[1]
        tar_p.pose.orientation.z = delta_quat_xyzw[2]
        tar_p.pose.orientation.w = delta_quat_xyzw[3]
        waypoints = self.compose_cartesian_waypoints(cur_p, tar_p)
        self.plan_and_execute(waypoints)
        return TargetPoseResponse()

    def compose_cartesian_waypoints(self, pose0, pose1):
        waypoints = [copy.deepcopy(pose0)]

        dx = pose1.position.x - pose0.position.x
        delta_x = self.delta_position if dx > 0 else -self.delta_position
        abs_dx = np.abs(dx)
        dy = pose1.position.y - pose0.position.y
        delta_y = self.delta_position if dy > 0 else -self.delta_position
        abs_dy = np.abs(dy)
        dz = pose1.position.z - pose0.position.z
        delta_z = self.delta_position if dz > 0 else -self.delta_position
        abs_dz = np.abs(dz)

        current_euler = Rotation.from_quat([pose0.orientation.x, pose0.orientation.y, pose0.orientation.z, pose0.orientation.w]).as_euler(
            'xyz', degrees=True)
        target_euler = Rotation.from_quat([pose1.orientation.x, pose1.orientation.y, pose1.orientation.z, pose1.orientation.w]).as_euler(
            'xyz', degrees=True)

        da = target_euler[0] - current_euler[0]
        delta_a = self.delta_angle if da > 0 else -self.delta_angle
        abs_da = np.abs(da)
        db = target_euler[1] - current_euler[1]
        delta_b = self.delta_angle if db > 0 else -self.delta_angle
        abs_db = np.abs(db)
        dc = target_euler[2] - current_euler[2]
        delta_c = self.delta_angle if dc > 0 else -self.delta_angle
        abs_dc = np.abs(dc)

        n_t = max(int(np.max([abs_dx, abs_dy, abs_dz]) / self.delta_position),
                  int(np.max([abs_da, abs_db, abs_dc]) / self.delta_angle))
        p = copy.deepcopy(pose0)

        for _ in range(n_t):
            if np.abs(pose0.position.x - pose1.position.x) <= 0.0002:
                delta_x = 0
            if np.abs(pose0.position.y - pose1.position.y) <= 0.0002:
                delta_y = 0
            if np.abs(pose0.position.z - pose1.position.z) <= 0.0002:
                delta_z = 0
            if np.abs(current_euler[0] - target_euler[0]) <= 0.009:
                delta_a = 0
            if np.abs(current_euler[1] - target_euler[1]) <= 0.009:
                delta_b = 0
            if np.abs(current_euler[2] - target_euler[2]) <= 0.009:
                delta_c = 0
            p.position.x += delta_x
            p.position.y += delta_y
            p.position.z += delta_z
            current_euler[0] += delta_a
            current_euler[1] += delta_b
            current_euler[2] += delta_c
            quat_xyzw = Rotation.from_euler('xyz', current_euler.copy(), degrees=True).as_quat()
            p.orientation.x = quat_xyzw[0]
            p.orientation.y = quat_xyzw[1]
            p.orientation.z = quat_xyzw[2]
            p.orientation.w = quat_xyzw[3]
            waypoints.append(copy.deepcopy(p))

        return waypoints

    def plan_and_execute(self, waypoints):
        (plan, fraction) = self.moveit_group.compute_cartesian_path(
                                   waypoints,   # waypoints to follow
                                   self.delta_position,      # eef_step
                                   0.0)         # jump_threshold
        # moveit sometimes uses the same time value for the last two trajectory points, causing failure execution
        if plan.joint_trajectory.points[-2].time_from_start.nsecs == plan.joint_trajectory.points[-1].time_from_start.nsecs:
            plan.joint_trajectory.points[-1].time_from_start.nsecs += 10000

        # self.plot_eef_v(plan)
        self.moveit_group.execute(plan, wait=True)
        dt = plan.joint_trajectory.points[-1].time_from_start.secs + plan.joint_trajectory.points[-1].time_from_start.nsecs / 1e9
        rospy.loginfo("Time spent: "+str(dt)+" secs")

    def plan_and_show(self, waypoints, show=False, save_tr=False, tr_name='tr'):
        (plan, fraction) = self.moveit_group.compute_cartesian_path(
                                   waypoints,   # waypoints to follow
                                   self.delta_position,      # eef_step
                                   0.0)         # jump_threshold

        # moveit sometimes uses the same time value for the last two trajectory points, causing failure execution
        if plan.joint_trajectory.points[-2].time_from_start.nsecs == plan.joint_trajectory.points[-1].time_from_start.nsecs:
            plan.joint_trajectory.points[-1].time_from_start.nsecs += 1000

        if show:
            self.plot_eef_v(plan, save_tr, tr_name)
        return plan

    def current_pose_callback(self, data):
        self.current_pose_msg = data
        self.current_xyz = np.array([
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z
        ])
        self.current_header_seq = data.header.seq

    def plot_eef_v(self, plan, save=False, tr_name='tr'):
        cartesian_positions = []
        cartesian_velocities = []
        time_frames = []
        time_difference = [0.0]
        for i in range(len(plan.joint_trajectory.points)-1):
            position = plan.joint_trajectory.points[i].positions
            velocity = plan.joint_trajectory.points[i].velocities
            jacobian = self.moveit_group.get_jacobian_matrix(list(position))
            cartesian_position = np.dot(jacobian, np.array(position))
            cartesian_positions.append(cartesian_position)
            cartesian_velocity = np.dot(jacobian, np.array(velocity))
            cartesian_velocities.append(cartesian_velocity)
            time_frames.append(plan.joint_trajectory.points[i].time_from_start.secs + plan.joint_trajectory.points[i].time_from_start.nsecs / 1e9)
            if i > 0:
                time_difference.append(time_frames[-1] - time_frames[-2])
            # print('Timestamp: {}'.format(time_frames[-1]))
            # print('Time difference: {}'.format(time_difference[-1]))
            # print('Cartesian velocity: {}'.format(cartesian_velocity))

        if save:
            script_dir = os.path.dirname(__file__)
            data_dir = os.path.join(script_dir, '..', '..', 'test', 'cartesian_velocities')
            n = 0
            while os.path.exists(os.path.join(data_dir, tr_name+'_eef_v_'+str(n)+'.npy')):
                n += 1
            np.save(os.path.join(data_dir, tr_name+'_eef_v_'+str(n)+'.npy'), np.array(cartesian_velocities))
            np.save(os.path.join(data_dir, tr_name+'_timestamps_'+str(n)+'.npy'), np.array(time_frames))

        legends = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        # plt.plot(cartesian_velocities)
        # plt.legend(legends)
        # plt.xlabel('Horizon')
        # plt.ylabel('Velocity')
        # plt.title('End-effector velocity')
        # plt.show()

        print(cartesian_positions[-1])
        # plt.plot(cartesian_positions)
        # plt.legend(legends)
        # plt.xlabel('Horizon')
        # plt.ylabel('Pose')
        # plt.title('End-effector pose')
        # plt.show()

        # plt.plot(time_frames)
        # plt.xlabel('Horizon')
        # plt.ylabel('Timestamps')
        # plt.title('Trajectory timestamps')
        # plt.show()
        #
        # plt.plot(time_difference)
        # plt.xlabel('Horizon')
        # plt.ylabel('Time difference')
        # plt.title('Trajectory waypoint time difference')
        # plt.show()


if __name__ == '__main__':
    controller = Controller()
    rospy.spin()