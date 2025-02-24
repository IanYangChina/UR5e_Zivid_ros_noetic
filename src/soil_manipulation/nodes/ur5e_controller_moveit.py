#!/usr/bin/env python3.8

import os
import copy
import sys
import rospy, rosnode
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from utils.poses import *
import moveit_commander
from scipy.spatial.transform import Rotation
from soil_manipulation.srv import TargetPose, TargetPoseResponse, Reset, ResetResponse, MoveDistance, MoveDistanceResponse
from soil_manipulation.srv import PrintPose, PrintPoseResponse, Rest, RestResponse, Skill, SkillResponse
from soil_manipulation.srv import PreManipulation, PreManipulationResponse, Task, TaskResponse
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

def w2quat(axis_angle):
    axis_angle = np.asarray(axis_angle)
    w = np.sqrt(np.sum(axis_angle**2))
    out = np.zeros(shape=(4,))

    v = (axis_angle / w) * np.sin(w / 2)
    out[0] = np.cos(w / 2)
    out[1] = v[0]
    out[2] = v[1]
    out[3] = v[2]

    return out


def get_skill_waypoints(p, skill_params, skill='dig'):
    waypoints = [copy.deepcopy(p)]
    if skill == 'dig':
        p_ = copy.deepcopy(p)
        move_distance = (skill_params[0] + 1 / 3) * 0.18
        p_.position.x += move_distance
        rotate_x = (skill_params[1] + 1 / 3) * (np.pi * 3 / 16)
        delta_quat = Rotation.from_euler('xyz', np.array([rotate_x, 0.0, 0.0]),
                                       degrees=False).as_quat()
        delta_quat_wxyz = [delta_quat[-1], delta_quat[0], delta_quat[1], delta_quat[2]]
        tar_quat = qmul([p_.orientation.w, p_.orientation.x, p_.orientation.y, p_.orientation.z],
                        delta_quat_wxyz)
        p_.orientation.w = tar_quat[0]
        p_.orientation.x = tar_quat[1]
        p_.orientation.y = tar_quat[2]
        p_.orientation.z = tar_quat[3]
        waypoints.append(copy.deepcopy(p_))

        insert_angle = rotate_x + np.pi / 2
        insert_distance = (skill_params[2] + 1) / 2 * 0.05  # map [-1, 1] to [0, 0.05]
        insert_distance_x = insert_distance * np.cos(insert_angle)
        insert_distance_z = insert_distance * np.sin(insert_angle)
        insert_distance_z = min(insert_distance_z, 0.055)
        p__ = copy.deepcopy(p_)
        p__.position.x -= insert_distance_x
        p__.position.z -= insert_distance_z
        waypoints.append(copy.deepcopy(p__))

        push_angle = (skill_params[3] + 3) * np.pi / 3  # map [-1, 1] to [2*pi/3, 4*pi/3]
        push_distance = skill_params[4] * 0.18 + 0.22  # map [-1, 1] to [0.04, 0.40]
        push_distance_x = push_distance * np.cos(push_angle)
        push_distance_x = min(max(-(0.13 - np.abs(0.12*np.sin(rotate_x)) - insert_distance_x + move_distance),
                              push_distance_x),  0.0)
        push_distance_z = push_distance * np.sin(push_angle)
        push_distance_z = max(push_distance_z, -0.05+insert_distance_z)
        p___ = copy.deepcopy(p__)
        p___.position.x += push_distance_x
        p___.position.z += push_distance_z
        waypoints.append(copy.deepcopy(p___))

        rotate_x_back = -rotate_x
        delta_quat = Rotation.from_euler('xyz', np.array([rotate_x_back, 0.0, 0.0]),
                                          degrees=False).as_quat()
        delta_quat_wxyz = [delta_quat[-1], delta_quat[0], delta_quat[1], delta_quat[2]]
        tar_quat = qmul([p___.orientation.w, p___.orientation.x, p___.orientation.y, p___.orientation.z],
                        delta_quat_wxyz)
        p____ = copy.deepcopy(p___)
        p____.orientation.w = tar_quat[0]
        p____.orientation.x = tar_quat[1]
        p____.orientation.y = tar_quat[2]
        p____.orientation.z = tar_quat[3]

        move_up_distance = 0.1
        p____.position.z += move_up_distance
        waypoints.append(copy.deepcopy(p____))

    elif skill == 'smooth':
        p_ = copy.deepcopy(p)
        rotate_x = skill_params[0] * (np.pi / 6) + (np.pi / 6)  # map [-1, 1] to [0, pi/3]
        delta_quat = Rotation.from_euler('xyz', np.array([-rotate_x, 0.0, 0.0]),
                                         degrees=False).as_quat()
        delta_quat_wxyz = [delta_quat[-1], delta_quat[0], delta_quat[1], delta_quat[2]]
        tar_quat = qmul([p_.orientation.w, p_.orientation.x, p_.orientation.y, p_.orientation.z],
                        delta_quat_wxyz)
        p_.orientation.w = tar_quat[0]
        p_.orientation.x = tar_quat[1]
        p_.orientation.y = tar_quat[2]
        p_.orientation.z = tar_quat[3]
        waypoints.append(copy.deepcopy(p_))

        move_distance = skill_params[1] * 0.12 - 0.12  # map [-1, 1] to [-0.24, 0]
        p__ = copy.deepcopy(p_)
        p__.position.x += move_distance
        waypoints.append(copy.deepcopy(p__))

        descend_distance = 0.02 * skill_params[2] - 0.03
        p___ = copy.deepcopy(p__)
        p___.position.z += descend_distance
        waypoints.append(copy.deepcopy(p___))

        smooth_distance = skill_params[3] * 0.18 + 0.22  # map [-1, 1] to [0.04, 0.40]
        p____ = copy.deepcopy(p___)
        p____.position.x += smooth_distance
        waypoints.append(copy.deepcopy(p____))

        move_up_distance = 0.1
        p_____ = copy.deepcopy(p____)
        p_____.position.x += move_up_distance
        waypoints.append(copy.deepcopy(p_____))

    return waypoints


class Controller:
    def __init__(self, translation_speed=0.05, rotation_speed=10):
        rospy.init_node('controller_node', anonymous=True)

        self.current_joint_msg = JointState()
        self.current_pose_msg = PoseStamped()
        self.current_xyz = np.array([0.0, 0.0, 0.0])
        self.current_euler = np.array([0.0, 0.0, 0.0])
        self.current_header_seq = 0

        self.moveit_commander = moveit_commander.roscpp_initialize(sys.argv)
        self.moveit_robot = moveit_commander.RobotCommander(robot_description="robot_description")
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_group = moveit_commander.MoveGroupCommander("manipulator", robot_description="robot_description")

        self.delta_position = 0.005  # meter per waypoint
        self.delta_angle = 5  # angle per waypoint

        # self.init_robot()
        rospy.Subscriber('/joint_states', JointState, self.current_pose_callback)

        self.print_pose_service = rospy.Service('print_pose', PrintPose, self.print_pose)
        self.reset_service = rospy.Service('reset', Reset, self.reset)
        self.rest_service = rospy.Service('rest', Rest, self.rest)
        self.move_distance_service = rospy.Service('move_distance', MoveDistance, self.move_distance)
        self.move_target_service = rospy.Service('move_to_target', TargetPose, self.move_target)
        self.pre_mani_service = rospy.Service('pre_manipulation', PreManipulation, self.pre_manipulation)
        self.skill_service = rospy.Service('skill', Skill, self.execute_skill)
        self.task_service = rospy.Service('task', Task, self.execute_task)

        rospy.Subscriber('/keyboard', String, callback=self.keyboard_callback)
        self.translation_speed = translation_speed
        self.rotation_speed = rotation_speed

    def init_robot(self):
        rospy.loginfo("Initializing robot...")
        plan = self.moveit_group.plan(joints=waiting_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        pass

    def pre_manipulation(self, req):
        plan = self.moveit_group.plan(joints=pre_manipulation_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        return PreManipulationResponse()

    def reset(self, req):
        plan = self.moveit_group.plan(joints=waiting_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        return ResetResponse()

    def rest(self, req):
        plan = self.moveit_group.plan(joints=resting_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        return RestResponse()

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
            if key_pressed == '7':
                delta_quat_xyzw = Rotation.from_euler('xyz', np.array([0.0, 0.0, self.rotation_speed]),
                                                      degrees=True).as_quat()
            elif key_pressed == '8':
                delta_quat_xyzw = Rotation.from_euler('xyz', np.array([0.0, 0.0, -self.rotation_speed]),
                                                      degrees=True).as_quat()
            elif key_pressed == '9':
                delta_quat_xyzw = Rotation.from_euler('xyz', np.array([0.0, self.rotation_speed, 0.0]),
                                                      degrees=True).as_quat()
            elif key_pressed == '0':
                delta_quat_xyzw = Rotation.from_euler('xyz', np.array([0.0, -self.rotation_speed, 0.0]),
                                                      degrees=True).as_quat()
            elif key_pressed == '-':
                delta_quat_xyzw = Rotation.from_euler('xyz', np.array([self.rotation_speed, 0.0, 0.0]),
                                                      degrees=True).as_quat()
            elif key_pressed == '=':
                delta_quat_xyzw = Rotation.from_euler('xyz', np.array([-self.rotation_speed, 0.0, 0.0]),
                                                      degrees=True).as_quat()
            else:
                delta_quat_xyzw = Rotation.from_euler('xyz', np.array([0.0, 0.0, 0.0]),
                                                      degrees=True).as_quat()
            delta_quat_wxyz = [delta_quat_xyzw[-1], delta_quat_xyzw[0], delta_quat_xyzw[1], delta_quat_xyzw[2]]
            tar_quat = qmul([p.pose.orientation.w, p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z],
                            delta_quat_wxyz)
            target_pose.pose.orientation.w = tar_quat[0]
            target_pose.pose.orientation.x = tar_quat[1]
            target_pose.pose.orientation.y = tar_quat[2]
            target_pose.pose.orientation.z = tar_quat[3]

        self.plan_and_execute([p.pose, target_pose.pose])

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
        self.plan_and_execute([p, tar_p])

        return MoveDistanceResponse()

    def move_target(self, req):
        p = self.moveit_group.get_current_pose().pose
        tar_p = copy.deepcopy(p)
        tar_p.pose.position.x = req.x
        tar_p.pose.position.y = req.y
        tar_p.pose.position.z = req.z
        tar_quat_xyzw = Rotation.from_euler('xyz', np.array([req.a, req.b, req.c]), degrees=True).as_quat()
        tar_p.pose.orientation.x = tar_quat_xyzw[0]
        tar_p.pose.orientation.y = tar_quat_xyzw[1]
        tar_p.pose.orientation.z = tar_quat_xyzw[2]
        tar_p.pose.orientation.w = tar_quat_xyzw[3]
        waypoints = self.compose_cartesian_waypoints(p, tar_p)
        self.plan_and_execute(waypoints)
        return TargetPoseResponse()

    def execute_task(self, req):
        plan = self.moveit_group.plan(joints=waiting_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        plan = self.moveit_group.plan(joints=pre_manipulation_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)

        task_ind = req.n
        rospy.loginfo(f"task id: {task_ind}")

        if task_ind == -1:
            # system identification motion 1
            p = self.moveit_group.get_current_pose().pose
            w0 = copy.deepcopy(p)
            w0.position.x += 0.09
            w1 = copy.deepcopy(w0)
            w1.position.z -= 0.05
            w2 = copy.deepcopy(w1)
            w2.position.x -= 0.12
            w3 = copy.deepcopy(w2)
            w3.position.z += 0.12
            self.plan_and_execute([p, w0, w1, w2, w3])
            # self.plan_and_show([p, w0, w1, w2, w3], save_tr=False, tr_name='sys_id_1')
        elif task_ind == -2:
            # system identification motion 2
            p = self.moveit_group.get_current_pose().pose
            w0 = copy.deepcopy(p)
            w0.position.x += 0.09
            w0.position.y += 0.09
            quat_xyzw = Rotation.from_euler('xyz', np.array([0.0, 0.0, -45]),
                                            degrees=True).as_quat()
            quat_wxyz = [quat_xyzw[-1], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            tar_quat = qmul([w0.orientation.w, w0.orientation.x, w0.orientation.y, w0.orientation.z],
                            quat_wxyz)
            w0.orientation.w = tar_quat[0]
            w0.orientation.x = tar_quat[1]
            w0.orientation.y = tar_quat[2]
            w0.orientation.z = tar_quat[3]
            w1 = copy.deepcopy(w0)
            w1.position.z -= 0.05
            w2 = copy.deepcopy(w1)
            w2.position.x -= 0.12
            w2.position.y -= 0.12
            w3 = copy.deepcopy(w2)
            w3.position.z += 0.12
            self.plan_and_execute([p, w0, w1, w2, w3])
        elif task_ind == -3:
            # system identification motion 2
            p = self.moveit_group.get_current_pose().pose
            w0 = copy.deepcopy(p)
            w0.position.x += 0.10
            w0.position.y += 0.10
            quat_xyzw = Rotation.from_euler('xyz', np.array([0.0, 0.0, -45]),
                                            degrees=True).as_quat()
            quat_wxyz = [quat_xyzw[-1], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            tar_quat = qmul([w0.orientation.w, w0.orientation.x, w0.orientation.y, w0.orientation.z],
                            quat_wxyz)
            w0.orientation.w = tar_quat[0]
            w0.orientation.x = tar_quat[1]
            w0.orientation.y = tar_quat[2]
            w0.orientation.z = tar_quat[3]
            w1 = copy.deepcopy(w0)
            w1.position.z -= 0.04
            w2 = copy.deepcopy(w1)
            w2.position.x -= 0.15
            w2.position.y -= 0.15
            w3 = copy.deepcopy(w2)
            w3.position.z += 0.12
            self.plan_and_execute([p, w0, w1, w2, w3])
            # self.plan_and_show([p, w0, w1, w2, w3], save_tr=True, tr_name='sys_id_3')

        plan = self.moveit_group.plan(joints=waiting_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        return TaskResponse()

    def execute_skill(self, req):
        plan = self.moveit_group.plan(joints=pre_manipulation_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        rospy.sleep(1.0)
        p = self.moveit_group.get_current_pose().pose
        waypoints_dig0 = get_skill_waypoints(p, [0.34867424, -0.19277918, -0.00096845, -0.12814793, -0.03206819], skill='dig')
        self.plan_and_execute(waypoints_dig0)
        rospy.sleep(1.0)
        plan = self.moveit_group.plan(joints=waiting_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        input("Press Enter to continue...")

        plan = self.moveit_group.plan(joints=pre_manipulation_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        rospy.sleep(1.0)
        waypoints_dig1 = get_skill_waypoints(p, [0.74840283, -0.00368124, 0.00075751, 0.3649089, 0.47189257], skill='dig')
        self.plan_and_execute(waypoints_dig1)
        rospy.sleep(1.0)
        plan = self.moveit_group.plan(joints=waiting_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        input("Press Enter to continue...")

        plan = self.moveit_group.plan(joints=pre_manipulation_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        rospy.sleep(1.0)
        waypoints_dig2 = get_skill_waypoints(p, [0.50663837, -0.60888322, 0.04501462, -0.21178022, 0.13505678], skill='dig')
        self.plan_and_execute(waypoints_dig2)
        rospy.sleep(1.0)
        plan = self.moveit_group.plan(joints=waiting_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        input("Press Enter to continue...")

        plan = self.moveit_group.plan(joints=pre_manipulation_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)
        rospy.sleep(1.0)
        waypoints_dig3 = get_skill_waypoints(p, [0.24303342, -0.09475144, 0.00259957, -0.29466704, 0.07773928], skill='dig')
        self.plan_and_execute(waypoints_dig3)
        rospy.sleep(1.0)
        plan = self.moveit_group.plan(joints=waiting_joint_state.position)
        self.moveit_group.execute(plan[1], wait=True)

        waypoints_smooth0 = get_skill_waypoints(p, [6.3202399e-01, 7.6014292e-04, -7.9036134e-01, 1.0000000e+00], skill='smooth')
        waypoints_smooth1 = get_skill_waypoints(p, [0.54956645, -0.04671443, -0.37464836, 1.], skill='smooth')
        waypoints_smooth2 = get_skill_waypoints(p, [4.1230908e-01, 7.7734629e-05, 7.4184136e-03, 4.3179125e-01], skill='smooth')
        waypoints_smooth3 = get_skill_waypoints(p, [6.3202399e-01, 7.6014292e-04, -7.9036134e-01, 1.0000000e+00], skill='smooth')

        return SkillResponse()

    def compose_cartesian_waypoints(self, pose0, pose1):
        # todo: quat to euler is problematic
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
                                   self.delta_position)      # eef_step
                                   #0.0)         # jump_threshold
        # moveit sometimes uses the same time value for the last two trajectory points, causing failure execution
        for n in range(len(plan.joint_trajectory.points)):
            if n > 0 and plan.joint_trajectory.points[n].time_from_start.secs == plan.joint_trajectory.points[n-1].time_from_start.secs:
                while plan.joint_trajectory.points[n].time_from_start.nsecs <= plan.joint_trajectory.points[n-1].time_from_start.nsecs:
                    plan.joint_trajectory.points[n].time_from_start.nsecs += 10000

        self.moveit_group.execute(plan, wait=True)
        dt = plan.joint_trajectory.points[-1].time_from_start.secs + plan.joint_trajectory.points[-1].time_from_start.nsecs / 1e9
        rospy.loginfo("Time spent: "+str(dt)+" secs")

    def plan_and_show(self, waypoints, show=False, save_tr=False, tr_name='tr'):
        print(waypoints)
        (plan, fraction) = self.moveit_group.compute_cartesian_path(
                                   waypoints,   # waypoints to follow
                                   self.delta_position)      # eef_step
                                   #0.0)         # jump_threshold

        # moveit sometimes uses the same time value for the last two trajectory points, causing failure execution
        if plan.joint_trajectory.points[-2].time_from_start.nsecs == plan.joint_trajectory.points[-1].time_from_start.nsecs:
            plan.joint_trajectory.points[-1].time_from_start.nsecs += 10000

        if show or save_tr:
            self.plot_eef_v(plan, save_tr, tr_name)
        return plan

    def current_pose_callback(self, data):
        self.current_joint_msg = data
        self.current_pose_msg = self.moveit_group.get_current_pose()
        self.current_xyz = np.array([
            self.current_pose_msg.pose.position.x,
            self.current_pose_msg.pose.position.y,
            self.current_pose_msg.pose.position.z
        ])
        self.current_euler = Rotation.from_quat([self.current_pose_msg.pose.orientation.x,
                                                 self.current_pose_msg.pose.orientation.y,
                                                 self.current_pose_msg.pose.orientation.z,
                                                 self.current_pose_msg.pose.orientation.w]).as_euler('xyz', degrees=True)
        self.current_header_seq = data.header.seq

    def print_pose(self, req):
        rospy.loginfo("Current pose: ")
        rospy.loginfo("Position: x={}, y={}, z={}".format(self.current_xyz[0],
                                                          self.current_xyz[1],
                                                          self.current_xyz[2]))
        rospy.loginfo("Euler (deg): a={}, b={}, c={}".format(self.current_euler[0],
                                                             self.current_euler[1],
                                                             self.current_euler[2]))
        rospy.loginfo("Euler (rad): a={}, b={}, c={}".format(np.radians(self.current_euler[0]),
                                                             np.radians(self.current_euler[1]),
                                                             np.radians(self.current_euler[2])))
        rospy.loginfo("Orientation: x={}, y={}, z={}, w={}".format(self.current_pose_msg.pose.orientation.x,
                                                                   self.current_pose_msg.pose.orientation.y,
                                                                   self.current_pose_msg.pose.orientation.z,
                                                                   self.current_pose_msg.pose.orientation.w))
        return PrintPoseResponse()

    def plot_eef_v(self, plan, save=False, tr_name='tr'):
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.join(script_dir, '..', 'src', 'moveit_trajectories')
        cartesian_positions = []
        cartesian_velocities = []
        time_frames = []
        time_difference = [0.0]
        dps = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        for i in range(len(plan.joint_trajectory.points)-1):
            position = plan.joint_trajectory.points[i].positions
            velocity = plan.joint_trajectory.points[i].velocities
            jacobian = self.moveit_group.get_jacobian_matrix(list(position))
            cartesian_velocity = np.dot(jacobian, np.array(velocity))
            cartesian_velocities.append(cartesian_velocity)
            time_frames.append(plan.joint_trajectory.points[i].time_from_start.secs + plan.joint_trajectory.points[i].time_from_start.nsecs / 1e9)
            if i > 0:
                time_difference.append(time_frames[-1] - time_frames[-2])
                dp = (time_difference[-1] * cartesian_velocity) + dps[-1]
                dps.append(dp)
            # print('Timestamp: {}'.format(time_frames[-1]))
            # print('Time difference: {}'.format(time_difference[-1]))
            # print('Cartesian velocity: {}'.format(cartesian_velocity))

        # legends = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        # plt.plot(cartesian_velocities)
        # plt.legend(legends)
        # plt.xlabel('Horizon')
        # plt.ylabel('Velocity')
        # plt.title('End-effector velocity')
        # plt.savefig(os.path.join(data_dir, tr_name+'_eef_v.png'), bbox_inches='tight', dpi=300)
        # plt.close()
        #
        # plt.plot(dps)
        # plt.legend(legends)
        # plt.xlabel('Horizon')
        # plt.ylabel('Pose')
        # plt.title('End-effector pose')
        # plt.savefig(os.path.join(data_dir, tr_name+'_eef_pose.png'), bbox_inches='tight', dpi=300)
        # plt.close()
        #
        # plt.plot(time_frames)
        # plt.xlabel('Horizon')
        # plt.ylabel('Timestamps')
        # plt.title('Trajectory timestamps')
        # plt.savefig(os.path.join(data_dir, tr_name+'_timestamps.png'), bbox_inches='tight', dpi=300)
        # print('Time frames: {}'.format(time_frames[-1]))
        # plt.close()
        #
        # plt.plot(time_difference)
        # plt.xlabel('Horizon')
        # plt.ylabel('Time difference')
        # plt.title('Trajectory waypoint time difference')
        # plt.savefig(os.path.join(data_dir, tr_name+'_time_diff.png'), bbox_inches='tight', dpi=300)
        # plt.close()

        if save:
            n = 0
            while os.path.exists(os.path.join(data_dir, tr_name+'_v_'+str(n)+'.npy')):
                n += 1
            np.save(os.path.join(data_dir, tr_name+'_v_'+str(n)+'.npy'), np.array(cartesian_velocities))
            np.save(os.path.join(data_dir, tr_name+'_timestamps_'+str(n)+'.npy'), np.array(time_frames))


if __name__ == '__main__':
    controller = Controller()
    rospy.spin()
