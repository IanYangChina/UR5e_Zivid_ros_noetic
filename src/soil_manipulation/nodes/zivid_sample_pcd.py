#!/usr/bin/env python3

import os
import rospy
import roslaunch
import ros_numpy
import open3d as o3d
from zivid_camera.srv import *
from sensor_msgs.msg import PointCloud2

script_path = os.path.dirname(os.path.realpath(__file__))


class Sample:
    def __init__(self):
        rospy.init_node("pcd_sample_node", anonymous=True)
        rospy.loginfo("Starting pcd_sample_node")
        rospy.Subscriber("/zivid_camera/points/xyzrgba", PointCloud2, self.on_points, queue_size=2)
        ca_suggest_settings_service = "/zivid_camera/capture_assistant/suggest_settings"
        rospy.wait_for_service(ca_suggest_settings_service)
        self.capture_assistant_service = rospy.ServiceProxy(
            ca_suggest_settings_service, CaptureAssistantSuggestSettings
        )
        self.capture_service = rospy.ServiceProxy("/zivid_camera/capture", Capture)
        self.capture_assistant_suggest_settings()

    def on_points(self, data):
        rospy.loginfo("PointCloud received")
        cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(data)
        points = ros_numpy.point_cloud2.get_xyz_points(cloud_array, remove_nans=True)
        pcd_raw = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
        raw_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        o3d.visualization.draw_geometries([raw_cam_frame, pcd_raw])
        os.makedirs(os.path.join(script_path, '..', 'src', 'pcds'), exist_ok=True)
        path_to_save_pcd = os.path.join(script_path, '..', 'src', 'pcds', 'pcd_0.ply')
        i = 0
        while True:
            if os.path.exists(path_to_save_pcd):
                i += 1
                path_to_save_pcd = os.path.join(script_path, '..', 'src', 'pcds', 'pcd_'+str(i)+'.ply')
            else:
                break
        o3d.io.write_point_cloud(path_to_save_pcd, pcd_raw)
        rospy.loginfo("Point cloud has been saved in ../src/")

    def capture_assistant_suggest_settings(self):
        max_capture_time = rospy.Duration.from_sec(1.20)
        rospy.loginfo(
            "Calling capture assistant service with max capture time = %.2f sec",
            max_capture_time.to_sec(),
        )
        self.capture_assistant_service(
            max_capture_time=max_capture_time,
            ambient_light_frequency=CaptureAssistantSuggestSettingsRequest.AMBIENT_LIGHT_FREQUENCY_NONE,
        )

    def capture(self):
        rospy.loginfo("Calling capture service")
        self.capture_service()


if __name__ == '__main__':
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch_pcd = roslaunch.parent.ROSLaunchParent(uuid, [os.path.join(script_path, '..', 'launch', 'camera.launch')])
    launch_pcd.start()
    rospy.sleep(5)
    sample = Sample()
    while True:
        sample.capture()
        rospy.sleep(2)
        ans = input("[USER INPUT] Is the point cloud satisfactory? [y/n]")
        if ans == 'n':
            continue
        elif ans == 'y':
            break
        else:
            print("Invalid input: ", ans)

    rospy.sleep(2)
    rospy.loginfo("Exiting program...")
    launch_pcd.shutdown()
    exit()
