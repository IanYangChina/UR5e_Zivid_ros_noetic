cmake_minimum_required(VERSION 3.0.2)
project(soil_manipulation)

find_package(catkin REQUIRED message_generation std_msgs)

catkin_python_setup()

add_service_files(FILES
    TargetPose.srv
    Reset.srv
    MoveDistance.srv
    PrintPose.srv
    Rest.srv
    Skill.srv
)

generate_messages(DEPENDENCIES std_msgs)
catkin_package(CATKIN_DEPENDS message_runtime std_msgs)

catkin_install_python(PROGRAMS
    nodes/send_pose_cmd.py
    nodes/ur5e_state_listener.py
    nodes/ur5e_controller_moveit.py
    nodes/keyboard_event_node.py
    nodes/zivid_live_2d.py
    nodes/verify_calibration_result_aruco.py
    nodes/zivid_sample_pcd.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
    launch/ur5e_moveit_controller.launch
    launch/ur5e_state_listener.launch
    launch/tracking.launch
    launch/veryfy_calibration_result_aruco.launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
