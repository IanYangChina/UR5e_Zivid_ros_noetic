cmake_minimum_required(VERSION 3.0.2)
project(soil_manipulation)

find_package(catkin REQUIRED message_generation std_msgs)

catkin_python_setup()

add_service_files(FILES
    TargetPose.srv
    Reset.srv
    MoveDistance.srv
)

generate_messages(DEPENDENCIES std_msgs)
catkin_package(CATKIN_DEPENDS message_runtime std_msgs)

catkin_install_python(PROGRAMS
    nodes/send_pose_cmd.py
    nodes/ur5e_state_listener.py
    nodes/ur5e_controller_moveit.py
    nodes/keyboard_event_node.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
    launch/ur5e_moveit_controller.launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
