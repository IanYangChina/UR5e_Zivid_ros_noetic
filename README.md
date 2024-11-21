### UR5e manipualtion scripts

##### Robotics and Autonomous Systems Lab
##### Robot Perception And Learning (RoPAL) - Cardiff University

<img src="/doc/CardiffUnivLogo.jpg" width="80"/>

It is tested on:
- Ubuntu 20.04
- Python 2.7 + 3.7
- Ros Noetic

Dependencies
- https://github.com/UniversalRobots/Universal_Robots_ROS_Driver
- https://github.com/ros-industrial/universal_robot
- https://github.com/zivid/zivid-ros
- Install `transform3d`, `rosnumpy`

Troubleshoot
- ROS master machine ip: 192.168.0.190
- UR5e ip: 192.168.0.210
- If `ros` can't find packages installed via `apt`: run `rospack profile` to refresh cache

Command lines
- `roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.0.210`