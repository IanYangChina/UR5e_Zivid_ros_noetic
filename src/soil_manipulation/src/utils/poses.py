import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


resting_joint_state = JointState()
resting_joint_state.position = [1.5731966495513916, -3.0082599125304164, 2.686049524937765, -4.316548009912008, -1.604182545338766, 1.5630271434783936]
pre_manipulation_joint_state = JointState()
pre_manipulation_joint_state.position = [1.303176760673523, -0.91327919185672, 1.81486684480776, -2.4488688908019007, -1.5766990820514124, 1.3097814321517944]
waiting_joint_state = JointState()
waiting_joint_state.position = [0.41005825996398926, -1.212908999328949, 1.6579297224627894, -1.993023534814352, -1.575240437184469, 1.3089301586151123]


def construct_homogeneous_transform_matrix(translation, orientation):
    translation = np.array(translation).reshape((3, 1))  # xyz
    if len(orientation) == 4:
        rotation = Rotation.from_quat(np.array(orientation).reshape((4, 1)))  # xyzw
    else:
        assert len(orientation) == 3, 'orientation should be a quaternion or 3 axis angles'
        rotation = np.radians(np.array(orientation).astype("float")).reshape((3, 1))  # xyz in radians
        rotation = Rotation.from_euler('xyz', rotation).as_matrix()
    transformation = np.append(rotation, translation, axis=1)
    transformation = np.append(transformation, np.array([[0, 0, 0, 1]]), axis=0)
    return transformation
