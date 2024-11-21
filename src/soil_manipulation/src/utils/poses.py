import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


resting_joint_state = JointState()
resting_joint_state.position = [1.5731966495513916, -3.0082599125304164, 2.686049524937765, -4.316548009912008, -1.604182545338766, 1.5630271434783936]
pre_manipulation_joint_state = JointState()
pre_manipulation_joint_state.position = [1.2400542497634888, -1.0693372052958985, 2.164511028920309, -2.6441108189024867, -1.5791690985309046, 1.2469240427017212]
waiting_joint_state = JointState()
waiting_joint_state.position = [0.9554719924926758, -1.213200883274414, 1.6578214804278772, -1.9930149517455042, -1.5751455465899866, 1.3087184429168701]
# waiting_joint_state.position = [1.7802430391311646, -1.2130856674960633, 1.657809082661764, -1.9930278263487757, -1.575104061757223, 1.3088291883468628]


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
