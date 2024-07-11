import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


resting_joint_state = JointState()
resting_joint_state.position = [1.5731966495513916, -3.0082599125304164, 2.686049524937765, -4.316548009912008, -1.604182545338766, 1.5630271434783936]
pre_manipulation_joint_state = JointState()
pre_manipulation_joint_state.position = [1.0184741020202637, -1.0002725285342713, 1.5955079237567347, -1.9981147251524867, -1.3647955099688929, 0.9977901577949524]
waiting_joint_state = JointState()
waiting_joint_state.position = [0.5248939990997314, -2.085085531274313, 2.2386086622821253, -1.998213907281393, -1.3647540251361292, 0.9977448582649231]


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
