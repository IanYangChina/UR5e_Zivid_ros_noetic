import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


resting_joint_state = JointState()
resting_joint_state.position = [1.5731966495513916, -3.0082599125304164, 2.686049524937765, -4.316548009912008, -1.604182545338766, 1.5630271434783936]
pre_manipulation_joint_state0 = JointState()
pre_manipulation_joint_state0.position = [1.3307002782821655, -1.1519576472095032, 1.702266041432516, -2.0840684376158656, -1.5801962057696741, 1.3920601606369019]
pre_manipulation_joint_state = JointState()
pre_manipulation_joint_state.position = [1.3340355157852173, -0.8765847247890015, 1.7732933203326624, -2.430845399896139, -1.5812013784991663, 1.3963029384613037]
waiting_joint_state = JointState()
waiting_joint_state.position = [0.6153032183647156, -1.2132848066142579, 1.6578343550311487, -1.9928943119444789, -1.5750649611102503, 1.30869460105896]


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
