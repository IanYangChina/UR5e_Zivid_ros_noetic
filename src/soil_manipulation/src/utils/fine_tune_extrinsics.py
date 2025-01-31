import os
import yaml
import open3d as o3d
import numpy as np
np.printoptions(suppress=True, precision=4)
from copy import deepcopy as dcp


def construct_homogeneous_transform_matrix(translation, orientation):
    translation = np.array(translation).reshape((3, 1))  # xyz
    if len(orientation) == 4:
        rotation = o3d.geometry.get_rotation_matrix_from_quaternion(np.array(orientation).reshape((4, 1)))  # wxyz
    else:
        assert len(orientation) == 3, 'orientation should be a quaternion or 3 axis angles'
        rotation = np.radians(np.array(orientation).astype("float")).reshape((3, 1))  # CBA in radians
        rotation = o3d.geometry.get_rotation_matrix_from_zyx(rotation)
    transformation = np.append(rotation, translation, axis=1)
    transformation = np.append(transformation, np.array([[0, 0, 0, 1]]), axis=0)
    return transformation


script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
# Load the camera extrinsics
with open(os.path.join(script_path, 'extrinsics.yml'), 'r') as f:
    cam_extrinsics = yaml.load(f, Loader=yaml.FullLoader)
transform_world_to_cam = cam_extrinsics['matrix']
transform_cam_to_world = np.linalg.inv(transform_world_to_cam)

# Create a plane
mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.5, height=1.5, depth=0.001)
mesh_box.translate([-0.25, -0.25, 0.0])

# Create frames
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
cam_frame.transform(transform_world_to_cam)

# Load PCD
pcd = o3d.io.read_point_cloud(os.path.join(script_path, 'pcds', 'pcd_0.ply'))
pcd_in_world_frame = pcd.transform(transform_world_to_cam)

# Visualize
o3d.visualization.draw_geometries([
    world_frame,
    cam_frame,
    mesh_box,
    pcd_in_world_frame
], width=800, height=600)

done = False
while not done:
    delta_x = float(input("Enter delta x: "))
    delta_y = float(input("Enter delta y: "))
    delta_z = float(input("Enter delta z: "))
    delta_rx = float(input("Enter delta rx: "))
    delta_ry = float(input("Enter delta ry: "))
    delta_rz = float(input("Enter delta rz: "))
    print(f"Alternate the tranformation matrix by the following deltas: \n"
          f"x {delta_x}, y {delta_y}, z {delta_z}, rx {delta_rx}, ry {delta_ry}, rz {delta_rz}.")
    T = construct_homogeneous_transform_matrix([delta_x, delta_y, delta_z], [delta_rz, delta_ry, delta_rx])
    new_transform_world_to_cam = T @ transform_world_to_cam
    pcd = dcp(pcd_in_world_frame)
    pcd = pcd.transform(T)
    new_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    new_cam_frame.transform(new_transform_world_to_cam)
    o3d.visualization.draw_geometries([
        world_frame,
        new_cam_frame,
        mesh_box,
        pcd
    ], width=800, height=600)
    done = input("Done? (y/n): ") == 'y'

print(f"  - {new_transform_world_to_cam[0]}")
print(f"  - {new_transform_world_to_cam[1]}")
print(f"  - {new_transform_world_to_cam[2]}")
print(f"  - {new_transform_world_to_cam[3]}")
np.save(os.path.join(script_path, 'cam_extrinsics_fine_tuned.npy'), new_transform_world_to_cam)