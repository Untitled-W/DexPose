import torch
import numpy as np
import open3d as o3d
from utils.wmq import visualize_hand_and_joints
from .robot_wrapper import load_robot, HandRobotWrapper

def quat_to_aa_wmq_old(quat, side='right'):
    from pytransform3d import rotations
    operator2mano = {
        "right": np.array([
                    [0, 0, -1],
                    [-1, 0, 0],
                    [0, 1, 0]]),
        "left": np.array([
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, -1, 0]])}
    euler = rotations.euler_from_matrix(
            rotations.matrix_from_quaternion(quat) @ operator2mano[side], 0, 1, 2, extrinsic=False
        )
    return torch.from_numpy(euler)

def quat_to_aa_wmq(quat, side='right'):
    """
    quat is not batched
    """
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles
    operator2mano = {
        "right": np.array([
                    [0, 0, -1],
                    [-1, 0, 0],
                    [0, 1, 0]]),
        "left": np.array([
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, -1, 0]])}
    euler = matrix_to_euler_angles(quaternion_to_matrix(quat) @ operator2mano[side], 'XYZ')
    return euler

# robot_name = "shadow_hand"
# robot_name = 'inspire_hand'
# robot_name = 'leap_hand'
robot_name = 'allegro_hand'
# robot_name = 'schunk_hand'

robot_hand = load_robot(robot_name)  # type: HandRobotWrapper
qq = torch.zeros(robot_hand.n_dofs)
qq[3:6] = quat_to_aa_wmq_old(torch.zeros(4))
robot_hand.compute_forward_kinematics(qq)

from utils.tools import extract_hand_points_and_mesh
human_keypoints, _ = extract_hand_points_and_mesh(torch.ones(3)*0.1, torch.zeros((16,4)), "right")

robot_link_mesh = {}
for link_name in robot_hand.mesh:
    v_tensor = robot_hand.current_status[link_name].transform_points(robot_hand.mesh[link_name]['vertices'])
    v_numpy = v_tensor.detach().cpu().numpy()
    f_numpy = robot_hand.mesh[link_name]['faces'].detach().cpu().numpy()
    link_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(v_numpy),
        triangles=o3d.utility.Vector3iVector(f_numpy)
    )
    robot_link_mesh[link_name] = link_mesh
    print(f'"{link_name}"',":[],")
robot_approx_mesh = {}
for link_name in robot_hand.link_approx_names:
    v_tensor = robot_hand.current_status[link_name].transform_points(robot_hand.mesh[link_name]['c_vertices'])
    v_numpy = v_tensor.detach().cpu().numpy()
    f_numpy = robot_hand.mesh[link_name]['c_faces'].detach().cpu().numpy()
    link_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(v_numpy),
        triangles=o3d.utility.Vector3iVector(f_numpy)
    )
    robot_approx_mesh[link_name] = link_mesh

robot_keypoints = robot_hand.get_joint_world_coordinates_dict()
from utils.wmq import visualize_hand_and_joints
visualize_hand_and_joints(
    mano_joint=human_keypoints[0],
    robot_keypoints=robot_keypoints,
    robot_link_mesh=robot_link_mesh,
    robot_approx_mesh=robot_approx_mesh,
    human_keypoints=human_keypoints[0],
    contact_points=robot_hand.get_contact_candidates(),
    surface_points=robot_hand.get_surface_points(),
    penetration_keypoints=robot_hand.get_penetration_keypoints(),
    filename="retarget_wmq/vis_hand/1{}".format(robot_name)
)

