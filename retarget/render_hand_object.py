from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tyro
import os
import torch
from torch_cluster import fps
import pytorch_kinematics as pk
import open3d as o3d
from pytorch3d.transforms import (
    quaternion_to_matrix, matrix_to_quaternion, axis_angle_to_quaternion
)

from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer
from vis_utils import vis_dex_frames_plotly

import warnings; warnings.filterwarnings("ignore", category=UserWarning)

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def farthest_point_sampling(pos: torch.FloatTensor, n_sampling: int):
    bz, N = pos.size(0), pos.size(1)
    feat_dim = pos.size(-1)
    device = pos.device
    sampling_ratio = float(n_sampling / N)
    pos_float = pos.float()

    batch = torch.arange(bz, dtype=torch.long).view(bz, 1).to(device)
    mult_one = torch.ones((N,), dtype=torch.long).view(1, N).to(device)

    batch = batch * mult_one
    batch = batch.view(-1)
    pos_float = pos_float.contiguous().view(-1, feat_dim).contiguous() # (bz x N, 3)
    # sampling_ratio = torch.tensor([sampling_ratio for _ in range(bz)], dtype=torch.float).to(device)
    # batch = torch.zeros((N, ), dtype=torch.long, device=device)
    sampled_idx = fps(pos_float, batch, ratio=sampling_ratio, random_start=True)
    # shape of sampled_idx?
    return sampled_idx
  
def pt_transform(points, transformation):
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation @ points_homogeneous.T).T
    return transformed_points[:, :3]





def viz_hand_object(robots: Optional[Tuple[RobotName]], data_root: Path, fps: int):
    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    if robots is None:
        viewer = HandDatasetSAPIENViewer(headless=True, use_ray_tracing=True)
    else:
        viewer = RobotHandDatasetSAPIENViewer(
            list(robots), HandType.right, headless=True, use_ray_tracing=True
        )

    # Data ID, feel free to change it to visualize different trajectory
    data_id = 4

    sampled_data = dataset[data_id]
    for key, value in sampled_data.items():
        if "pose" not in key:
            print(f"{key}: {value}")
    viewer.load_object_hand(sampled_data)
    viewer.retarget_and_save(sampled_data, data_id)
    # viewer.render_dexycb_data(sampled_data, data_id, fps)


def main(dexycb_dir: str, robots: Optional[List[RobotName]] = None, fps: int = 10):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    The human trajectory is visualized as provided, while the robot trajectory is generated from position retargeting

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to render, if None, render human hand trajectory only
        fps: frequency to render hand-object trajectory

    """
    data_root = Path(dexycb_dir).absolute()
    robot_dir = (
        Path("/home/qianxu/Desktop/Project/DexPose/thirdparty/dex-retargeting/assets").absolute() / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")

    viz_hand_object(robots, data_root, fps)


def test(robot_name: Optional[List[RobotName]] = RobotName.allegro, dexycb_dir: str = '/home/qianxu/Desktop/Project/interaction_pose/thirdparty_module/dex-retargeting/data', hand_type: str = "right", fps: int = 10):

    robot_name_str = str(robot_name).split(".")[-1]
    data_root = Path(dexycb_dir).absolute()
    dataset = DexYCBVideoDataset(data_root, hand_type=hand_type)

    data_id = 4
    sampled_data = dataset[data_id]

    def load_robot(robot_name_str: str, side):

        urdf_files = f'/home/qianxu/Desktop/Project/DexPose/retarget/urdf/{robot_name_str}_hand_{side}_glb.urdf'

        # Load the URDF file for the robot
        with open(urdf_files, 'rb') as f:
            urdf_str = f.read()

        # Construct the kinematic chain from the URDF
        robot_chain = pk.build_chain_from_urdf(urdf_str)
        
        for i, jt in enumerate(robot_chain.get_joint_parameter_names()):
            print(i, jt)
        robot_chain.print_tree()
        
        for joint_name in robot_chain.get_joint_parameter_names():
            for link in robot_chain.links:
                if link.joint.name == joint_name:
                    parent_link = link.parent
                    child_link = link.name
                    print(f"{joint_name}: {parent_link} - {child_link}")
                    break

        return robot_chain

    def get_point_clouds(data: dict):
        ycb_mesh_files = data["object_mesh_file"] # a list of .obj file path
        meshes = []
        for mesh_file in ycb_mesh_files:
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            meshes.append(mesh)
        
        original_pc = [np.asarray(mesh.vertices) for mesh in meshes if mesh is not None]

        original_pc_ls = [
                farthest_point_sampling(torch.from_numpy(points).unsqueeze(0), 1000)[:1000]  for points in original_pc
            ]
        pc_ds = [pc[pc_idx] for pc, pc_idx in zip(original_pc, original_pc_ls)]

        return pc_ds[0] # only 1 object in the dataset

    def get_finger_group(robot_name_str: str):
        if robot_name_str == "allegro":
            return {
                "finger 1": [0, 1, 2, 3],
                "finger 2": [4, 5, 6, 7],
                "finger 3": [8, 9, 10, 11],
                "finger 4": [12, 13, 14, 15],
            }
        else:
            raise ValueError(f"Finger group for {robot_name_str} not defined.")

    robot_chain = load_robot(robot_name, hand_type)
    finger_groups = get_finger_group(str(robot_name))

    qpos_file = f'/home/qianxu/Desktop/Project/DexPose/retarget/hand_qpos/{robot_name_str}_seq_{data_id}_from_0_qpos.npy'
    qpos = np.load(qpos_file)

    keypoints =[]
    # Iterate through the qpos and compute the forward kinematics
    for i in range(qpos.shape[0]):
        ret = robot_chain.forward_kinematics(qpos[i])
        keypoints.append([i._matrix[:, :3, 3] for i in list(ret.values())[6:]])
    hand_joints = np.array(keypoints)

    pc = get_point_clouds(sampled_data)
    object_pose = sampled_data["object_pose"] # T x 7, 7=q+t, q=xyzw, t=xyz
    pc_ls = []
    for i in range(object_pose.shape[0]):
        pose = torch.from_numpy(object_pose[i])
        transformation = torch.eye(4)
        transformation[:3, :3] = quaternion_to_matrix(torch.cat((pose[3:4], pose[:3])))
        transformation[:3, 3] = pose[4:]
        transformed_pc = pt_transform(pc, transformation)
        pc_ls.append(transformed_pc)

    vis_dex_frames_plotly(
        pc_ls=pc_ls,
        gt_hand_joints=hand_joints,
        finger_groups=finger_groups,
        show_axis=True,
        filename="test"
    )    



if __name__ == "__main__":
    # tyro.cli(main)
    tyro.cli(test)
