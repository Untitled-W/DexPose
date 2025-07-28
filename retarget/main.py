from pathlib import Path
from typing import Optional, Tuple, List


import numpy as np
import tyro
import os
import torch
import pickle
from torch_cluster import fps
import pytorch_kinematics as pk
import open3d as o3d
from pytorch3d.transforms import quaternion_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix
from pytransform3d import transformations as pt

from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig

# from .hand_robot_viewer import RobotHandDatasetSAPIENViewer
from utils.viewer import RobotHandDatasetSAPIENViewer
from dataset.base_structure import HumanSequenceData, ORIGIN_DATA_PATH, HUMAN_SEQ_PATH
from utils.vis_utils import *

import warnings; warnings.filterwarnings("ignore", category=UserWarning)


# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_



def viz_hand_object(sampled_data: HumanSequenceData, robots: Optional[Tuple[RobotName]], fps: int):
    robot_dir = (
        Path("/home/qianxu/Desktop/Project/DexPose/thirdparty/dex-retargeting/assets").absolute() / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    
    viewer = RobotHandDatasetSAPIENViewer(
        list(robots), HandType.right, headless=True, use_ray_tracing=True
    )
    data_id = sampled_data.which_dataset + "_" + sampled_data.which_sequence

    viewer.load_object_hand_dex(sampled_data)
    viewer.retarget_and_save_dex(sampled_data, data_id)


DATASET_CONFIGS = {
    'oakinkv2': {
        'processor_name': 'oakinkv2',
        'root_path': ORIGIN_DATA_PATH['Oakinkv2'],
        'save_path': HUMAN_SEQ_PATH['Oakinkv2'],
        'task_interval': 10,
        'which_dataset': 'Oakinkv2',
        'seq_data_name': 'debug',
        'sequence_indices': list(range(0, 5))  # Example sequence indices for processing
    },
    
    'taco': {
        'processor_name': 'taco',
        'root_path': ORIGIN_DATA_PATH['Taco'],
        'save_path': HUMAN_SEQ_PATH['Taco'],
        'task_interval': 1,
        'which_dataset': 'Taco',
        'seq_data_name': 'debug',
        'sequence_indices': list(range(0, 10))  # Example sequence indices for processing
    },

    'dexycb': {
        'processor_name': 'dexycb',
        'root_path': ORIGIN_DATA_PATH['DexYCB'],
        'save_path': HUMAN_SEQ_PATH['DexYCB'],
        'task_interval': 1,
        'which_dataset': 'DexYCB',
        'seq_data_name': 'debug',
        'sequence_indices': list(range(0, 10))  # Example sequence indices for
    }
}


def main_retarget():
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    The human trajectory is visualized as provided, while the robot trajectory is generated from position retargeting

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to render, if None, render human hand trajectory only
        fps: frequency to render hand-object trajectory

    """
    dataset_names = ['dexycb']
    processed_data = []
    for dataset_name in dataset_names:
        file_path = os.path.join(DATASET_CONFIGS[dataset_name]['save_path'],f'seq_{DATASET_CONFIGS[dataset_name]["seq_data_name"]}_{DATASET_CONFIGS[dataset_name]["task_interval"]}.p')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            processed_data.extend(data)

        sampled_data = processed_data[0]  # Example: take the first data item for visualization

        fps = 10
        robots = [RobotName.allegro, RobotName.shadow, RobotName.leap]

        viz_hand_object(sampled_data, robots, fps)



def visualize_qpos(seq_data: HumanSequenceData, robot_name_str: Optional[str] = None):
    """
    Visualize a sequence of dexterous hand movements.
    
    Args:
        seq_data (DexHandSequenceData): The sequence data containing hand poses and meshes.
        filename (Optional[str]): If provided, save the visualization to this file.
    """
    data_id = seq_data.which_dataset + "_" + seq_data.which_sequence
    hand_type = 'left' if seq_data.side == 0 else 'right'
    fps = 10

    robot = load_robot(robot_name_str, hand_type)

    qpos_file = f'/home/qianxu/Desktop/Project/DexPose/retarget/hand_qpos/{robot_name_str}_seq_{data_id}_qpos.npy'
    qpos = np.load(qpos_file)

    ### hand meshes ###
    hand_meshes = []
    for i in range(qpos.shape[0]):
        robot.set_qpos(torch.from_numpy(qpos[i].astype(np.float32)))
        hand_mesh = robot.get_hand_mesh()
        # print(len(hand_mesh.vertices), len(hand_mesh.triangles))
        hand_meshes.append(hand_mesh)


    ### hand joints ###
    hand_joints = []
    
    hand_pose = torch.cat([quaternion_to_axis_angle(seq_data.hand_coeffs).reshape(-1, 48), seq_data.hand_tsls], dim=-1)
    hand_pose = hand_pose.unsqueeze(1).numpy()  # Convert to numpy array

    mano_layer = MANOLayer(hand_type, np.zeros(10).astype(np.float32))
    for i in range(hand_pose.shape[0]):
        joint = compute_hand_geometry(hand_pose[i], mano_layer= mano_layer)
        hand_joints.append(joint)
    hand_joints = torch.tensor(hand_joints, dtype=torch.float32)



    ### point clouds ###
    pc = get_point_clouds_from_human_data(seq_data)
    pc_ls = apply_transformation_human_data(pc, seq_data.obj_poses)
    pc_ls = [np.asarray(pc_ls)] # due to some dim issue


    vis_frames_plotly(
        pc_ls=pc_ls,
        gt_hand_joints=hand_joints,
        hand_mesh=hand_meshes,
        show_axis=True,
        filename=f"/home/qianxu/Desktop/Project/DexPose/retarget/vis_results/{robot_name_str}_seq_{data_id}_qpos"
    )    

    
def test_by_plotly_vis():
    dataset_names = ['dexycb']
    processed_data = []
    for dataset_name in dataset_names:
        file_path = os.path.join(DATASET_CONFIGS[dataset_name]['save_path'],f'seq_{DATASET_CONFIGS[dataset_name]["seq_data_name"]}_{DATASET_CONFIGS[dataset_name]["task_interval"]}.p')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            processed_data.extend(data)

        sampled_data = processed_data[0]  # Example: take the first data item for visualization
        robots = [RobotName.allegro, RobotName.shadow, RobotName.leap]

        for robot_name in robots:
            robot_name_str = str(robot_name).split(".")[-1]
            print(f"Visualizing for robot: {robot_name_str}")
            visualize_qpos(sampled_data, robot_name_str)


if __name__ == "__main__":
    main_retarget()
    test_by_plotly_vis()