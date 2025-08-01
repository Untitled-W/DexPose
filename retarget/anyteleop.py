from pathlib import Path
from typing import Optional, Tuple, List


import numpy as np
from tqdm import tqdm, trange
import os
import random
import pickle
from torch_cluster import fps
import pytorch_kinematics as pk
import open3d as o3d
from pytorch3d.transforms import quaternion_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix
from pytransform3d import transformations as pt

from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig

from utils.viewer import RobotHandDatasetSAPIENViewer
from dataset.base_structure import DexSequenceData, HUMAN_SEQ_PATH, DEX_SEQ_PATH
from utils.vis_utils import visualize_dex_hand_sequence

import warnings; warnings.filterwarnings("ignore", category=UserWarning)


# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


DATASET_CONFIGS = {

    'oakinkv2': {
        'processor_name': 'oakinkv2',
        'load_path': HUMAN_SEQ_PATH['Oakinkv2'],
        'save_path': DEX_SEQ_PATH['Oakinkv2'],
        'task_interval': 20,
        'which_dataset': 'Oakinkv2',
        'seq_data_name': 'retarget',
        'sequence_indices': list(range(0, 1))  # Example sequence indices for processing
    },
    
    'taco': {
        'processor_name': 'taco',
        'load_path': HUMAN_SEQ_PATH['Taco'],
        'save_path': DEX_SEQ_PATH['Taco'],
        'task_interval': 1,
        'which_dataset': 'Taco',
        'seq_data_name': 'retarget',
        'sequence_indices': list(range(0, 1))  # Example sequence indices for processing
    },

    'dexycb': {
        'processor_name': 'dexycb',
        'load_path': HUMAN_SEQ_PATH['DexYCB'],
        'save_path': DEX_SEQ_PATH['DexYCB'],
        'task_interval': 1,
        'which_dataset': 'DexYCB',
        'seq_data_name': 'retarget',
        'sequence_indices': list(range(0, 1))  # Example sequence indices for processing
    }
}


def main_retarget(dataset_names: List[str], robots: List[RobotName]):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    The human trajectory is visualized as provided, while the robot trajectory is generated from position retargeting

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to render, if None, render human hand trajectory only
        fps: frequency to render hand-object trajectory

    """
    processed_data = []

    for dataset_name in dataset_names:
        file_path = os.path.join(DATASET_CONFIGS[dataset_name]['load_path'],f'seq_{DATASET_CONFIGS[dataset_name]["seq_data_name"]}_{DATASET_CONFIGS[dataset_name]["task_interval"]}.p')
        with open(file_path, 'rb') as f:
            load_data = pickle.load(f)

        for idx in tqdm(DATASET_CONFIGS[dataset_name]['sequence_indices'], desc=f"Processing {dataset_name}"):
            sampled_data = load_data[idx]
            hand_side = HandType.right if sampled_data.side == 1 else HandType.left
            if hand_side == HandType.left: continue
            viewer = RobotHandDatasetSAPIENViewer(
                list(robots), hand_side, headless=True, use_ray_tracing=True
            )
            data_id = sampled_data.which_dataset + "_" + sampled_data.which_sequence

            import torch
            viewer.load_object_hand_dex(sampled_data)
            outputs = viewer.retarget_and_save_dex(sampled_data, data_id)
            processed_data.extend(outputs)

    return processed_data

def save_results(dex_data: List[DexSequenceData]):
    dataset_data = {}
    for data in dex_data:
        name = data.which_dataset.lower() + "_" + data.which_hand
        if name not in dataset_data:
            dataset_data[name] = []
        dataset_data[name].append(data)

    for name, data_list in dataset_data.items():
        dataset_name, robot_name = name.split("_")
        save_path = DATASET_CONFIGS[dataset_name]['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, f'seq_{robot_name}_{DATASET_CONFIGS[dataset_name]["seq_data_name"]}_{DATASET_CONFIGS[dataset_name]["task_interval"]}.p')
        with open(file_path, 'wb') as f:
            pickle.dump(data_list, f)

def check_data_correctness_by_vis(dex_data: List[DexSequenceData]):
    """
    Check data correctness by visualizing a few sequences.
    
    Args:
        dex_data: List of HumanSequenceData objects
    """
    
    dataset_data = {}
    for data in dex_data:
        name = data.which_dataset + "_" + data.which_hand
        if name not in dataset_data:
            dataset_data[name] = []
        dataset_data[name].append(data)

    for name, data_list in dataset_data.items():
        print(f"{name} has {len(data_list)} sequences")
        # Sample a few sequences for visualization
        sampled_data = data_list
        # sampled_data = random.sample(data_list, 2)
        for d in sampled_data:
            print(f"Visualizing sequence {d.which_sequence}")
            visualize_dex_hand_sequence(d, f'/home/qianxu/Desktop/Project/DexPose/retarget/vis_results/{d.which_hand}_{d.which_dataset}_{d.which_sequence}.html')



if __name__ == "__main__":

    dataset_names = ['dexycb', 'taco', 'oakinkv2']
    # robots = [RobotName.allegro, RobotName.shadow, RobotName.leap]
    robots = [RobotName.inspire]
    robot_dir = (
        Path("/home/qianxu/Desktop/Project/DexPose/thirdparty/dex-retargeting/assets").absolute() / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    processed_data = []
    
    GENERATE = True
    if GENERATE:
        processed_data = main_retarget(dataset_names, robots)
        # save_results(processed_data)
    else:
        for dataset_name in dataset_names:
            for robot in robots:
                robot_name_str = str(robot).split(".")[-1]
                file_path = os.path.join(DATASET_CONFIGS[dataset_name]['save_path'], f'seq_{robot_name_str}_{DATASET_CONFIGS[dataset_name]["seq_data_name"]}_{DATASET_CONFIGS[dataset_name]["task_interval"]}.p')
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist, skipping...")
                    continue
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                processed_data.extend(data)

    # show_human_statistics(processed_data)
    
    check_data_correctness_by_vis(processed_data)