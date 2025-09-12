from pathlib import Path
from typing import Optional, Tuple, List


import numpy as np
from tqdm import tqdm, trange
import os
import torch
import pickle
from torch_cluster import fps
import pytorch_kinematics as pk
from typing import Any, Dict, List
import open3d as o3d
from pytorch3d.transforms import quaternion_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix
from pytransform3d import transformations as pt

from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig

from utils.viewer import RobotHandDatasetSAPIENViewer
from utils.vis_utils import visualize_dex_hand_sequence, visualize_dex_hand_sequence_together

import warnings; warnings.filterwarnings("ignore", category=UserWarning)


# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def get_piece_by_frame(
    seq_data: Dict[str, Any], 
    frame_slice: slice = None
) -> Dict[str, Any]:
    
    if frame_slice is None:
        return seq_data

    sliced_data = seq_data.copy()
    # --- 对与时间维度相关的字段应用切片 ---

    # 1. 手部数据 (时间维度在 dim=0)
    sliced_data['hand_tsls'] = seq_data['hand_tsls'][frame_slice]
    sliced_data['hand_coeffs'] = seq_data['hand_coeffs'][frame_slice]
    if "hand_poses" in seq_data:
        sliced_data['hand_poses'] = seq_data['hand_poses'][frame_slice]
    if "hand_joints" in seq_data:
        sliced_data['hand_joints'] = seq_data['hand_joints'][frame_slice]

    # 2. 物体位姿数据 (时间维度在 dim=1)
    # 形状为 (K, T, 4, 4)，所以我们在第二个维度上切片
    sliced_data['obj_poses'] = seq_data['obj_poses'][:, frame_slice]

    return sliced_data

def main_retarget(seq_data_ls, robots: List[RobotName]):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    The human trajectory is visualized as provided, while the robot trajectory is generated from position retargeting

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to render, if None, render human hand trajectory only
        fps: frequency to render hand-object trajectory

    """
    from collections import defaultdict as DefaultDict
    processed_data = DefaultDict(list)

    for sampled_data in tqdm(seq_data_ls, desc=f"Processing"):
        hand_side = HandType.right if sampled_data["side"] == 1 else HandType.left
        if hand_side == HandType.left: continue
        viewers = [
            RobotHandDatasetSAPIENViewer(
                list(robots), hand_side, headless=True, use_ray_tracing=True
            ),
            # RobotHandDatasetSAPIENViewer(
            #     list(robots), hand_side, headless=True, use_ray_tracing=True, optimizer_type="position_pc"
            # )
        ]
        data_id = sampled_data["which_dataset"] + "_" + sampled_data["which_sequence"]
        sampled_data['obj_poses'] = sampled_data["obj_poses"] #@ sampled_data['mesh_norm_trans']

        for viewer in viewers:
            viewer.load_object_hand_dex(sampled_data)
            outputs = viewer.retarget_and_save_dex(get_piece_by_frame(sampled_data), data_id)
            processed_data[viewer.optimizer_type].extend(outputs)

    return processed_data

def save_results(dex_data_ls, save_path):

    for optimizer_type, dex_data in dex_data_ls.items():

        dataset_data = {}
        for data in dex_data:
            name = data["which_dataset"].lower() + "-" + data["which_hand"]
            if name not in dataset_data:
                dataset_data[name] = []
            dataset_data[name].append(data)

        for name, data_list in dataset_data.items():
            dataset_name, robot_name = name.split("-")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, f'seq_{dataset_name}_{optimizer_type}_{robot_name}.p')
            with open(file_path, 'wb') as f:
                pickle.dump(data_list, f)

def check_data_correctness_separate(dex_data_ls, root_folder):
    """
    Check data correctness by visualizing a few sequences.
    
    Args:
        dex_data: List of HumanSequenceData objects
    """
    
    for optimizer_type, dex_data in dex_data_ls.items():

        dataset_data = {}
        for data in dex_data:
            name = data['which_dataset'] + "_" + data['which_hand']
            if name not in dataset_data:
                dataset_data[name] = []
            dataset_data[name].append(data)

        for name, data_list in dataset_data.items():
            print(f"{name} has {len(data_list)} sequences")
            import random
            sampled_data = random.sample(data_list, 2)
            # sampled_data = data_list

            for d in sampled_data:
                print(f"Visualizing sequence {d['which_sequence']} with {d['which_hand']} with {optimizer_type}")
                visualize_dex_hand_sequence(d, f'{root_folder}/{optimizer_type}_{d["which_hand"]}_{d["which_dataset"]}_{d["which_sequence"]}_{d["side"]}')

def check_data_correctness_together(dex_data_ls, root_folder):
    """
    Check data correctness by visualizing a few sequences.
    
    Args:
        dex_data: List of HumanSequenceData objects
    """

    name_list = list(dex_data_ls.keys())
    num_seq = len(dex_data_ls[name_list[0]])

    for i in range(num_seq):
        data_ls = [dex_data_ls[name_list[j]][i] for j in range(len(name_list))]
        name = data_ls[0]['which_dataset'] + "_" + data_ls[0]['which_hand'] + "_" + data_ls[0]['which_sequence'] + "_" + str(data_ls[0]['side'])
        visualize_dex_hand_sequence_together(data_ls, name_list,
                   f'{root_folder}/{name}')

if __name__ == "__main__":

    
    # robots = [RobotName.allegro, RobotName.leap, RobotName.inspire, ]
    robots = [RobotName.shadow, ]

    robot_dir = (
        Path("/home/qianxu/Desktop/Project/DexPose/thirdparty/dex-retargeting/assets").absolute() / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    processed_data = []
    
    working_dir = "/home/qianxu/Desktop/Project/DexPose/retarget/0912"

    PROCESS = True
    if PROCESS:
        # file_path = "/home/qianxu/Desktop/Project/DexPose/data_dict_wqx.pth"
        # with open(file_path, "rb") as f:
        #     seq_data_ls = torch.load(f)[:16]
        file_path = "/home/qianxu/Desktop/Project/DexPose/seq_shadow_hand_debug_1.p"
        with open(file_path, "rb") as f:
            seq_data_ls = pickle.load(f)[0:16:4]
        processed_data = main_retarget(seq_data_ls, robots)
        save_results(processed_data, working_dir)
    else:
        folder_path = working_dir
        import os
        from collections import defaultdict as DefaultDict
        processed_data = DefaultDict(list)

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".p"):
                with open(os.path.join(folder_path, file_name), "rb") as f:
                    seq_data_ls = pickle.load(f)
                _, dataset_name, optimizer_type, robot_name, _ = file_name.split("_")
                processed_data[optimizer_type].extend(seq_data_ls)

    # show_human_statistics(processed_data)

    # check_data_correctness_separate(processed_data, working_dir)

    check_data_correctness_together(processed_data, working_dir)