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
    quaternion_to_matrix, matrix_to_quaternion, axis_angle_to_quaternion, quaternion_to_axis_angle
)
from pytransform3d import transformations as pt

from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig


from utils.dexycb_dataset import DexYCBVideoDataset
from .hand_robot_viewer import RobotHandDatasetSAPIENViewer
from .hand_viewer import HandDatasetSAPIENViewer

from utils.hand_model import load_robot
from utils.mano_layer import MANOLayer
from utils.tools import (pt_transform,
                        get_point_clouds_from_dexycb,
                        compute_hand_geometry)
from utils.vis_utils import vis_frames_plotly

import warnings; warnings.filterwarnings("ignore", category=UserWarning)


# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_



def viz_hand_object(data_id, robots: Optional[Tuple[RobotName]], data_root: Path, fps: int):
    dataset = DexYCBVideoDataset(data_root, hand_type="right", mode='full')
    if robots is None:
        viewer = HandDatasetSAPIENViewer(headless=True, use_ray_tracing=True)
    else:
        viewer = RobotHandDatasetSAPIENViewer(
            list(robots), HandType.right, headless=True, use_ray_tracing=True
        )

    sampled_data = dataset[data_id]
    for key, value in sampled_data.items():
        if "pose" not in key:
            print(f"{key}: {value}")
    viewer.load_object_hand(sampled_data)
    viewer.retarget_and_save(sampled_data, data_id)
    # viewer.render_dexycb_data(sampled_data, data_id, fps)


def main(data_id: int, dexycb_dir: str, robots: Optional[List[RobotName]] = None, fps: int = 10):
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

    viz_hand_object(data_id, robots, data_root, fps)



def test(data_id: int = 4, robots: Optional[List[RobotName]] = [RobotName.allegro], dexycb_dir: str = '/home/qianxu/Desktop/Project/interaction_pose/thirdparty_module/dex-retargeting/data', hand_type: str = "right", fps: int = 10):

    robot_name = robots[0]
    robot_name_str = str(robot_name).split(".")[-1]
    data_root = Path(dexycb_dir).absolute()
    dataset = DexYCBVideoDataset(data_root, hand_type=hand_type, mode="sub")

    sampled_data = dataset[data_id]

 
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

    extrinsic_mat =sampled_data["extrinsics"] 
    pose_vec = pt.pq_from_transform(extrinsic_mat)
    camera_pose = torch.eye(4)
    camera_pose[:3, :3] = quaternion_to_matrix(torch.from_numpy(pose_vec[3:7]))
    camera_pose[:3, 3] = torch.from_numpy(pose_vec[0:3])
    camera_pose = camera_pose.numpy()
    camera_pose = np.linalg.inv(camera_pose)

        
    start_frame = 0
    ### hand joints ###
    hand_joints = []
    hand_pose_frame = sampled_data["hand_pose"]

    import pickle
    with open("a_rtg.pkl", "wb") as f:
        pickle.dump(hand_pose_frame, f)

    mano_layer = MANOLayer(hand_type, np.zeros(10).astype(np.float32))
    for i in range(hand_pose_frame.shape[0]):
        joint = compute_hand_geometry(hand_pose_frame[i], mano_layer= mano_layer)
        if joint is None:
            start_frame += 1
            continue
        joint = joint @ camera_pose[:3, :3].T + camera_pose[:3, 3]
        joint = np.ascontiguousarray(joint)
        hand_joints.append(joint)
    hand_joints = torch.tensor(hand_joints, dtype=torch.float32)


    ### point clouds ###
    pc = get_point_clouds_from_dexycb(sampled_data)
    object_pose = sampled_data["object_pose"] # T x 7, 7=q+t, q=xyzw, t=xyz
    pc_ls = []
    for i in range(start_frame, object_pose.shape[0]):
        pose = torch.from_numpy(object_pose[i])
        transformation = torch.eye(4)
        transformation[:3, :3] = quaternion_to_matrix(torch.cat((pose[3:4], pose[:3])))
        transformation[:3, 3] = pose[4:]
        transformed_pc = pt_transform(pc, transformation)
        transformed_pc = transformed_pc @ camera_pose[:3, :3].T + camera_pose[:3, 3]
        transformed_pc = np.ascontiguousarray(transformed_pc)
        pc_ls.append(torch.from_numpy(transformed_pc))
    pc_ls = [torch.stack(pc_ls).type(torch.float32)]


    vis_frames_plotly(
        pc_ls=pc_ls,
        gt_hand_joints=hand_joints,
        hand_mesh=hand_meshes,
        show_axis=True,
        filename=f"/home/qianxu/Desktop/Project/DexPose/retarget/vis_results/{robot_name_str}_seq_{data_id}_qpos"
    )    


def run(data_id : int = 4, mode : str = "test", robots: Optional[List[RobotName]] = [RobotName.allegro], dexycb_dir: str = '/home/qianxu/Desktop/Project/interaction_pose/thirdparty_module/dex-retargeting/data', hand_type: str = "right", fps: int = 10):
    if mode == "test":
        test(data_id, robots, dexycb_dir, hand_type, fps)
    elif mode == "main":
        main(data_id, dexycb_dir, robots, fps)
    elif mode == "all":
        main(data_id, dexycb_dir, robots, fps)
        import time; time.sleep(1)
        test(data_id, robots, dexycb_dir, hand_type, fps)
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'test' and 'main'.")


def test_multiple():
    # Example usage of the test function with multiple data IDs
    data_ids = [1]
    robots = [RobotName.inspire, ]
            #   RobotName.svh, 
            #   RobotName.leap, 
            #   RobotName.allegro, 
            #   RobotName.shadow, 
            #   RobotName.panda]  # List of robots to test

    for data_id in data_ids:
        for robot in robots:
            print(f"Testing with data_id: {data_id}, robot: {robot}")
            main(data_id=data_id, dexycb_dir='/home/qianxu/Desktop/Project/interaction_pose/thirdparty_module/dex-retargeting/data', robots=[robot], fps=10)
            import time; time.sleep(1)
            test(data_id=data_id, robots=[robot], dexycb_dir='/home/qianxu/Desktop/Project/interaction_pose/thirdparty_module/dex-retargeting/data', hand_type="right", fps=10)


if __name__ == "__main__":
    # tyro.cli(main)
    # tyro.cli(test)
    # tyro.cli(run, description="Run the hand-object visualization for DexYCB dataset.")
    test_multiple()