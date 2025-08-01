import torch
import numpy as np
import pickle
import pytorch_kinematics as pk
from pathlib import Path
from typing import Optional, Tuple, List
from pytransform3d import transformations as pt
from pytorch3d.transforms import (
    quaternion_to_matrix, 
    matrix_to_quaternion, 
    axis_angle_to_quaternion, 
    quaternion_to_axis_angle,
    matrix_to_axis_angle
)
from pytransform3d import rotations

from utils.hand_model import load_robot
from utils.mano_layer import MANOLayer
from utils.tools import (pt_transform,
                        get_point_clouds_from_dexycb,
                        compute_hand_geometry)
from utils.vis_utils import vis_frames_plotly
from utils.dexycb_dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType

dexycb_dir = '/home/qianxu/Desktop/Project/interaction_pose/data/DexYCB/dex-ycb-20210415'
hand_type: str = "right"

def test(qpos_cp, data_id, filename, robots: Optional[List[RobotName]] = [RobotName.allegro],  fps: int = 10):

    robot_name = robots[0]
    robot_name_str = str(robot_name).split(".")[-1]
    data_root = Path(dexycb_dir).absolute()
    dataset = DexYCBVideoDataset(data_root, hand_type=hand_type, mode="sub")

    sampled_data = dataset[data_id]
    print(data_id, sampled_data["capture_name"])

    robot = load_robot(robot_name_str, hand_type)



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
    # hand_pose_frame[11:,0, :3] = qpos[:, 3:6]

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


    print(f"raw data: {sampled_data['hand_pose'][start_frame, 0]}")
    print(f"qpos_cp: {qpos_cp[0]}")
    # print(f"qpos: {qpos[0]}")

    def get_robot_error():
        import sapien
        from sapien.asset import create_dome_envmap
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(64)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")
        scene = sapien.Scene()
        scene.set_timestep(1 / 240)

        # Lighting
        scene.set_environment_map(
            create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
        )
        scene.add_directional_light(
            np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True
        )
        scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
        scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))

        # Add ground
        visual_material = sapien.render.RenderMaterial()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
        visual_material.set_roughness(0.7)
        visual_material.set_metallic(1)
        visual_material.set_specular(0.04)
        scene.add_ground(-1, render_material=visual_material)

        sapien.render.set_log_level("error")

        loader = scene.create_urdf_loader()
        
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        sp_robot = loader.load('/home/qianxu/Desktop/Project/DexPose/retarget_test/urdf/allegro_hand_right_glb.urdf')
        return sp_robot

    def get_robot():
        from retarget_test.hand_robot_viewer import RobotHandDatasetSAPIENViewer
        viewer = RobotHandDatasetSAPIENViewer(
            list(robots), HandType.right, headless=True, use_ray_tracing=True
        )
        return viewer.robots[0]

    # sp_robot = get_robot()
    # sp_robot.compute_forward_kinematics(qpos_cp)
    # target_link_poses = [
    #     sp_robot.get_link_pose(index) for index in [33, 23, 43, 53, 29, 19, 39, 49]
    # ]
    # print("target_link_poses:", target_link_poses)



    ### hand meshes ###
    hand_meshes = []
    for i in range(qpos_cp.shape[0]):
        robot.set_qpos(torch.from_numpy(qpos_cp[i].astype(np.float32)))
        hand_mesh = robot.get_hand_mesh()
        # print(len(hand_mesh.vertices), len(hand_mesh.triangles))
        hand_meshes.append(hand_mesh)

    # with open('log/vis_hand_mesh_check2.log', 'wb') as f:
    #     for i, hand_mesh in enumerate(hand_meshes):
    #         f.write(f"Frame {i}: {np.asarray(hand_meshes[-1].vertices)[:10]} vertices, {len(hand_mesh.triangles)} triangles\n".encode('utf-8'))




    ### point clouds ###
    pc = get_point_clouds_from_dexycb(sampled_data)
    object_pose = sampled_data["object_pose"] # T x 7, 7=q+t, q=xyzw, t=xyz
    pc_ls = []
    T = object_pose.shape[0]
    for i in range(start_frame, T):
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
        filename=filename
    )    


if __name__ == "__main__":

    data_id = 1

    robot_qpos = np.load("/home/qianxu/Desktop/Project/DexPose/retarget_test/hand_qpos/allegro_seq_1_qpos.npy")


    data_root = Path(dexycb_dir).absolute()
    dataset = DexYCBVideoDataset(data_root, hand_type=hand_type, mode="sub")
    sampled_data = dataset[data_id]
    human_wrist_data = robot_qpos.copy()
    hand_pose = sampled_data["hand_pose"]

    def operator2mano(axis):
        wrist_quat = axis_angle_to_quaternion(torch.from_numpy(axis.astype(np.float32)))
        print("quat: ", wrist_quat[0])
        if hand_type == "left":
            operator2mano = torch.tensor([
                                [0, 0, -1],
                                [1, 0, 0],
                                [0, -1, 0],
                            ], dtype=torch.float32)
        else:
            operator2mano = torch.tensor([
                                [0, 0, -1],
                                [-1, 0, 0],
                                [0, 1, 0],
                            ], dtype=torch.float32)
        target_wrist_pose = np.eye(3)
        target_wrist_pose= (
            quaternion_to_matrix(wrist_quat) @ operator2mano.T
        )
        print("operator2mano: ", operator2mano)
        print("target_wrist_pose: ", target_wrist_pose[0])
        new_axis_old = matrix_to_axis_angle(target_wrist_pose).numpy()
        new_axis = np.stack([rotations.euler_from_matrix(
            i, 0, 1, 2, extrinsic=False
        ) for i in target_wrist_pose])
        print("new_axis_old: ", new_axis_old[0])
        print("new_axis: ", new_axis[0])
        return new_axis

    human_wrist_data[:, 3:6] = operator2mano(sampled_data['hand_pose'][11:, 0, :3])



    with open('/home/qianxu/Desktop/Project/interaction_pose/data/DexYCB/dex-ycb-20210415/human_save/seq_debug_1.p', 'rb') as f:
        load_data = pickle.load(f)[data_id]
    seq_wrist_data = robot_qpos.copy()
    seq_wrist_data[:, 3:6] = operator2mano(quaternion_to_axis_angle(load_data.hand_coeffs[:,0,:]).numpy())



    test_0 = np.zeros_like(robot_qpos)
    test_0[:, 0:3] = robot_qpos.copy()[:, 0:3]
    test_x = np.zeros_like(robot_qpos)
    test_x[:, 0:3] = robot_qpos.copy()[:, 0:3]
    test_x[:, 3:6] = np.array([np.pi / 2, 0, 0], dtype=np.float32)
    test_y = np.zeros_like(robot_qpos)
    test_y[:, 0:3] = robot_qpos.copy()[:, 0:3]
    test_y[:, 3:6] = np.array([0, np.pi / 2, 0], dtype=np.float32)
    test_z = np.zeros_like(robot_qpos)
    test_z[:, 0:3] = robot_qpos.copy()[:, 0:3]
    test_z[:, 3:6] = np.array([0, 0, np.pi / 2], dtype=np.float32)



    exps = [
        # (test_0, "zero"),
        # (test_x, "test_x"),
        # (test_y, "test_y"),
        # (test_z, "test_z"),
        (robot_qpos, "robot_qpos"),
        (human_wrist_data, "human_wrist_data"),
        (seq_wrist_data, "seq_wrist_data")

    ]

    for qpos, filename in exps:
        print(f"Testing {filename}...")
        test(qpos, data_id, f"/home/qianxu/Desktop/Project/DexPose/log/vis_results/{filename}")