from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import cv2
from tqdm import trange
import numpy as np
import sapien
import torch
import pytorch_kinematics as pk
from pytransform3d import rotations
from pytorch3d.transforms import quaternion_to_axis_angle, matrix_to_quaternion
from sapien import internal_renderer as R
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
from manopth.manolayer import ManoLayer

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

from dataset.base_structure import HumanSequenceData, DexSequenceData
from utils.tools import extract_hand_points_and_mesh


def compute_smooth_shading_normal_np(vertices, indices):
    """
    Compute the vertex normal from vertices and triangles with numpy
    Args:
        vertices: (n, 3) to represent vertices position
        indices: (m, 3) to represent the triangles, should be in counter-clockwise order to compute normal outwards
    Returns:
        (n, 3) vertex normal

    References:
        https://www.iquilezles.org/www/articles/normals/normals.htm
    """
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)  # (n, 3) normal without normalization to 1

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal


class HandDatasetSAPIENViewer:
    def __init__(self, headless=False, use_ray_tracing=False):
        # Setup
        if not use_ray_tracing:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        else:
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(64)
            sapien.render.set_ray_tracing_path_depth(8)
            sapien.render.set_ray_tracing_denoiser("oidn")

        # Scene
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

        # Viewer
        if not headless:
            viewer = Viewer()
            viewer.set_scene(scene)
            viewer.set_camera_xyz(1.5, 0, 1)
            viewer.set_camera_rpy(0, -0.8, 3.14)
            viewer.control_window.toggle_origin_frame(False)
            self.viewer = viewer
        else:
            self.camera = scene.add_camera("cam", 1920, 640, 0.9, 0.01, 100)
            self.camera.set_local_pose(
                sapien.Pose([1.5, 0, 1], [0, 0.389418, 0, -0.921061])
            )

        self.headless = headless

        # Create table
        white_diffuse = sapien.render.RenderMaterial()
        white_diffuse.set_base_color(np.array([0.8, 0.8, 0.8, 1]))
        white_diffuse.set_roughness(0.9)
        builder = scene.create_actor_builder()
        builder.add_box_collision(
            sapien.Pose([0, 0, -0.02]), half_size=np.array([0.5, 2.0, 0.02])
        )
        builder.add_box_visual(
            sapien.Pose([0, 0, -0.02]),
            half_size=np.array([0.5, 2.0, 0.02]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([0.4, 1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([-0.4, 1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([0.4, -1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([-0.4, -1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        self.table = builder.build_static(name="table")
        self.table.set_pose(sapien.Pose([0.5, 0, 0]))

        # Caches
        sapien.render.set_log_level("error")
        self.scene = scene
        self.internal_scene: R.Scene = scene.render_system._internal_scene
        self.context: R.Context = sapien.render.SapienRenderer()._internal_context
        self.mat_hand = self.context.create_material(
            np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0
        )

        self.mano_face: Optional[np.ndarray] = None
        self.objects: List[sapien.Entity] = []
        self.nodes: List[R.Node] = []

    def clear_all(self):
        for actor in self.objects:
            self.scene.remove_actor(actor)
        for _ in range(len(self.objects)):
            actor = self.objects.pop()
            self.scene.remove_actor(actor)
        self.clear_node()

    def clear_node(self):
        for _ in range(len(self.nodes)):
            node = self.nodes.pop()
            self.internal_scene.remove_node(node)

    def load_object_hand_dex(self, data: Union[HumanSequenceData, DexSequenceData]):
        data_ids = data["object_names"]
        mesh_files = data["object_mesh_path"]
        for data_id, mesh_file in zip(data_ids, mesh_files):
            self._load_object(data_id, mesh_file)

        self.mano_face = ManoLayer(
            flat_hand_mean=False,
            ncomps=45,
            side="left" if data["side"] == 0 else "right",
            use_pca=True,
        ).th_faces.cpu().numpy()

    def _load_object(self, name, mesh_file):
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(mesh_file)
        actor = builder.build_static(name=name)
        self.objects.append(actor)


    def _update_hand(self, vertex):
        self.clear_node()
        normal = compute_smooth_shading_normal_np(vertex, self.mano_face)
        mesh = self.context.create_mesh_from_array(vertex, self.mano_face, normal)
        model = self.context.create_model([mesh], [self.mat_hand])
        node = self.internal_scene.add_node()
        node.set_position(np.array([0, 0, 0]))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        self.nodes.append(node)



class RobotHandDatasetSAPIENViewer(HandDatasetSAPIENViewer):
    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
        optimizer_type='position',
        headless=False,
        use_ray_tracing=False,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.retarget2pk: List[np.ndarray] = []
        self.hand_type = hand_type
        self.optimizer_type = optimizer_type

        # Load optimizer and filter
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position_pc, hand_type
            )

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            override = dict(add_dummy_free_joint=True, type=optimizer_type)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # Build robot
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )
            urdf_name = urdf_path.name
            temp_path = 'retarget/urdf/' + urdf_name
            if not os.path.exists('retarget/urdf'):
                os.makedirs('retarget/urdf')
                robot_urdf.write_xml_file(temp_path)
            print("Viewer:", temp_path)

            robot = loader.load(temp_path)
            self.robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)

            self.chain = pk.build_chain_from_urdf(open(temp_path, "rb").read()).to(dtype=torch.float)

            retarget2pk = np.array(
                [retargeting.joint_names.index(n) for n in self.chain.get_joint_parameter_names()]
            )
            self.retarget2sapien.append(retarget2sapien)
            self.retarget2pk.append(retarget2pk)


    def load_object_hand_dex(self, data: Union[HumanSequenceData, DexSequenceData]):
        super().load_object_hand_dex(data)
        data_ids = data["object_names"]
        mesh_files = data["object_mesh_path"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        for _ in range(len(self.robots)):
            for data_id, mesh_file in zip(data_ids, mesh_files):
                self._load_object(data_id, mesh_file)

    def render_data_dex(self, data: DexSequenceData, data_id, fps=5, y_offset=0.8):
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))
            self.camera.set_local_pose(local_pose)

        hand_pose = torch.cat([quaternion_to_axis_angle(data.hand_coeffs).reshape(-1, 48), data.hand_tsls], dim=-1)
        hand_pose = hand_pose.unsqueeze(1).numpy()  # Convert to numpy array

        # from 4 x 4 matrix to 7D vector
        object_pose = data.obj_poses  # N x T x 4 x 4
        object_tsl = object_pose[..., :3, 3]  # N x T x 3
        object_quat = matrix_to_quaternion(object_pose[..., :3, :3])
        # from xyzw to wxyz
        object_quat = torch.cat([object_quat[:, :, 3:], object_quat[..., :3]], axis=-1)
        object_pose = torch.cat([object_quat, object_tsl], axis=-1).numpy()  # N x T x 7

        num_frame = hand_pose.shape[0]
        num_copy = len(self.robots) + 1
        num_ycb_objects = len(data.object_names)
        pose_offsets = []

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]
            robot_names = "_".join(robot_names)
            video_path = (
                Path(__file__).parent.resolve() / f"data/{robot_names}_{data_id}_video.mp4"
            )
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (self.camera.get_width(), self.camera.get_height()),
            )

        # Warm start
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(
            hand_pose_start[0, 0:3]
        )
        vertex, joint = self._compute_hand_geometry(hand_pose_start)
        
        for robot, retargeting, retarget2sapien in zip(
            self.robots, self.retargetings, self.retarget2sapien
        ):
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )

        # Loop rendering
        step_per_frame = int(60 / fps)
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[:, i, ...]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            # Update poses for YCB objects
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]

                # Quaternion convention: xyzw -> wxyz
                pose = sapien.Pose(
                    pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]])
                )
                self.objects[k].set_pose(pose)
                for copy_ind in range(num_copy):
                    self.objects[k + copy_ind * num_ycb_objects].set_pose(
                        pose_offsets[copy_ind] * pose
                    )

            # Update pose for human hand
            self._update_hand(vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(
                self.robots, self.retargetings, self.retarget2sapien
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)

            self.scene.update_render()
            if self.headless:
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                writer.write(rgb[..., ::-1])
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            writer.release()

    def retarget_and_save_dex(self, data: HumanSequenceData, data_id, fps=5, y_offset=0.8):
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))
            self.camera.set_local_pose(local_pose)

        # from 4 x 4 matrix to 7D vector
        object_pose = data["obj_poses"]  # N x T x 4 x 4
        object_tsl = object_pose[..., :3, 3]  # N x T x 3
        object_quat = matrix_to_quaternion(object_pose[..., :3, :3])
        # from xyzw to wxyz
        object_quat = torch.cat([object_quat[:, :, 3:], object_quat[..., :3]], axis=-1)
        object_pose = torch.cat([object_quat, object_tsl], axis=-1).numpy()  # N x T x 7

        num_frame = object_pose.shape[1]
        num_copy = len(self.robots) + 1
        num_ycb_objects = len(data["object_names"])
        pose_offsets = []

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)

        # Prepare to save qpos
        qpos_dict = {robot_name: [] for robot_name in self.robot_names}

        start_frame = 0
        # Warm start
        joints, vertex = extract_hand_points_and_mesh(data["hand_tsls"][start_frame], data["hand_coeffs"][start_frame], data["side"])
        joints = joints.squeeze(0)
        wrist_quat = data["hand_coeffs"][start_frame, 0]

        for robot, retargeting, retarget2sapien in zip(
            self.robots, self.retargetings, self.retarget2sapien
        ):
            retargeting.warm_start(
                joints[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )

        from utils.tools import get_point_clouds_from_human_data, apply_transformation_human_data
        pc, pc_norm = get_point_clouds_from_human_data(data, return_norm=True)
        pc_ls, pc_norm_ls = apply_transformation_human_data(pc, data["obj_poses"], norm=pc_norm)

        # Loop rendering
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[:, i, ...]
            joints, vertex = extract_hand_points_and_mesh(data["hand_tsls"][i], data["hand_coeffs"][i], data["side"])
            joints = joints.squeeze(0)
            vertex = vertex.squeeze(0)

            # print("Frame:", i, "Joints:", joints)

            # Update poses for YCB objects
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]

                # Quaternion convention: xyzw -> wxyz
                pose = sapien.Pose(
                    pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]])
                )
                self.objects[k].set_pose(pose)
                for copy_ind in range(num_copy):
                    self.objects[k + copy_ind * num_ycb_objects].set_pose(
                    pose_offsets[copy_ind] * pose
                    )

            # Update pose for human hand
            self._update_hand(vertex)

            # Update poses for robot hands and save qpos
            for robot, retargeting, retarget2pk, retarget2sapien, robot_name in zip(
            self.robots, self.retargetings, self.retarget2pk, self.retarget2sapien, self.robot_names
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joints[indices, :]
                # print("Frame:", i, "Ref value:", ref_value)
                retargeting.set_pc(pc_ls[i], pc_norm_ls[i])
                qpos_retarget = retargeting.retarget(ref_value)
                qpos_sapien = qpos_retarget[retarget2sapien]
                qpos_pk = qpos_retarget[retarget2pk]
                # print("Frame:", i, "Qpos Sapien:", qpos_sapien)
                # print("Frame:", i, "Qpos PK:", qpos_pk)
                robot.set_qpos(qpos_sapien)
                qpos_dict[robot_name].append(qpos_pk.copy())

                # if i == 30:
                #     from .vis_utils import vis_pc_coor_plotly
                #     vis_pc_coor_plotly(pc_ls=[pc_ls[i]], 
                #                        posi_pts_ls=[retargeting.optimizer.robot.get_penetration_keypoints().detach().cpu().numpy()],
                #                        hand_mesh=retargeting.optimizer.robot.get_trimesh_data(), 
                #                        obj_norm_ls=[pc_norm_ls[i]],
                #                        filename="xxx_0")
                    
                #     import sys; sys.exit(0)

        # Save qpos to disk as npy files
        # save_dir = Path('retarget/hand_qpos')
        # save_dir.mkdir(parents=True, exist_ok=True)
        retargeted_data = []
        for robot_name, qpos_list in qpos_dict.items():
            qpos_arr = torch.from_numpy(np.stack(qpos_list, axis=0).astype(np.float32))
            robot_name_str= str(robot_name).split('.')[-1]+'_hand'
            # np.save(save_dir / f"{str(robot_name).split('.')[-1]}_{data_id}.npy", qpos_arr)
            retargeted_data.append(dict(
                which_hand=robot_name_str,
                hand_poses=qpos_arr,
                side=data["side"],

                hand_tsls=data["hand_tsls"],
                hand_coeffs=data["hand_coeffs"],
                # hand_joints=data["hand_joints"],

                obj_poses=data["obj_poses"],
                obj_point_clouds=None,
                obj_feature=None,
                object_names=data["object_names"],
                object_mesh_path=data["object_mesh_path"],
                # mesh_norm_trans=data["mesh_norm_trans"],

                frame_indices=data["frame_indices"],
                task_description=data["task_description"],
                which_dataset=data["which_dataset"],
                which_sequence=data["which_sequence"],
                extra_info=data["extra_info"]
            ))

            # print("QPOS:")
            # for i, q in enumerate(qpos_arr):
            #     print(f"Frame {i}: {q}")

        return retargeted_data