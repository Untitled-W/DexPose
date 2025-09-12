import json
import os
from typing import List

import numpy as np
import numpy.typing as npt
import open3d as o3d
import pinocchio as pin
import pytorch_kinematics as pk
import pytorch3d.ops
import pytorch3d.structures
from torchsdf import index_vertices_by_faces, compute_sdf
import torch
import trimesh

# ==============================================================================
# 辅助函数 (这些函数可能在您的原始代码库中)
# 您需要提供这些函数的实现
# ==============================================================================



# ==============================================================================
# 融合后的主类
# ==============================================================================

class HandRobotWrapper:
    """
    一个融合了 Pinocchio 的机器人学查询功能和 PyTorch Kinematics/3D 
    的批处理、基于GPU的网格/动力学功能的机器人封装类。
    """

    def __init__(self, urdf_path: str, mesh_path: str, device=None,
                 n_surface_points=2000,
                 penetration_points_path='mjcf/penetration_points.json',
                 contact_points_path='mjcf/contact_points.json',
                 fingertip_points_path='mjcf/fingertip.json',
                 tip_aug=None):
        """
        初始化函数
        
        Parameters
        ----------
        urdf_path: str
            URDF文件的路径
        mesh_path: str
            包含手部网格文件（如.obj）的目录路径
        device: str | torch.Device
            PyTorch 张量所使用的设备
        n_surface_points: int
            从手部表面采样的总点数
        penetration_points_path, contact_points_path, fingertip_points_path: str
            指向预定义关键点JSON文件的路径
        tip_aug: float, optional
            增加指尖区域采样权重的增强因子
        """
        # ---------------------------------------------------------------------- #
        # 1. Pinocchio 初始化 (来自 RobotWrapper)
        # ---------------------------------------------------------------------- #
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()
        self.q0 = pin.neutral(self.model)
        if self.model.nv != self.model.nq:
            raise NotImplementedError("无法处理带有特殊关节的机器人。")

        # ---------------------------------------------------------------------- #
        # 2. PyTorch Kinematics 和设备设置 (来自 HandModelMJCF)
        # ---------------------------------------------------------------------- #
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read()).to(dtype=torch.float, device=self.device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        # ---------------------------------------------------------------------- #
        # 3. 加载接触点/关键点 JSON 文件 (来自 HandModelMJCF)
        # ---------------------------------------------------------------------- #
        contact_points = json.load(open(contact_points_path, 'r')) if contact_points_path is not None else None
        penetration_points = json.load(open(penetration_points_path, 'r')) if penetration_points_path is not None else None
        fingertip_points = json.load(open(fingertip_points_path, "r")) if fingertip_points_path is not None else None

        # ---------------------------------------------------------------------- #
        # 4. 递归构建网格和处理关键点 (来自 HandModelMJCF)
        # ---------------------------------------------------------------------- #
        self.mesh = {}
        areas = {}

        def build_mesh_recurse(body):
            vvisuals = [v for v in body.link.visuals if v.geom_param is not None]
            if len(vvisuals) > 0:
                link_name = body.link.name
                link_vertices, link_faces = [], []
                n_link_vertices = 0

                for visual in vvisuals:
                    if visual.geom_type is None: continue
                    
                    if visual.geom_type == "box":
                        scale = torch.tensor(visual.geom_param, dtype=torch.float, device=self.device)
                        link_mesh = o3d.geometry.TriangleMesh.create_box(
                            width=scale[0].item(), height=scale[1].item(), depth=scale[2].item())
                    elif visual.geom_type == "capsule":
                        radius, half_height = visual.geom_param[0], visual.geom_param[1]
                        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=half_height * 2)
                        cylinder.translate((0, 0, -half_height))
                        sphere_top = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                        sphere_top.translate((0, 0, half_height))
                        sphere_bottom = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                        sphere_bottom.translate((0, 0, -half_height))
                        link_mesh = cylinder + sphere_top + sphere_bottom
                    else: # "mesh"
                        link_mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_path, visual.geom_param[0]))
                        if len(visual.geom_param) > 1 and visual.geom_param[1] is not None:
                            scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=self.device)
                            vertices = np.asarray(link_mesh.vertices) * scale.cpu().numpy()
                            link_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    
                    vertices = torch.from_numpy(np.asarray(link_mesh.vertices)).to(dtype=torch.float, device=self.device)
                    faces = torch.from_numpy(np.asarray(link_mesh.triangles)).to(dtype=torch.long, device=self.device)
                    
                    pos = visual.offset.to(dtype=torch.float, device=self.device)
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)

                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                
                # 为没有在JSON中定义的link创建空列表
                if link_name not in contact_points: contact_points[link_name] = []
                if link_name not in penetration_points: penetration_points[link_name] = []
                if link_name not in fingertip_points: fingertip_points[link_name] = []

                self.mesh[link_name] = {
                    'vertices': link_vertices,
                    'faces': link_faces,
                    'contact_candidates': torch.tensor(contact_points[link_name], dtype=torch.float32, device=self.device).reshape(-1, 3),
                    'penetration_keypoints': torch.tensor(penetration_points[link_name], dtype=torch.float32, device=self.device).reshape(-1, 3),
                    'fingertip_keypoints': torch.tensor(fingertip_points[link_name], dtype=torch.float32, device=self.device).reshape(-1, 3),
                }

                self.mesh[link_name]['face_verts'] = index_vertices_by_faces(link_vertices, link_faces)
                areas[link_name] = trimesh.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()

            for children in body.children:
                build_mesh_recurse(children)
        
        build_mesh_recurse(self.chain._root)
        
        # ---------------------------------------------------------------------- #
        # 5. 设置关节范围 (来自 HandModelMJCF)
        # ---------------------------------------------------------------------- #
        self.joints_names, self.joints_lower, self.joints_upper = [], [], []
        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(body.joint.range[0] if body.joint.range is not None else -np.pi)
                self.joints_upper.append(body.joint.range[1] if body.joint.range is not None else np.pi)
            for children in body.children:
                set_joint_range_recurse(children)
        
        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.tensor(self.joints_lower, dtype=torch.float, device=self.device)
        self.joints_upper = torch.tensor(self.joints_upper, dtype=torch.float, device=self.device)
        
        # ---------------------------------------------------------------------- #
        # 6. 采样表面点 (来自 HandModelMJCF)
        # ---------------------------------------------------------------------- #
        if tip_aug:
            fingertip_link_names = [k for k in areas.keys() if 'distal' in k]
            middle_link_names = [k for k in areas.keys() if 'middle' in k]
            for name in fingertip_link_names: areas[name] *= tip_aug
            for name in middle_link_names: areas[name] *= tip_aug * 0.8
        
        total_area = sum(areas.values())
        if total_area > 0:
            num_samples = {k: int(v / total_area * n_surface_points) for k, v in areas.items()}
            num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values())
            for link_name in self.mesh:
                if num_samples[link_name] == 0:
                    self.mesh[link_name]['surface_points'] = torch.empty((0, 3), dtype=torch.float, device=self.device)
                    continue
                mesh = pytorch3d.structures.Meshes(self.mesh[link_name]['vertices'].unsqueeze(0), self.mesh[link_name]['faces'].unsqueeze(0))
                dense_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
                surface_points = pytorch3d.ops.sample_farthest_points(dense_cloud, K=num_samples[link_name])[0][0]
                self.mesh[link_name]['surface_points'] = surface_points.to(dtype=torch.float, device=self.device)

        # ---------------------------------------------------------------------- #
        # 7. 全局化接触点和关键点 (来自 HandModelMJCF)
        # ---------------------------------------------------------------------- #
        self.link_name_to_link_index = {name: i for i, name in enumerate(self.mesh.keys())}

        contact_candidates_list = [self.mesh[link_name]['contact_candidates'] for link_name in self.mesh]
        self.global_index_to_link_index = torch.tensor(
            sum([[i] * len(p) for i, p in enumerate(contact_candidates_list)], []), dtype=torch.long, device=self.device)
        self.contact_candidates = torch.cat(contact_candidates_list, dim=0)
        self.n_contact_candidates = self.contact_candidates.shape[0]

        penetration_keypoints_list = [self.mesh[link_name]['penetration_keypoints'] for link_name in self.mesh]
        self.global_index_to_link_index_penetration = torch.tensor(
            sum([[i] * len(p) for i, p in enumerate(penetration_keypoints_list)], []), dtype=torch.long, device=self.device)
        self.penetration_keypoints = torch.cat(penetration_keypoints_list, dim=0)
        self.n_keypoints = self.penetration_keypoints.shape[0]

        # ---------------------------------------------------------------------- #
        # 8. 初始化状态变量 (来自 HandModelMJCF)
        # ---------------------------------------------------------------------- #
        self.hand_pose = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None
        self.contact_point_indices = None

    # -------------------------------------------------------------------------- #
    # 属性 (来自 RobotWrapper)
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names)[1:] # 忽略 universe

    @property
    def dof_joint_names(self) -> List[str]:
        # pinocchio的'names'包含了所有joint, 包括fixed.
        # self.joints_names (来自set_joint_range_recurse)更可靠
        return self.joints_names

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        return [frame.name for frame in self.model.frames]

    @property
    def joint_limits(self):
        # 使用从 pytorch_kinematics 派生的范围，因为它们更可靠
        return torch.stack([self.joints_lower, self.joints_upper], dim=1)

    # -------------------------------------------------------------------------- #
    # 查询函数 (来自 RobotWrapper)
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        if not self.model.hasFrame(name):
            raise ValueError(f"{name} 不是一个有效的 link/frame 名称。")
        return self.model.getFrameId(name)

    # -------------------------------------------------------------------------- #
    # 基于 Pinocchio 的动力学函数 (来自 RobotWrapper)
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics_pinocchio(self, qpos: npt.NDArray, contact_point_indices=None):
        """基于Pinocchio计算单个qpos的前向动力学 (CPU)"""
        pin.forwardKinematics(self.model, self.data, qpos)
        pin.updateFramePlacements(self.model, self.data)
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device)
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, n_contact, 4, 4)
                transforms[mask] = cur[mask]
            self.contact_points = torch.cat([self.contact_points, torch.ones(batch_size, n_contact, 1, dtype=torch.float, device=self.device)], dim=2)
            self.contact_points = (transforms @ self.contact_points.unsqueeze(3))[:, :, :3, 0]

    def get_link_pose_pinocchio(self, link_name: str) -> npt.NDArray:
        """基于Pinocchio获取单个link的位姿 (CPU)"""
        link_id = self.get_link_index(link_name)
        pose: pin.SE3 = self.data.oMf[link_id]
        return pose.homogeneous

    def compute_single_link_local_jacobian_pinocchio(self, qpos: npt.NDArray, link_name: str) -> npt.NDArray:
        """基于Pinocchio计算单个link的雅可比矩阵 (CPU)"""
        link_id = self.get_link_index(link_name)
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J
    

    # -------------------------------------------------------------------------- #
    #  基于 PyTorch Kinematics/3D 的工具函数 (来自 HandModelMJCF)
    # -------------------------------------------------------------------------- #
    def get_trimesh_data(self, i):
        """
        Get full mesh
        
        Returns
        -------
        data: trimesh.Trimesh
        """
        data = trimesh.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            data += trimesh.Trimesh(vertices=v, faces=f)
        return data
    
    def get_surface_points(self):
        """
        Get surface points
        
        Returns
        -------
        points: (B, `n_surface_points`, 3)
            surface points
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['surface_points'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['surface_points']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_tip_points(self):
        """
        Get tip points
        """
        # points = []
        batch_size = self.global_translation.shape[0]
        # for link_name in self.mesh:
        #     n_surface_points = self.mesh[link_name]['fingertip_keypoints'].shape[0]
        #     points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['fingertip_keypoints']))
        # points = torch.cat(points, dim=-2).to(self.device)
        # points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)

        points = [self.current_status[link_name].transform_points(self.mesh[link_name]['fingertip_keypoints']).expand(batch_size, -1, -1) for link_name in self.mesh]
        points = torch.concat(points, dim=1) @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points
    
    def get_intersect(self, M)->bool:
        """
        Get intersection between the hand and the object
        
        Parameters
        ----------
        points: (N, 3)
            surface points
        
        Returns
        -------
        intersect: (bool )
            whether the hand intersects with the object
        """
        num_ls = []
        for idx in range(M):
            hand_mesh = self.get_trimesh_data(idx)
            points = self.ref_points
            oritation = np.array([0, 1, 1.]).repeat(points.shape[0], axis=0).reshape((-1, 3))
            tt = trimesh.ray.ray_pyembree.RayMeshIntersector(hand_mesh)
            _, index_ray0 = tt.intersects_id(ray_origins=points, ray_directions=oritation, multiple_hits=True, return_locations=False)
            _, index_ray1 = tt.intersects_id(ray_origins=points, ray_directions=-oritation)
            count0 = np.bincount(index_ray0)
            count1 = np.bincount(index_ray1)
            bool_0 = count0 % 2 != 0
            bool_1 = count1 % 2 != 0
            bool = np.logical_or(bool_0, bool_1)
            num = np.count_nonzero(bool)
            num_ls.append(num)
        num_ls = torch.tensor(num_ls).to(self.device)
        return num_ls
    
    def cal_distance(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes
        
        Interiors are positive, exteriors are negative
        
        Use analytical method and our modified Kaolin package
        
        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """
        # Consider each link seperately: 
        #   First, transform x into each link's local reference frame using inversed fk, which gives us x_local
        #   Next, calculate point-to-mesh distances in each link's frame, this gives dis_local
        #   Finally, the maximum over all links is the final distance from one point to the entire ariticulation
        # In particular, the collision mesh of ShadowHand is only composed of Capsules and Boxes
        # We use analytical method to calculate Capsule sdf, and use our modified Kaolin package for other meshes
        # This practice speeds up the reverse penetration calculation
        # Note that we use a chamfer box instead of a primitive box to get more accurate signs
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        
        for link_name in self.mesh:
            if link_name in ['forearm', 'wrist', 'ffknuckle', 'mfknuckle', 'rfknuckle', 'lfknuckle', 'thbase', 'thhub']:
                continue
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            # if 'geom_param' not in self.mesh[link_name]:
            face_verts = self.mesh[link_name]['face_verts']
            dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
            dis_local = torch.sqrt(dis_local + 1e-8)
            dis_local = dis_local * (-dis_signs)
            # else:
            #     print(self.mesh[link_name]['geom_param'])
            #     height = self.mesh[link_name]['geom_param'][1] * 2
            #     radius = self.mesh[link_name]['geom_param'][0]
            #     nearest_point = x_local.detach().clone()
            #     nearest_point[:, :2] = 0
            #     nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], 0, height)
            #     dis_local = radius - (x_local - nearest_point).norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis
        # raise NotImplementedError
    
    def self_penetration(self):
        """
        Calculate self penetration energy
        
        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        batch_size = self.global_translation.shape[0]
        points = self.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        link_indices = self.global_index_to_link_index_penetration.clone().repeat(batch_size,1)
        transforms = torch.zeros(batch_size, self.n_keypoints, 4, 4, dtype=torch.float, device=self.device)
        for link_name in self.mesh:
            mask = link_indices == self.link_name_to_link_index[link_name]
            cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, self.n_keypoints, 4, 4)
            transforms[mask] = cur[mask]
        points = torch.cat([points, torch.ones(batch_size, self.n_keypoints, 1, dtype=torch.float, device=self.device)], dim=2)
        points = (transforms @ points.unsqueeze(3))[:, :, :3, 0]
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        dis = 0.02 - dis
        E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
        return E_spen.sum((1,2))
    
    def get_E_joints(self):
        """
        Calculate joint energy
        
        Returns
        -------
        E_joints: (N,) torch.Tensor
            joint energy
        """
        E_joints = E_joints = torch.sum((self.hand_pose[:, 6:] > self.joints_upper[6:]) * (self.hand_pose[:, 6:] - self.joints_upper[6:]), dim=-1) + \
        torch.sum((self.hand_pose[:, 6:] < self.joints_lower[6:]) * (self.joints_lower[6:] - self.hand_pose[:, 6:]), dim=-1)
        return E_joints
    
    def get_penetration_keypoints(self):
        """
        Get penetration keypoints
        
        Returns
        -------
        points: (N, `n_keypoints`, 3) torch.Tensor
            penetration keypoints
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['penetration_keypoints'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['penetration_keypoints']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_contact_candidates(self):
        """
        Get all contact candidates
        
        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['contact_candidates'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['contact_candidates']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        # torch.cat([item[1]['contact_candidates'] for item in self.mesh.items() if 'th' in item[0]], dim=0)
        return points
    
    
    def cal_distance(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes
        
        Interiors are positive, exteriors are negative
        
        Use analytical method and our modified Kaolin package
        
        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """
        # Consider each link seperately: 
        #   First, transform x into each link's local reference frame using inversed fk, which gives us x_local
        #   Next, calculate point-to-mesh distances in each link's frame, this gives dis_local
        #   Finally, the maximum over all links is the final distance from one point to the entire ariticulation
        # In particular, the collision mesh of ShadowHand is only composed of Capsules and Boxes
        # We use analytical method to calculate Capsule sdf, and use our modified Kaolin package for other meshes
        # This practice speeds up the reverse penetration calculation
        # Note that we use a chamfer box instead of a primitive box to get more accurate signs
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        
        for link_name in self.mesh:
            if link_name in ['robot0:forearm', 'robot0:wrist_child', 'robot0:ffknuckle_child', 
                             'robot0:mfknuckle_child', 'robot0:rfknuckle_child', 'robot0:lfknuckle_child', 
                             'robot0:thbase_child', 'robot0:thhub_child']:
                continue
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            # if 'geom_param' not in self.mesh[link_name]:
            face_verts = self.mesh[link_name]['face_verts']
            dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
            dis_local = torch.sqrt(dis_local + 1e-8)
            dis_local = dis_local * (-dis_signs)
            # else:
            #     print(self.mesh[link_name]['geom_param'])
            #     height = self.mesh[link_name]['geom_param'][1] * 2
            #     radius = self.mesh[link_name]['geom_param'][0]
            #     nearest_point = x_local.detach().clone()
            #     nearest_point[:, :2] = 0
            #     nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], 0, height)
            #     dis_local = radius - (x_local - nearest_point).norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis
        # raise NotImplementedError
    
    def self_penetration(self):
        """
        Calculate self penetration energy
        
        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        batch_size = self.global_translation.shape[0]
        points = self.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        link_indices = self.global_index_to_link_index_penetration.clone().repeat(batch_size,1)
        transforms = torch.zeros(batch_size, self.n_keypoints, 4, 4, dtype=torch.float, device=self.device)
        for link_name in self.mesh:
            mask = link_indices == self.link_name_to_link_index[link_name]
            cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, self.n_keypoints, 4, 4)
            transforms[mask] = cur[mask]
        points = torch.cat([points, torch.ones(batch_size, self.n_keypoints, 1, dtype=torch.float, device=self.device)], dim=2)
        points = (transforms @ points.unsqueeze(3))[:, :, :3, 0]
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        dis = 0.02 - dis
        E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
        return E_spen.sum((1,2))
    
    def get_E_joints(self):
        """
        Calculate joint energy
        
        Returns
        -------
        E_joints: (N,) torch.Tensor
            joint energy
        """
        E_joints = E_joints = torch.sum((self.hand_pose[:, 6:] > self.joints_upper[6:]) * (self.hand_pose[:, 6:] - self.joints_upper[6:]), dim=-1) + \
        torch.sum((self.hand_pose[:, 6:] < self.joints_lower[6:]) * (self.joints_lower[6:] - self.hand_pose[:, 6:]), dim=-1)
        return E_joints
    
    def get_penetration_keypoints(self):
        """
        Get penetration keypoints
        
        Returns
        -------
        points: (N, `n_keypoints`, 3) torch.Tensor
            penetration keypoints
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['penetration_keypoints'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['penetration_keypoints']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_contact_candidates(self):
        """
        Get all contact candidates
        
        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['contact_candidates'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['contact_candidates']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        # torch.cat([item[1]['contact_candidates'] for item in self.mesh.items() if 'th' in item[0]], dim=0)
        return points
    