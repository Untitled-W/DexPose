from typing import List, Dict, Tuple, Union
import numpy as np
import numpy.typing as npt
import pinocchio as pin
import json
import os
import open3d as o3d
import pytorch_kinematics as pk
import pytorch3d.ops
import pytorch3d.structures
from torchsdf import index_vertices_by_faces, compute_sdf
from pytorch3d.ops.knn import knn_points
import torch
import trimesh
from tqdm import tqdm

class RobotWrapper:
    """
    This class does not take mimic joint into consideration
    """

    def __init__(self, urdf_path: str, use_collision=False, use_visual=False):
        # Create robot model and data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        if use_visual or use_collision:
            raise NotImplementedError

        self.q0 = pin.neutral(self.model)
        if self.model.nv != self.model.nq:
            raise NotImplementedError("Can not handle robot with special joint.")

    # -------------------------------------------------------------------------- #
    # Robot property
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        link_names = []
        for i, frame in enumerate(self.model.frames):
            link_names.append(frame.name)
        return link_names

    @property
    def joint_limits(self):
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        if name not in self.link_names:
            raise ValueError(
                f"{name} is not a link name. Valid link names: \n{self.link_names}"
            )
        return self.model.getFrameId(name, pin.BODY)

    def get_joint_parent_child_frames(self, joint_name: str):
        joint_id = self.model.getFrameId(joint_name)
        parent_id = self.model.frames[joint_id].parent
        child_id = -1
        for idx, frame in enumerate(self.model.frames):
            if frame.previousFrame == joint_id:
                child_id = idx
        if child_id == -1:
            raise ValueError(f"Can not find child link of {joint_name}")

        return parent_id, child_id

    # -------------------------------------------------------------------------- #
    # Kinematics function
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J


class HandRobotWrapper(RobotWrapper):

    def __init__(self, robot_name: str, urdf_path: str, mesh_path: str, device=None,
                 n_surface_points=2000,
                 penetration_points_path='/home/qianxu/Desktop/Project/interaction_pose/mjcf/penetration_points.json',
                 contact_points_path='/home/qianxu/Desktop/Project/interaction_pose/mjcf/contact_points.json',
                 fingertip_points_path='/home/qianxu/Desktop/Project/interaction_pose/mjcf/fingertip.json',
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
        super().__init__(urdf_path)
        self.robot_name = robot_name

        # ---------------------------------------------------------------------- #
        # 2. PyTorch Kinematics 和设备设置 (来自 HandModelMJCF)
        # ---------------------------------------------------------------------- #
        # if device is None:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # else:
        #     self.device = device
        self.device = 'cpu'

        self.chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read()).to(dtype=torch.float, device=self.device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())
        self.pin2pk = np.array([
            self.dof_joint_names.index(n) for n in self.chain.get_joint_parameter_names()
            ])
        self.current_status = None

        # ---------------------------------------------------------------------- #
        # 3. 加载接触点/关键点 JSON 文件 (来自 HandModelMJCF)
        # ---------------------------------------------------------------------- #
        # TODO Fix later. Make these paths configurable.
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
                    else:
                        link_mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_path, visual.geom_param[0]))
                        if len(visual.geom_param) > 1 and visual.geom_param[1] is not None:
                            scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=self.device)
                            vertices = np.asarray(link_mesh.vertices) * scale.cpu().numpy()
                            link_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    
                    vertices = torch.from_numpy(np.asarray(link_mesh.vertices)).to(dtype=torch.float, device=self.device)
                    faces = torch.from_numpy(np.asarray(link_mesh.triangles)).to(dtype=torch.long, device=self.device)


                    def fit_capsule_to_points(points: torch.Tensor, padding_factor: float = 1.05):
                        """
                        Fits an arbitrarily oriented capsule to a point cloud using Principal Component Analysis (PCA).

                        This function finds the axis of greatest variance in the point cloud to define the
                        capsule's orientation and length, then finds the maximum distance from that axis
                        to define the radius.

                        Args:
                            points (torch.Tensor): A tensor of points with shape (N, 3), where N is the number of vertices.
                            padding_factor (float): A small factor to slightly enlarge the capsule's radius,
                                                    ensuring it fully encloses the original mesh.

                        Returns:
                            Dict[str, torch.Tensor or float]: A dictionary containing the capsule parameters:
                                - 'start': The start point of the capsule's core segment, shape (1, 3).
                                - 'end': The end point of the capsule's core segment, shape (1, 3).
                                - 'radius': The calculated radius of the capsule (float).
                        """
                        if points.shape[0] < 3:
                            # Cannot determine a capsule from fewer than 3 points. Return a small sphere at the center.
                            center = torch.mean(points, dim=0, keepdim=True) if points.shape[0] > 0 else torch.zeros(1, 3, device=points.device)
                            return {'start': center, 'end': center, 'radius': 0.01}

                        # 1. Center the points by subtracting their mean
                        mean = torch.mean(points, dim=0)
                        centered_points = points - mean

                        # 2. Perform Singular Value Decomposition (SVD) to find principal components
                        # V will contain the principal axes (eigenvectors) of the point cloud's covariance matrix.
                        # Note: torch.svd is deprecated, using torch.linalg.svd is recommended.
                        U, S, Vh = torch.linalg.svd(centered_points)
                        V = Vh.T
                        
                        # The first principal component is the direction of greatest variance (the capsule's main axis)
                        direction = V[:, 0]

                        # 3. Project all points onto this principal axis to find the extent of the capsule
                        projections = torch.matmul(centered_points, direction)
                        min_proj = torch.min(projections)
                        max_proj = torch.max(projections)

                        # 4. Define the start and end points of the capsule's core segment in world space
                        p1 = mean + min_proj * direction
                        p2 = mean + max_proj * direction

                        # 5. Calculate the radius. This is the maximum distance from any point
                        #    to the central line segment defined by p1 and p2.
                        line_vec = p2 - p1
                        line_len_sq = torch.dot(line_vec, line_vec)

                        if line_len_sq < 1e-8:
                            # The points form a sphere, not an elongated shape. The capsule is a sphere.
                            p2 = p1  # The line segment is just a point.
                            dists_to_center = torch.linalg.norm(points - p1, dim=1)
                            radius = torch.max(dists_to_center)
                        else:
                            # For each point, find the closest point on the line segment p1-p2
                            t = torch.clamp(torch.matmul(points - p1, line_vec) / line_len_sq, 0.0, 1.0)
                            closest_points_on_line = p1.unsqueeze(0) + t.unsqueeze(1) * line_vec.unsqueeze(0)
                            
                            # The radius is the maximum distance from any original point to its closest point on the line
                            dists_to_line = torch.linalg.norm(points - closest_points_on_line, dim=1)
                            radius = torch.max(dists_to_line)

                        return {
                            'start': p1.unsqueeze(0),
                            'end': p2.unsqueeze(0),
                            'radius': radius.item() * padding_factor
                        }
                    

                    def _rotation_matrix_between_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                        """
                        Computes the rotation matrix that rotates unit vector 'a' to unit vector 'b'.
                        
                        Args:
                            a (torch.Tensor): The starting unit vector (shape: 3).
                            b (torch.Tensor): The target unit vector (shape: 3).

                        Returns:
                            torch.Tensor: The 3x3 rotation matrix.
                        """
                        # Ensure vectors are unit vectors
                        a = torch.nn.functional.normalize(a, dim=0)
                        b = torch.nn.functional.normalize(b, dim=0)
                        
                        v = torch.cross(a, b)
                        s = torch.linalg.norm(v)
                        c = torch.dot(a, b)
                        
                        # Handle the case where vectors are parallel or anti-parallel
                        if s < 1e-8:
                            # If vectors are in the same direction, return identity
                            if c > 0:
                                return torch.eye(3, device=a.device, dtype=a.dtype)
                            # If vectors are in opposite directions, return 180-degree rotation
                            else:
                                return -torch.eye(3, device=a.device, dtype=a.dtype)

                        vx = torch.tensor([[ 0.0, -v[2],  v[1]],
                                        [ v[2],  0.0, -v[0]],
                                        [-v[1],  v[0],  0.0]], device=a.device, dtype=a.dtype)
                                        
                        R = torch.eye(3, device=a.device, dtype=a.dtype) + vx + vx @ vx * ((1 - c) / (s**2))
                        return R


                    def create_capsule_mesh(capsule_params: dict, resolution: int = 20):
                        """
                        Generates vertices and faces for a 3D mesh that represents an arbitrarily oriented capsule.

                        Args:
                            capsule_params (dict): A dictionary containing capsule parameters.
                            resolution (int): The resolution of the generated cylinder and spheres.

                        Returns:
                            Tuple[np.ndarray, np.ndarray]: vertices and faces of the capsule mesh.
                        """
                        p_start = capsule_params['start']
                        p_end = capsule_params['end']
                        radius = capsule_params['radius']

                        line_vec = p_end.squeeze() - p_start.squeeze()
                        height = torch.linalg.norm(line_vec)

                        if height > 1e-6:
                            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height.item(), resolution=resolution)
                        else:
                            cylinder = o3d.geometry.TriangleMesh()
                        
                        sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution).translate([0, 0, height.item() / 2])
                        sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution).translate([0, 0, -height.item() / 2])
                        base_capsule_mesh = cylinder + sphere1 + sphere2

                        # --- CORRECTED ROTATION CALCULATION ---
                        
                        z_axis = torch.tensor([0.0, 0.0, 1.0], device=p_start.device, dtype=torch.float)
                        
                        # Handle the zero-height case (sphere)
                        if height < 1e-6:
                            rotation_matrix = torch.eye(3, device=p_start.device, dtype=torch.float)
                        else:
                            target_axis = torch.nn.functional.normalize(line_vec, dim=0)
                            # Use our new, correct helper function
                            rotation_matrix = _rotation_matrix_between_vectors(z_axis, target_axis)
                        
                        translation_vector = (p_start.squeeze() + p_end.squeeze()) / 2.0
                        
                        transform_matrix = torch.eye(4, device=p_start.device, dtype=torch.float)
                        transform_matrix[:3, :3] = rotation_matrix
                        transform_matrix[:3, 3] = translation_vector

                        base_verts_torch = torch.from_numpy(np.asarray(base_capsule_mesh.vertices)).to(dtype=torch.float, device=p_start.device)
                        base_verts_h = torch.cat([base_verts_torch, torch.ones(base_verts_torch.shape[0], 1, device=p_start.device)], dim=1)
                        transformed_verts_h = torch.matmul(transform_matrix, base_verts_h.T).T
                        
                        final_vertices = transformed_verts_h[:, :3].cpu().numpy()
                        final_faces = np.asarray(base_capsule_mesh.triangles)

                        return torch.from_numpy(final_vertices).to(dtype=torch.float), torch.from_numpy(final_faces).to(dtype=torch.long, device=self.device)


                    capsule_params = fit_capsule_to_points(vertices)
                    vertices, faces = create_capsule_mesh(capsule_params, resolution=40)
                    
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
                self.joints_lower.append(torch.tensor(body.joint.limits[0], dtype=torch.float, device=self.device) if body.joint.limits is not None else torch.tensor(-3.14, dtype=torch.float, device=self.device))
                self.joints_upper.append(torch.tensor(body.joint.limits[1], dtype=torch.float, device=self.device) if body.joint.limits is not None else torch.tensor(3.14, dtype=torch.float, device=self.device))
            for children in body.children:
                set_joint_range_recurse(children)
        
        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.tensor(self.joints_lower, dtype=torch.float, device=self.device)
        self.joints_upper = torch.tensor(self.joints_upper, dtype=torch.float, device=self.device)
        
        # self.joint_coords_map = self.get_joint_world_coordinates_dict()
        # print("--- Joint Coordinates Dictionary ---")
        # for name, coords in self.joint_coords_map.items():
        #     print(f"Joint '{name}': {coords}")

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


    def get_joint_world_coordinates_dict(self) -> Dict[str, torch.Tensor]:
        """
        Computes the world coordinates of all non-fixed joints after forward kinematics.

        This method should be called AFTER `compute_forward_kinematics`. It leverages
        the fact that a joint's world position is the origin of its child link's frame.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping each joint name to a tensor
                                    of its world coordinates. The tensor will have a
                                    shape of (T, 3), where T is the batch size from
                                    the `qpos` input.
        """
        if self.current_status is None:
            raise ValueError("compute_forward_kinematics must be called before querying joint coordinates.")
        
        joint_coords = {}

        # We traverse the kinematic chain to find each joint and its corresponding child link
        def traverse_chain(body):
            # We are interested in joints that can move
            if body.joint.joint_type != "fixed":
                joint_name = body.joint.name
                link_name = body.link.name

                # Get the Transform3d object for the child link
                # This contains a batch of T transformations
                link_transform = self.current_status[link_name]

                # Get the (T, 4, 4) transformation matrix
                matrix = link_transform.get_matrix()

                # The translation part of the matrix is the world coordinate of the link's origin,
                # which is the same as the joint's world position.
                # Slicing `[:, :3, 3]` selects all items in the batch (T),
                # the first 3 rows of the last column.
                coords = matrix[..., :3, 3]  # Shape: (T, 3)
                joint_coords[joint_name] = coords.squeeze(0)  # Remove batch dim if T=1

            # Recursively apply to all children
            for child in body.children:
                traverse_chain(child)

        # Start the traversal from the root of the chain
        traverse_chain(self.chain._root)

        tips = self.get_tip_points().detach().cpu()
        for ii, name in enumerate(["THJ0", "FFJ0", "MFJ0", "RFJ0", "LFJ0"]):
            joint_coords[name] = tips[..., ii, :]  # Add batch dimension

        return joint_coords


    def compute_forward_kinematics(self, qpos: torch.Tensor):
        """
        qpos: (T, n, ) 关节位置, support batch operation
        """
        self.current_status = self.chain.forward_kinematics(qpos)

    # -------------------------------------------------------------------------- #
    #  基于 PyTorch Kinematics/3D 的工具函数 (来自 HandModelMJCF)
    # -------------------------------------------------------------------------- #
    def get_trimesh_data_o(self):
        """
        Get full mesh
        Deprecated: use get_trimesh_data_single or get_trimesh_data instead.

        Returns
        -------
        data: trimesh.Trimesh
        """
        data = trimesh.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            data += trimesh.Trimesh(vertices=v, faces=f)
        return data
    

    def get_trimesh_data_single(self) -> o3d.geometry.TriangleMesh:
        """
        Gets the combined Open3D mesh for a SINGLE pose.
        This function assumes `compute_forward_kinematics` was called with a 
        non-batched or batch-of-1 qpos.
        
        Returns:
            o3d.geometry.TriangleMesh: A single mesh object.
        """
        full_mesh = o3d.geometry.TriangleMesh()
        
        for link_name in self.mesh:
            # For a single pose, the transform object is used directly
            # transform_points will return vertices of shape (N, 3)
            v_tensor = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            
            # If input was batched (T=1), squeeze out the batch dim
            if v_tensor.dim() == 3:
                v_tensor = v_tensor.squeeze(0)

            v_numpy = v_tensor.detach().cpu().numpy()
            f_numpy = self.mesh[link_name]['faces'].detach().cpu().numpy()

            link_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(v_numpy),
                triangles=o3d.utility.Vector3iVector(f_numpy)
            )
            full_mesh += link_mesh
            
        full_mesh = full_mesh.simplify_vertex_clustering(voxel_size=0.005)
        full_mesh.compute_vertex_normals()

        return full_mesh


    def get_trimesh_data(self) -> List[o3d.geometry.TriangleMesh]:
        """
        Gets a list of combined Open3D meshes for a BATCH of poses.
        This function assumes `compute_forward_kinematics` was called with a
        batched qpos of shape (T, n).

        Returns:
            List[o3d.geometry.TriangleMesh]: A list of T mesh objects, where each
                                             mesh corresponds to a pose in the batch.
        """
        # 1. Determine the batch size (T) from the stored transformations
        # We can peek at any link's status to find the length.
        if not self.mesh or self.current_status is None:
            return []
        first_link_name = next(iter(self.mesh))
        batch_size = len(self.current_status[first_link_name])

        batch_meshes = []
        # 2. Iterate through each pose in the batch (from 0 to T-1)
        # for t in tqdm(range(batch_size), desc="Generating meshes for batch poses"):
        for t in range(batch_size):
            # For each pose, create one combined mesh
            full_mesh_for_t = o3d.geometry.TriangleMesh()

            # 3. Iterate through each link for the current pose 't'
            for link_name in self.mesh:
                # Get the transformation for the current link AND current time step 't'
                transform_for_t = self.current_status[link_name][t]

                # Apply the single transformation to the link's local vertices
                v_tensor = transform_for_t.transform_points(self.mesh[link_name]['vertices'])
                
                # The result is already (N, 3), perfect for a single mesh
                v_numpy = v_tensor.detach().cpu().numpy()
                f_numpy = self.mesh[link_name]['faces'].detach().cpu().numpy()

                link_mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(v_numpy),
                    triangles=o3d.utility.Vector3iVector(f_numpy)
                )
                full_mesh_for_t += link_mesh
            
            # 4. Post-process the combined mesh for this single time step
            full_mesh_for_t = full_mesh_for_t.simplify_vertex_clustering(voxel_size=0.005)
            full_mesh_for_t.compute_vertex_normals()

            # 5. Add the completed mesh for this time step to our results list
            batch_meshes.append(full_mesh_for_t)

        return batch_meshes


    def get_surface_points(self):
        """
        Get surface points
        
        Returns
        -------
        points: (`n_surface_points`, 3)
            surface points
        """
        points = [self.current_status[link_name].transform_points(self.mesh[link_name]['surface_points']) for link_name in self.mesh]
        return torch.cat(points, dim=0)


    def get_all_joints_in_mano_order(self):
        """
        Get all joints in MANO order.

        Returns
        -------
        points: (`batch_size`, 16, 3)
            all joints in MANO order
        """
        joint_names_mano_order = [
            "WRJ1",
            "THJ4",
            "THJ2",
            "THJ1",
            "THJ0",
            "FFJ3",
            "FFJ2",
            "FFJ1",
            "FFJ0",
            "MFJ3",
            "MFJ2",
            "MFJ1",
            "MFJ0",
            "RFJ3",
            "RFJ2",
            "RFJ1",
            "RFJ0",
            "LFJ3",
            "LFJ2",
            "LFJ1",
            "LFJ0",
        ]
        points = []
        joints_dict = self.get_joint_world_coordinates_dict()
        for joint_name in joint_names_mano_order:
            points.append(joints_dict[joint_name])
        points = torch.stack(points, dim=-2)
        return points

    def get_align_points(self):
        """
        Get aligned points.

        Returns
        -------
        points: (`batch_size`, `n_aligned_points`, 3)
            aligned points
        """
        points = [
            self.current_status[link_name].transform_points(self.mesh[link_name]['fingertip_keypoints'])
            for link_name in self.mesh if self.mesh[link_name]['fingertip_keypoints'].shape[0] > 0
        ]
        return torch.cat(points, dim=-2)

    def get_tip_points(self):
        return self.get_align_points()[..., [8, 1, 3, 5, 7], :]


    def get_penetration_keypoints(self):
        """
        Get penetration keypoints
        
        Returns
        -------
        points: (`batch_size`, `n_keypoints`, 3) torch.Tensor
            penetration keypoints
        """
        points = [
            self.current_status[link_name].transform_points(self.mesh[link_name]['penetration_keypoints'])
            for link_name in self.mesh if self.mesh[link_name]['penetration_keypoints'].shape[0] > 0
        ]
        return torch.cat(points, dim=-2)

    def get_contact_candidates(self):
        """
        Get all contact candidates
        
        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        points = [self.current_status[link_name].transform_points(self.mesh[link_name]['contact_candidates']) for link_name in self.mesh]
        return torch.cat(points, dim=0)

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

    def cal_distance(self, obj_points, object_normal, return_idx=False):
        """
        计算接触点到物体表面的投影距离。

        Args:
            obj_points (torch.Tensor): 物体表面点云，形状为 [B, N, 3]。
            object_normal (torch.Tensor): 物体表面点云对应的法线向量，形状为 [B, N, 3]。

        Returns:
            torch.Tensor: 一个标量张量，表示平均距离。
        """
        contact_points = self.get_contact_candidates()
        if contact_points.dim() == 2:
            contact_points = contact_points.unsqueeze(0)
        obj_points_tensor = torch.from_numpy(obj_points).to(dtype=torch.float, device=self.device)
        obj_normal_tensor = torch.from_numpy(object_normal).to(dtype=torch.float, device=self.device)
        # print(obj_points_tensor.shape, obj_normal_tensor.shape, contact_points.shape)
        batch_size = contact_points.shape[0]

        # 为每个 contact_point 找到 obj_points 中最近的一个点
        knn_result = knn_points(contact_points, obj_points_tensor, K=1) # query_points (p1): [B, M, 3], search_points (p2): [B, N, 3]
        nearest_indices = knn_result.idx
        # 3a. Broadcast the source tensors to match the query batch size.
        expanded_obj_points = obj_points_tensor.expand(batch_size, -1, -1)
        expanded_obj_normals = obj_normal_tensor.expand(batch_size, -1, -1)
        # 3b. Prepare the index for gathering. We need to gather all 3 coordinates (x,y,z) for each point.
        index_for_gather = nearest_indices.expand(-1, -1, 3)

        # --- Step 4: FIX - Perform the Gather Operation Correctly ---
        nearest_obj_points = torch.gather(expanded_obj_points, 1, index_for_gather)
        nearest_normals = torch.gather(expanded_obj_normals, 1, index_for_gather)
        vector = contact_points - nearest_obj_points # 形状: [M, 3]
        signed_distance = (vector * nearest_normals.detach()).sum(dim=-1) # 形状: [M] 我们使用 detach() 来防止梯度流向法线向量
        distance = torch.abs(signed_distance).mean()
        
        if return_idx:
            return distance, contact_points, nearest_obj_points
        return distance

    def self_penetration(self, threshold: float = 0.1) -> torch.Tensor:
        """
        计算基于关键点的自穿透能量。

        此版本假设关键点已在世界坐标系下，并且处理的是单个实例（无批处理维度）。

        Args:
            threshold (float): 两个关键点被视为穿透的距离阈值。

        Returns:
            torch.Tensor: 一个张量(Batch, )，表示总的自穿透惩罚能量。
        """
        # 1. 直接获取世界坐标系下的所有自穿透关键点
        # 预期形状为 (N, 3)，其中 N 是关键点数量
        points = self.get_penetration_keypoints()

        # 2. 高效计算所有点对之间的欧氏距离矩阵
        # torch.cdist(A, A) 会生成一个 (N, N) 的矩阵，其中 matrix[i, j] 是点 i 和点 j 之间的距离
        dist_matrix = torch.cdist(points, points, p=2)

        # 3. 计算穿透深度
        # 对于距离小于阈值的点对，该值为正
        penetration_depth = threshold - dist_matrix

        # 4. 只保留正的穿透深度（即实际发生的穿透），其他置为0
        # 这相当于一个ReLU激活函数
        penetration_penalty = torch.relu(penetration_depth)

        # 5. 求和得到总能量
        # 此时，矩阵的对角线（点与自身的距离为0）会产生 `threshold` 的惩罚值，这是不正确的。
        # 同时，(i, j) 和 (j, i) 的惩罚被计算了两次。
        # 通过只对上三角部分（不含对角线）求和，可以完美解决这两个问题。
        total_energy = torch.triu(penetration_penalty, diagonal=1).mean()

        return total_energy

    def self_penetration_part(self, threshold: float = 0.02) -> torch.Tensor:
        """
        计算基于关键点的自穿透能量。

        Args:
            threshold (float): 两个关键点被视为穿透的距离阈值。

        Returns:
            torch.Tensor: 一个张量(Batch, )，表示总的自穿透惩罚能量。
        """
        # 1. 直接获取世界坐标系下的所有自穿透关键点
        # 预期形状为 (B, N, 3)，其中 N 是关键点数量
        points = self.get_penetration_keypoints()
        if points.dim() == 2:
            points = points.unsqueeze(0)  # (1, N, 3)
        points = points[..., :16, :].reshape(-1, 4, 4, 3)  # (B, N, 3)

        points_A = points[:, :3, :, :]  # Shape: (B, 3, 4, 3)
        points_B = points[:, 1:, :, :]  # Shape: (B, 3, 4, 3)
        diff_vector = points_A - points_B  # Shape remains (B, 3, 4, 3)
        distance_matrix = torch.linalg.norm(diff_vector, dim=-1)
        dist_matrix = threshold - distance_matrix
        dist_matrix = torch.relu(dist_matrix)
        penetration_penalty = dist_matrix.mean()

        return penetration_penalty

    def get_E_joints(self):
        """
        Calculate joint energy
        
        Returns
        -------
        E_joints: (N,) torch.Tensor
            joint energy
        """
        qpos = self.dexpose[self.pin2pk][6:]
        E_joints = torch.sum((qpos > self.joints_upper[6:]) * (qpos - self.joints_upper[6:]), dim=-1) + \
            torch.sum((qpos < self.joints_lower[6:]) * (self.joints_lower[6:] - qpos), dim=-1)
        return E_joints

    def cal_object_penetration(
        self, 
        obj_points, 
        return_penetrating_points: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算物体点云到手部表面网格的穿透能量。

        该函数通过计算SDF（符号距离场）来衡量穿透。
        SDF的符号约定为：手部网格内部为正，外部为负。

        Args:
            obj_points (torch.Tensor): 
                在世界坐标系下的物体表面点云，形状为 (B, N, 3) 或 (N, 3)，
                其中 B 是批次大小，N 是点的数量。
            return_penetrating_points (bool, optional):
                如果为 True，函数将额外返回穿透点的坐标。默认为 False。

        Returns:
            torch.Tensor: 
                一个标量张量，表示总的穿透惩罚能量。
            or
            Tuple[torch.Tensor, torch.Tensor]:
                如果 return_penetrating_points 为 True，则返回一个元组：
                - penetration_energy (torch.Tensor): 穿透能量。
                - penetrating_points (torch.Tensor): 穿透点的坐标，形状为 (K, 3)，
                其中 K 是穿透点的数量。如果处理的是批处理数据，则返回
                形状为 (B, K, 3) 的张量。
        """
        # --- Input Handling: Ensure obj_points has a batch dimension for consistency ---
        obj_points_tensor = torch.from_numpy(obj_points).to(dtype=torch.float, device=self.device)
        is_batched = obj_points_tensor.dim() == 3
        if not is_batched:
            obj_points_tensor = obj_points_tensor.unsqueeze(0)  # Shape becomes (1, N, 3)

        batch_size = obj_points_tensor.shape[0]
        
        per_link_sdf_list = []
        links_to_ignore = {
            'forearm', 'wrist', 'ffknuckle', 'mfknuckle', 'rfknuckle', 'lfknuckle', 'thbase', 'thhub'
        }

        for link_name, link_data in self.mesh.items():
            if link_name in links_to_ignore:
                continue

            face_verts = self.current_status[link_name].transform_points(link_data['face_verts'])
            batch_signed_dist_list = []
            for b in range(batch_size):
                # compute_sdf expects (N, 3) and (F, 3, 3)
                dist_sq, signs, _, _ = compute_sdf(obj_points_tensor[b].to('cuda'), face_verts.to('cuda'))
                signed_dist = torch.sqrt(dist_sq.clamp(min=1e-8)) * (-signs)
                batch_signed_dist_list.append(signed_dist)
            
            # Stack results for this link across the batch
            per_link_sdf_list.append(torch.stack(batch_signed_dist_list, dim=0))

        if not per_link_sdf_list:
            energy = torch.tensor(0.0, device=self.device)
            if return_penetrating_points:
                # Return an empty tensor with correct shape if no points are penetrating
                empty_points = torch.empty(batch_size, 0, 3, device=self.device)
                return energy, empty_points if is_batched else empty_points.squeeze(0)
            return energy

        # all_sdfs shape: (num_links, B, num_obj_points)
        all_sdfs = torch.stack(per_link_sdf_list, dim=0)

        # We need to find max over links (dim=0)
        # pointwise_max_sdf shape: (B, num_obj_points)
        pointwise_max_sdf, _ = torch.max(all_sdfs, dim=0)

        # --- This is where the new logic is added ---

        # 1. Calculate the penetration penalty matrix
        penetration_matrix = torch.relu(pointwise_max_sdf)

        # 2. Calculate the final energy (the mean of all positive SDF values)
        penetration_energy = penetration_matrix.mean()

        # 3. If the user doesn't want the points, we can return early.
        if not return_penetrating_points:
            # If input was not batched, we shouldn't change the original function's return type
            return penetration_energy

        # 4. Find the indices of the penetrating points for EACH batch item
        # penetration_mask will be a boolean tensor of shape (B, N)
        penetration_mask = pointwise_max_sdf > 0

        # We can't directly index with a boolean mask for batched data to get a ragged tensor.
        # Instead, we return a list of tensors, one for each batch item.
        obj_points_tensor = obj_points_tensor.to(dtype=torch.float, device='cpu')
        penetration_mask = penetration_mask.to(dtype=torch.bool, device='cpu')
        penetrating_points = torch.stack([
            obj_points_tensor[b][penetration_mask[b]] for b in range(batch_size)
        ])
        
        # If the original input was not batched, we return a single tensor.
        # Otherwise, we return the list of tensors.
        final_penetrating_points = penetrating_points.squeeze(0)

        return penetration_energy, final_penetrating_points

    def cal_object_penetration_approx(self, obj_points: torch.Tensor, object_normal: torch.Tensor) -> torch.Tensor:
        """
        使用物体法线和KNN，近似计算物体到手部的穿透能量。

        这是一种速度更快但精度较低的替代方法，它不计算真实的SDF。
        它计算的是从每个物体点到最近的手部表面采样点的距离，
        并将其投影到物体法线上。

        Args:
            obj_points (torch.Tensor): 
                世界坐标系下的物体点云，形状为 (N, 3)。
            object_normal (torch.Tensor): 
                对应的物体表面法线，形状为 (N, 3)。

        Returns:
            torch.Tensor: 
                一个标量张量，表示总的近似穿透能量。
        """
        # 1. 获取当前姿态下，手部所有的表面采样点
        # 形状为 (M, 3)，M 是手部表面点总数
        hand_surface_points = self.get_surface_points()

        if hand_surface_points.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)

        # 2. 为每个物体点找到最近的手部表面点
        # knn_points 需要一个 batch 维度
        # obj_points (p1): [1, N, 3]
        # hand_surface_points (p2): [1, M, 3]
        knn_result = knn_points(obj_points.unsqueeze(0), hand_surface_points.unsqueeze(0), K=1)

        # knn_result.idx 的形状是 [1, N, 1]，包含了最近邻点的索引
        # 我们去掉多余的维度，得到形状为 [N] 的索引张量
        nearest_indices = knn_result.idx[0, :, 0] # 形状: [N]

        # 提取出那些最近的手部表面点
        nearest_hand_points = hand_surface_points[nearest_indices] # 形状: [N, 3]

        # 3. 计算从物体点指向手部点的向量
        vector = nearest_hand_points - obj_points # 形状: [N, 3]

        # 4. 将向量投影到物体法线上
        # 点积 (A * B).sum(-1) 得到投影长度（带符号）
        # 如果手部点在法线“内部”方向，投影为负值。
        # 我们关心的是穿透，即手部点在法线“外部”的情况，所以我们取反。
        penetration_depth = -(vector * object_normal.detach()).sum(dim=-1)

        # 5. 计算总能量，只考虑正值（实际穿透）
        penetration_energy = torch.relu(penetration_depth).sum()

        return penetration_energy
    



def load_robot(robot_name) -> HandRobotWrapper:

    assert robot_name in ['shadow_hand', 'inspire_hand', 'ability_hand', 'dclaw_gripper', 'panda_gripper', 'schunk_svh_hand', 'schunk_hand', 'allegro_hand', 'barrett_hand', 'leap_hand'], f"Robot {robot_name} not supported."
    hand_asset_root = os.path.join("/home/qianxu/Desktop/Project/DexPose/thirdparty/dex-retargeting/assets/robots/hands", robot_name)
    side = "right"
    robot = HandRobotWrapper(robot_name, os.path.join(hand_asset_root, f'new_{side}_glb.urdf'),
                    os.path.join(hand_asset_root, f'meshes'),
                    n_surface_points=500, device=torch.device('cuda:0'), tip_aug=None)
    return robot
