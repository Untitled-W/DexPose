"""
Specific dataset processors for OAKINKv2 and TACO datasets.
"""

import os
import ast
import json
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix
from manotorch.manolayer import ManoLayer
try:
    from .base_dataset import BaseDatasetProcessor, DatasetRegistry
    from .utils import apply_transformation_pt
    from .vis_utils import get_multiview_dff
    from .mesh_renderer import MeshRenderer3D
    from .vis_utils import vis_pc_coor_plotly
except ImportError:
    from base_dataset import BaseDatasetProcessor, DatasetRegistry
    from utils import apply_transformation_pt
    from loss import get_multiview_dff
    from mesh_renderer import MeshRenderer3D
    from loss import vis_pc_coor_plotly
from pytorch3d.transforms import (
    quaternion_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d,
    axis_angle_to_quaternion
)

import warnings
warnings.filterwarnings("ignore")


def geneate_camera_params(point: np.ndarray, n_views: int = 4) -> List[Dict[str, Any]]:
    camera_params = [
        {'elev': 30, 'azim': 30, 'fov': 60},     # Front
        {'elev': 30, 'azim': 120, 'fov': 60},    # Right
        {'elev': 30, 'azim': 210, 'fov': 60},   # Back
        {'elev': 30, 'azim': 300, 'fov': 60},   # Left
    ]
    return camera_params


def generate_single_view_from_hand(hand_center, object_center, object_transformation):
    '''
    input:
        - hand_center: T x 3,
        - object_center: 3,
        - object_transformation: T x 4 x 4
        all in torch.Tensor
    intermediate output:
        - view_ray: T x 3, (in object coordinate system), meaning the view ray from hand to object
    output:
        - camera_params: List[Dict[str, Any]], generate by the average of intermediate output
    '''
    # Compute the average view ray from hand to object in object coordinates
    # hand_center: T x 3, object_center: 3, object_transformation: T x 4 x 4

    # Compute view rays in world coordinates
    view_rays = hand_center - object_center  # (T, 3)

    # Transform view rays to object coordinates (inverse transform for each frame)
    view_rays_obj = []
    for t in range(object_transformation.shape[0]):
        # Inverse rotation and translation
        R_obj = object_transformation[t, :3, :3]
        t_obj = object_transformation[t, :3, 3]
        ray = view_rays[t] - t_obj
        ray_obj = np.linalg.inv(R_obj) @ ray
        view_rays_obj.append(ray_obj)
    view_rays_obj = np.stack(view_rays_obj, axis=0)  # (T, 3)

    # Average view ray in object coordinates
    avg_view_ray = np.mean(view_rays_obj, axis=0)
    avg_view_ray = avg_view_ray / (np.linalg.norm(avg_view_ray) + 1e-8)

    # Convert view ray to spherical coordinates for camera placement
    elev = np.degrees(np.arcsin(avg_view_ray[2]))
    azim = np.degrees(np.arctan2(avg_view_ray[1], avg_view_ray[0]))
    fov = 60

    camera_params = [{
        'elev': float(elev),
        'azim': float(azim),
        'fov': float(fov)
    }]
    return camera_params


@DatasetRegistry.register('oakinkv2')
class OAKINKv2Processor(BaseDatasetProcessor):
    """Processor for OAKINKv2 dataset."""
    
    def _setup_paths(self):
        """Setup OAKINKv2-specific paths."""
        self.desc_dir = os.path.join(self.root_path, 'program', 'desc_info')
        self.program_dir = os.path.join(self.root_path, 'program', 'program_info')
        self.anno_dir = os.path.join(self.root_path, 'anno_preview')
        self.obj_dir = os.path.join(self.root_path, 'object_preview', 'align_ds')
        
        # Load object info
        obj_info_path = os.path.join(self.root_path, 'object_preview', 'obj_desc.json')
        if os.path.exists(obj_info_path):
            with open(obj_info_path, 'r') as f:
                self.obj_info = json.load(f)
        else:
            self.obj_info = {}  # Fallback for testing
        
        # Skip problematic objects
        self.skip_obj_name_ls = [
            "O02@0039@00002", "O02@0039@00003", "O02@0039@00004",  # microwave
            "O02@0045@00001", "O02@0045@00002", "O02@0045@00003",  # lamp
            "O02@0054@00001",  # whiteboard
            'O02@0038@00001',  # weight metric
            'O02@0018@00001', 'O02@0018@00002',  # box
            'O02@0201@00001',  # tripod
            'O02@0053@00001', 'O02@0053@00002',  # laptop
            'O02@0019@00001', 'O02@0019@00002',  # drawer
        ]
    
    def _get_data_list(self) -> np.ndarray:
        """Get list of OAKINKv2 data files."""
        if not os.path.exists(self.anno_dir):
            return np.array([])  # Return empty array if directory doesn't exist
        
        data_ls = os.listdir(self.anno_dir)
        data_ls = sorted([
            os.path.splitext(filename)[0] 
            for filename in data_ls 
            if filename.endswith('.pkl')
        ])
        return np.asarray(data_ls)
    
    def _load_sequence_data(self, task_name: str) -> List[Dict[str, Any]]:
        """Load raw sequence data for OAKINKv2."""
        sequences = []
        
        try:
            # Load annotation files
            anno_filepath = os.path.join(self.anno_dir, task_name + '.pkl')
            with open(anno_filepath, "rb") as f:
                anno = pickle.load(f)
            
            with open(os.path.join(self.desc_dir, task_name + '.json'), 'r') as f:
                desc = json.load(f)
            
            with open(os.path.join(self.program_dir, task_name + '.json'), 'r') as f:
                program = json.load(f)
            
            # Parse segments
            segs = [ast.literal_eval(seg) for seg in desc.keys()]
            
            for seg_idx, seg in enumerate(segs):
                try:
                    description = desc[str(seg)]['seg_desc']
                    lh_primitive = program[str(seg)]['primitive_lh']
                    rh_primitive = program[str(seg)]['primitive_rh']
                    
                    # Check if segments should be skipped
                    l_skip, r_skip = self._check_skip_conditions(program[str(seg)])
                    
                    # Process left hand
                    if lh_primitive is not None and not l_skip:
                        l_obj_names = program[str(seg)]['obj_list_lh']
                        l_desc = self._get_hand_description(program[str(seg)], lh_primitive, 'lh')
                        
                        sequences.append({
                            'task_name': task_name,
                            'seg': seg,
                            'side': 'l',
                            'frame_indices': list(range(seg[0][0], seg[0][1], self.task_interval)),
                            'obj_names': l_obj_names,
                            'primitive': lh_primitive,
                            'description': l_desc,
                            'extra_desc': description,
                            'anno': anno,
                            'l_valid': True,
                            'r_valid': False
                        })
                    
                    # Process right hand
                    if rh_primitive is not None and not r_skip:
                        r_obj_names = program[str(seg)]['obj_list_rh']
                        r_desc = self._get_hand_description(program[str(seg)], rh_primitive, 'rh')
                        
                        sequences.append({
                            'task_name': task_name,
                            'seg': seg,
                            'side': 'r',
                            'frame_indices': list(range(seg[1][0], seg[1][1], self.task_interval)),
                            'obj_names': r_obj_names,
                            'primitive': rh_primitive,
                            'description': r_desc,
                            'extra_desc': description,
                            'anno': anno,
                            'l_valid': False,
                            'r_valid': True
                        })
                        
                except Exception as e:
                    logging.error(f"Error processing segment {seg} in {task_name}: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error loading {task_name}: {e}")
            return []
        
        return sequences
    
    def _check_skip_conditions(self, program_seg: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check if left/right hand segments should be skipped."""
        left_skip = True
        right_skip = True
        
        if (program_seg['obj_list_lh'] is not None and 
            len(program_seg['obj_list_lh']) > 0):
            left_skip = False
            for obj in program_seg['obj_list_lh']:
                if obj in self.skip_obj_name_ls:
                    left_skip = True
                    break
        
        if (program_seg['obj_list_rh'] is not None and 
            len(program_seg['obj_list_rh']) > 0):
            right_skip = False
            for obj in program_seg['obj_list_rh']:
                if obj in self.skip_obj_name_ls:
                    right_skip = True
                    break
        
        return left_skip, right_skip
    
    def _get_hand_description(self, program_seg: Dict[str, Any], primitive: str, hand: str) -> str:
        """Generate hand description."""
        obj_key = f'obj_list_{hand}'
        obj_list = program_seg[obj_key]
        
        if '_' in primitive:
            primitive_parts = primitive.split('_')
            if primitive_parts[-1] not in ['onto', 'outside', 'inside']:
                return ' '.join(primitive_parts)
            else:
                obj_names = list(set([self.obj_info[obj]['obj_name'] for obj in obj_list]))
                return f"{' '.join(primitive_parts)} the {' and '.join(obj_names)}"
        else:
            obj_names = list(set([self.obj_info[obj]['obj_name'] for obj in obj_list]))
            return f"{primitive} the {' and '.join(obj_names)}"

    def _extract_hand_coeffs(self, raw_data: Dict[str, Any], side: str, frame_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        anno = raw_data['anno']
        # Extract MANO parameters
        h_coeffs = torch.concatenate([
            anno["raw_mano"][ii][f'{side}h__pose_coeffs'] 
            for ii in frame_indices
        ], dim=0).to('cuda') # (T, 16, 4)
        
        h_tsl = torch.concatenate([
            anno["raw_mano"][ii][f'{side}h__tsl'] 
            for ii in frame_indices
        ], dim=0).to('cuda') # (T, 3)

        h_betas = torch.concatenate([
            anno["raw_mano"][ii][f'{side}h__betas'] 
            for ii in frame_indices
        ], dim=0).to('cuda') # (T, 10)

        return h_coeffs, h_tsl, h_betas


    def _process_hand_data(self, raw_data: Dict[str, Any], side: str, frame_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process OAKINKv2 hand data."""
        
        h_coeffs, h_tsl, h_betas = self._extract_hand_coeffs(raw_data, side, frame_indices)
        # h_betas = torch.zeros_like(h_betas).to('cuda')  # Use zero betas
        
        yup2xup = self._apply_coordinate_transform(side).to('cuda')
        
        # Transform hand pose
        hand_transformation = torch.eye(4).to('cuda').unsqueeze(0).repeat((h_coeffs.shape[0], 1, 1))
        hand_rotation_quat = h_coeffs[:, 0]
        hand_transformation[:, :3, :3] = quaternion_to_matrix(hand_rotation_quat)
        h_coeffs[:, 0] = matrix_to_quaternion(yup2xup[:3, :3] @ hand_transformation[:, :3, :3])
        h_coeffs_rot6d = matrix_to_rotation_6d(quaternion_to_matrix(h_coeffs))
        h_tsl = self._apply_transformation_pt(h_tsl, yup2xup)
        
        # Get hand joints
        if side == 'l':
            h_dict = self.mano_layer_left(h_coeffs, h_betas)
        else:
            h_dict = self.mano_layer_right(h_coeffs, h_betas)
        
        hand_joints = h_dict.joints + h_tsl[:, None, :]
        hand_params = torch.cat([h_coeffs_rot6d.flatten(-2, -1), h_tsl], dim=-1)
        
        return hand_joints, hand_params

    def _apply_transformation_pt(self, points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """Apply transformation to points."""
        return apply_transformation_pt(points, transform)
    
    def _load_object_data(self, raw_data: Dict[str, Any], frame_indices: List[int], hand_joints: torch.Tensor, disable_feature=False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[float]]:
        """Load OAKINKv2 object data."""
        obj_names = raw_data['obj_names']
        anno = raw_data['anno']
        
        # Get object meshes
        mesh_paths = []
        for obj_name in obj_names:
            obj_folder = os.path.join(self.obj_dir, obj_name)
            mesh_path = None
            if os.path.isdir(obj_folder):
                for fname in os.listdir(obj_folder):
                    if fname.lower().endswith('.ply') or fname.lower().endswith('.obj'):
                        mesh_path = os.path.join(obj_folder, fname)
                        break
            if mesh_path is None:
                raise FileNotFoundError(f"No .ply or .obj mesh found in {obj_folder}")
            mesh_paths.append(mesh_path)
        
        # Get object transformations
        try:
            obj_transf_all_list = [anno['obj_transf'][obj] for obj in obj_names]
        except KeyError:
            return None, None, None, None
        
        obj_transf_list = [
            np.stack([obj_transf_all[frame] for frame in frame_indices if frame in obj_transf_all])
            for obj_transf_all in obj_transf_all_list
        ]

        areas = []
        points_list = []
        features_list = []

        # for mesh in meshes:
        for obj_id, path in enumerate(mesh_paths):

            mesh = o3d.io.read_triangle_mesh(path)
            areas.append(self._compute_mesh_surface_area(mesh))

            if np.asarray(mesh.vertices).shape[0] == 0: return None, None, None, None
            
            if disable_feature:
                points = np.asarray(mesh.vertices)
                points_list.append(torch.from_numpy(points).float())
                features_list.append(torch.zeros_like(torch.from_numpy(points)).float().to('cuda'))

            else:
                renderer = MeshRenderer3D(device='cpu')
                renderer.load_mesh(path)     
                
                camera_params = generate_single_view_from_hand(
                    # hand_joints.mean(dim=1).cpu().numpy(),
                    hand_joints[:,0,:].cpu().numpy(),
                    np.asarray(mesh.vertices).mean(axis=0).astype(np.float32),
                    obj_transf_list[obj_id]
                    )
                renderer.setup_cameras(camera_params, auto_distance=True)
                renderer.render_views(image_size=256, lighting_type='ambient')

                renderer.extract_features(ftype='sd_dinov2', prompt='') ###!!!!should be '' not None!

                rgbs, depths, masks, points_ls, features = renderer.get_rendered_data()

                all_points_dff, all_features_dff = get_multiview_dff(
                    points_ls, masks, features, n_points=1000
                )

                points_list.append(all_points_dff)
                features_list.append(all_features_dff)


                VIS_2D = False
                if VIS_2D:
                    from loss import visualize_pointclouds_and_mesh
                    visualize_pointclouds_and_mesh(
                        pointcloud_list=[point.cpu().numpy() for point in points_ls],
                        color_list=[rgb[mask].cpu().numpy() for rgb, mask in zip(rgbs, masks)],
                        mesh=renderer.mesh,
                        # view_names=['Rendered Mesh'],
                        view_names = [f'View {i+1}' for i in range(len(rgbs))],
                        title='Mesh Renderer RGBD Fusion',
                        save_path=f"visualize_0705_test_dataset/debug_2d/{raw_data['task_name']}.html",
                        # use_plotly=True,
                        point_size=4
                    )
                    print()

            
        # Store mesh path info
        raw_data['mesh_path'] = '$'.join(mesh_paths)
        
        return points_list, obj_transf_list, features_list, areas
    
    def _compute_mesh_surface_area(self, mesh) -> float:
        """Compute mesh surface area."""
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        areas = np.zeros(len(triangles))
        
        for i, triangle in enumerate(triangles):
            v0, v1, v2 = vertices[triangle]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas[i] = area
        
        return np.sum(areas)
    
    def _get_task_description(self, raw_data: Dict[str, Any]) -> str:
        """Get task description for OAKINKv2."""
        return raw_data.get('description', '')
    
    def _should_create_mirror(self, seq_data: Dict[str, Any], side: str) -> bool:
        """OAKINKv2 creates mirror for left hand sequences."""
        return side == 'l'


@DatasetRegistry.register('taco')
class TACOProcessor(BaseDatasetProcessor):
    """Processor for TACO dataset."""
    
    def _setup_paths(self):
        """Setup TACO-specific paths."""
        self.hand_pose_dir = os.path.join(self.root_path, 'Hand_Poses')
        self.obj_dir = os.path.join(self.root_path, 'object_models')
        self.obj_pose_dir = os.path.join(self.root_path, 'Object_Poses')
        
        # Test data list
        self.test_data_ls = [53, 181, 25, 73, 35, 115, 89, 160, 7, 189, 166, 105, 201, 81, 38, 66, 56, 141, 142, 138, 59]
    
    def _get_data_list(self) -> np.ndarray:
        """Get list of TACO data items."""
        if not os.path.exists(self.hand_pose_dir):
            return np.array([])  # Return empty array if directory doesn't exist
        return np.asarray(os.listdir(self.hand_pose_dir))
    
    def _load_sequence_data(self, task_name: str) -> List[Dict[str, Any]]:
        """Load raw sequence data for TACO."""
        sequences = []
        
        seq_list = os.listdir(os.path.join(self.hand_pose_dir, task_name))
        
        for seq_name in seq_list:
            try:
                # Get object pose directory
                object_pose_dir = os.path.join(self.obj_pose_dir, task_name, seq_name)
                hand_pose_dir = os.path.join(self.hand_pose_dir, task_name, seq_name)
                
                # Extract tool and target names
                tool_name, target_name = None, None
                for file_name in os.listdir(object_pose_dir):
                    if file_name.startswith("tool_"):
                        tool_name = file_name.split(".")[0].split("_")[-1]
                    elif file_name.startswith("target_"):
                        target_name = file_name.split(".")[0].split("_")[-1]
                
                if tool_name is None or target_name is None:
                    continue
                
                # Extract action info
                verb, tool_real_name, target_real_name = self._extract_action_info(task_name)
                
                # Load poses
                tool_poses = np.load(os.path.join(object_pose_dir, f"tool_{tool_name}.npy"))
                target_poses = np.load(os.path.join(object_pose_dir, f"target_{target_name}.npy"))
                n_frames = tool_poses.shape[0]
                
                # Create sequences for tool (right hand) and target (left hand)
                sequences.extend([
                    {
                        'task_name': task_name,
                        'seq_name': seq_name,
                        'side': 'r',  # Tool manipulation with right hand
                        'obj_name': tool_name,
                        'obj_poses': tool_poses,
                        'hand_pose_dir': hand_pose_dir,
                        'n_frames': n_frames,
                        'action_verb': verb,
                        'obj_real_name': tool_real_name,
                        'frame_indices': list(range(0, n_frames, self.task_interval)),
                        'r_valid': True,
                        'l_valid': False
                    },
                    {
                        'task_name': task_name,
                        'seq_name': seq_name,
                        'side': 'l',  # Target holding with left hand
                        'obj_name': target_name,
                        'obj_poses': target_poses,
                        'hand_pose_dir': hand_pose_dir,
                        'n_frames': n_frames,
                        'action_verb': 'hold',
                        'obj_real_name': target_real_name,
                        'frame_indices': list(range(0, n_frames, self.task_interval)),
                        'r_valid': False,
                        'l_valid': True
                    }
                ])
                
            except Exception as e:
                logging.error(f"Error processing TACO sequence {task_name}/{seq_name}: {e}")
                continue
        
        return sequences
    
    def _extract_action_info(self, task_name: str) -> Tuple[str, str, str]:
        """Extract action information from task name."""
        import re
        match = re.match(r'\(([^,]+), ([^,]+), ([^,]+)\)', task_name.strip("'"))
        if match:
            return match.group(1), match.group(2), match.group(3)
        else:
            return "unknown", "object", "target"
    
    def _process_hand_data(self, raw_data: Dict[str, Any], side: str, frame_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process TACO hand data."""
        hand_pose_dir = raw_data['hand_pose_dir']
        device = torch.device('cuda')
        
        # Load hand shape
        if side == 'r':
            hand_beta = pickle.load(open(os.path.join(hand_pose_dir, "right_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().to(device)
        else:
            hand_beta = pickle.load(open(os.path.join(hand_pose_dir, "left_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().to(device)
        
        # Load hand poses
        if side == 'r':
            hand_pkl = pickle.load(open(os.path.join(hand_pose_dir, "right_hand.pkl"), "rb"))
        else:
            hand_pkl = pickle.load(open(os.path.join(hand_pose_dir, "left_hand.pkl"), "rb"))
        
        # Extract poses and translations
        hand_theta_list = []
        hand_trans_list = []
        
        keys = list(hand_pkl.keys())
        keys.sort()
        
        for key in keys:
            hand_theta_list.append(hand_pkl[key]["hand_pose"].detach().cpu().numpy())
            hand_trans_list.append(hand_pkl[key]["hand_trans"].detach().cpu().numpy())
        
        hand_thetas_raw = torch.from_numpy(np.float32(hand_theta_list)).to(device)
        hand_trans_raw = torch.from_numpy(np.float32(hand_trans_list)).to(device)
        
        # Convert to quaternions
        hand_thetas_raw = axis_angle_to_quaternion(hand_thetas_raw.reshape(hand_thetas_raw.shape[0], 16, 3))
        
        # Apply coordinate transformation
        yup2xup = torch.eye(4).to(device)
        
        hand_transform = torch.eye(4).to('cuda').unsqueeze(0).repeat((hand_thetas_raw.shape[0], 1, 1))
        hand_transform[:, :3, :3] = quaternion_to_matrix(hand_thetas_raw[:, 0])
        hand_transform[:, :3, 3] = hand_trans_raw
        hand_transform = yup2xup @ hand_transform
        hand_thetas_raw[:, 0] = matrix_to_quaternion(hand_transform[:, :3, :3])
        hand_trans_raw = hand_transform[:, :3, 3]
        
        # Use cached optimized hand parameters if available
        cache_path = f'/home/qianxu/cache_data/hand/{raw_data["seq_name"]}_{raw_data["task_name"]}{side}_hand_thetas.pt'
        if os.path.exists(cache_path):
            hand_thetas = torch.load(cache_path).to(device)
            hand_trans = torch.load(cache_path.replace('_thetas', '_trans')).to(device)
        else:
            # Use raw parameters (optimization would be called here in original code)
            hand_thetas = hand_thetas_raw
            hand_trans = hand_trans_raw
        
        # Get subset based on frame indices
        hand_thetas = hand_thetas[frame_indices]
        hand_trans = hand_trans[frame_indices]
        
        # Generate hand joints
        hand_betas = torch.zeros(hand_thetas.shape[0], 10).to(device)
        if side == 'l':
            hand_joints = self.mano_layer_left(hand_thetas, hand_betas).joints + hand_trans[:, None, :]
        else:
            hand_joints = self.mano_layer_right(hand_thetas, hand_betas).joints + hand_trans[:, None, :]
        
        # Convert to rotation 6D for parameters
        hand_thetas_rot6d = matrix_to_rotation_6d(quaternion_to_matrix(hand_thetas))
        hand_params = torch.cat([hand_thetas_rot6d.flatten(-2, -1), hand_trans], dim=-1)
        
        return hand_joints, hand_params
    

    def _get_coordinate_transform(self, side: str) -> torch.Tensor:
        """Get coordinate transformation for TACO dataset."""
        yup2xup = torch.eye(4)
        if side == 'l':  # Target
            yup2xup[:3, :3] = torch.from_numpy(
                R.from_euler('xyz', [0, 0, -90], degrees=True).as_matrix()
            ).to(torch.float32)
        else:  # Tool
            yup2xup[:3, :3] = torch.from_numpy(
                R.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()
            ).to(torch.float32)
        return yup2xup
    

    def _load_object_data(self, raw_data: Dict[str, Any], frame_indices: List[int], hand_joints, disable_feature=False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[float]]:
        """Load TACO object data."""
        obj_name = raw_data['obj_name']
        obj_poses = raw_data['obj_poses']
        
        # Load object mesh
        obj_path = os.path.join(self.obj_dir, obj_name + "_cm.obj")
        obj_mesh = o3d.io.read_triangle_mesh(obj_path)
        obj_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(obj_mesh.vertices) * 0.01)
        

        # Compute area (simplified)
        area = self._compute_mesh_surface_area(obj_mesh)

        # Apply coordinate transformation
        yup2xup = self._apply_coordinate_transform(raw_data['side']).T
        obj_poses_transformed = yup2xup.cpu().numpy().astype(np.float32) @ obj_poses
        obj_transf_subset = obj_poses_transformed[frame_indices]


        if disable_feature:

            points = np.asarray(obj_mesh.vertices).astype(np.float32)  # Use mesh vertices as points
            features = torch.zeros_like(torch.from_numpy(points)).to('cuda')  # Dummy features

        else: 

            renderer = MeshRenderer3D(device='cpu')
            renderer.load_mesh(obj_path, scale=0.01)  # Scale to match original mesh size

            camera_params = generate_single_view_from_hand(
                    # hand_joints.mean(dim=1).cpu().numpy(),
                    hand_joints[:,0,:].cpu().numpy(),
                    np.asarray(obj_mesh.vertices).mean(axis=0).astype(np.float32),
                    obj_transf_subset
                    )
            renderer.setup_cameras(camera_params, auto_distance=True)
            renderer.render_views(image_size=256, lighting_type='ambient')

            renderer.extract_features(ftype='sd_dinov2', prompt='') ###!!!!should be '' not None!

            rgbs, depths, masks, points_ls, features = renderer.get_rendered_data()

            all_points_dff, all_features_dff = get_multiview_dff(
                points_ls, masks, features, n_points=1000
            )

            points = all_points_dff[frame_indices]
            features = all_features_dff[frame_indices]

        
        # Store mesh path
        raw_data['mesh_path'] = obj_path
        
        return [points], [obj_transf_subset], [features.float()], [area]
        
    def _compute_mesh_surface_area(self, mesh) -> float:
        """Compute mesh surface area."""
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        areas = np.zeros(len(triangles))
        
        for i, triangle in enumerate(triangles):
            v0, v1, v2 = vertices[triangle]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas[i] = area
        
        return np.sum(areas)

    def _get_task_description(self, raw_data: Dict[str, Any]) -> str:
        """Get task description for TACO."""
        verb = raw_data['action_verb']
        obj_name = raw_data['obj_real_name']
        return f"{verb} the {obj_name}"
    
    def _should_create_mirror(self, seq_data: Dict[str, Any], side: str) -> bool:
        """TACO creates mirror for left hand (target) sequences."""
        return side == 'l'


