import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import torch
import copy
from tqdm import tqdm
from data_process.mesh_renderer import MeshRenderer3D
from data_process.vis_utils import visualize_pointclouds_and_mesh, get_multiview_dff, extract_pca, test_pca_matching
from manotorch.manolayer import ManoLayer
from utils.vis_utils import vis_pc_coor_plotly
from utils.tools import get_key_hand_joints, apply_transformation_pt, intepolate_feature, get_contact_pts, find_longest_false_substring, from_hand_rot6d, to_hand_rot6d
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix, quaternion_to_matrix, matrix_to_quaternion, axis_angle_to_quaternion


WMQ_IS_USING = True

if WMQ_IS_USING:

    ORIGIN_DATA_PATH = {
        "Taco": "/home/qianxu/Desktop/Project/interaction_pose/data/Taco",
        "Oakinkv2": '/home/qianxu/Desktop/New_Folder/OakInk2/OakInk-v2-hub',
        'DexYCB': '/home/qianxu/Desktop/Project/interaction_pose/thirdparty_module/dex-retargeting/data'
    }

    HUMAN_SEQ_PATH = {
        "Taco": "/home/qianxu/Desktop/Project/interaction_pose/data/Taco/human_save",
        "Oakinkv2": "/home/qianxu/Desktop/Project/interaction_pose/data/Oakinkv2/human_save",
        'DexYCB': '/home/qianxu/Desktop/Project/interaction_pose/thirdparty_module/dex-retargeting/data/human_save'
    }

    DEX_SEQ_PATH = {
        "Taco": "/home/qianxu/Desktop/Project/interaction_pose/data/Taco/dex_save",
        "Oakinkv2": "/home/qianxu/Desktop/Project/interaction_pose/data/Oakinkv2/dex_save",
        'DexYCB': '/home/qianxu/Desktop/Project/interaction_pose/thirdparty_module/dex-retargeting/data'
    }

else:

    ORIGIN_DATA_PATH = {
        "Taco": "/home/qianxu/Project24/TACO_Instructions/data",
        "Oakinkv2": '/home/qianxu/Desktop/New_Folder/OakInk2/OakInk-v2-hub',
        'DexYCB': '/home/qianxu/Desktop/Project/interaction_pose/data/DexYCB/dex-ycb-20210415'
    }

    HUMAN_SEQ_PATH = {
        "Taco": "/home/qianxu/Project25/DexPose/Taco/human_save",
        "Oakinkv2": "/home/qianxu/Desktop/Project/interaction_pose/data/Oakinkv2/human_save",
        'DexYCB': '/home/qianxu/Desktop/Project/interaction_pose/data/DexYCB/dex-ycb-20210415/human_save'
    }

    DEX_SEQ_PATH = {
        "Taco": "/home/qianxu/Project25/DexPose/Taco/dex_save",
        "Oakinkv2": "/home/qianxu/Desktop/Project/interaction_pose/data/Oakinkv2/dex_save",
        'DexYCB': '/home/qianxu/Desktop/Project/interaction_pose/data/DexYCB/dex-ycb-20210415/dex_save'
    }

# HumanSequenceData
@dataclass
class HumanSequenceData:
    """Standardized data structure for a human manipulation sequence."""
    # Hand data
    hand_tsls: torch.Tensor  # T X 3
    hand_coeffs: torch.Tensor  # T X 16*4
    side: int  # 0 for left, 1 for right
    
    # Object data
    obj_poses: torch.Tensor  # K X T X 4 X 4
    object_names: List[str]  # Names of objects in the sequence
    object_mesh_path: List[str]
    object_points_ls: List[torch.Tensor]  # K X N X 3
    object_features_ls: List[torch.Tensor]  # K X N X d
    
    # Metadata
    frame_indices: List[int]  # Frame indices in the original sequence
    task_description: str
    which_dataset: str
    which_sequence: str
    extra_info: Optional[Dict[str, Any]] = None


# DexSequenceData
@dataclass
class DexSequenceData:
    """Standardized data structure for a dexterous manipulation sequence."""
    # Dex Hand data
    which_hand: str
    hand_poses: torch.Tensor  # T X n_dof
    side: int  # 0 for left, 1 for right

    # Human Hand data
    hand_tsls: torch.Tensor  # T X 3
    hand_coeffs: torch.Tensor  # T X 16*4

    # Object data
    obj_poses: torch.Tensor  # K X T X 4 X 4
    obj_point_clouds: torch.Tensor  # K X N X 3
    obj_feature: torch.Tensor  # K X N X d
    object_names: List[str]  # Names of objects in the sequence
    object_mesh_path: List[str]

    # Metadata
    frame_indices: List[int]  # Frame indices in the original sequence
    task_description: str
    which_dataset: str
    which_sequence: str
    extra_info: Optional[Dict[str, Any]] = None


# BaseDatasetProcessor
class BaseDatasetProcessor(ABC):
    """Base class for dataset processors."""
    
    def __init__(
        self,
        root_path: str,
        save_path: str,
        which_dataset: str,
        task_interval: int = 1,
        seq_data_name: str = "dataset",
        sequence_indices: Optional[List[int]] = None
    ):
        self.root_path = root_path
        self.task_interval = task_interval
        self.which_dataset = which_dataset
        self.seq_data_name = seq_data_name
        self.sequence_indices = sequence_indices
        
        # Setup paths
        self._setup_paths()
        
        # Initialize data list
        self.data_ls = self._get_data_list()
        self.sequence_indices = sequence_indices if sequence_indices is not None else list(range(len(self.data_ls)))

        self.seq_save_path = f'{save_path}/seq_{seq_data_name}_{task_interval}.p'
        self.renderer = MeshRenderer3D()
        self.renderer.load_featurizer(ftype='sd_dinov2')
        self.manolayer = ManoLayer(rot_mode='quat', side='right', use_pca=False).cuda()

    @staticmethod
    def orientation_to_elev_azim(orientation: np.ndarray) -> tuple:
        """
        Convert a 3D orientation vector to elevation and azimuth angles for PyTorch3D.
        
        PyTorch3D uses y-up coordinate system where:
        - elev: angle between vector and horizontal plane (xz-plane, y=0)
        - azim: angle between projection on xz-plane and reference vector (0,0,1)
        
        Args:
            orientation: 3D vector (unnormalized) pointing from object to camera position
            
        Returns:
            tuple: (elevation, azimuth) in degrees
                - elevation: angle above/below horizontal plane (-90 to 90 degrees)
                - azimuth: angle from +z axis on horizontal plane (0 to 360 degrees)
        """
        # Normalize the orientation vector
        norm = np.linalg.norm(orientation)
        assert norm > 0, "Orientation vector must not be zero"
        
        orientation = orientation / norm
        x, y, z = orientation
        
        # Calculate elevation (angle from horizontal xz-plane)
        # In y-up system, elevation is angle between vector and xz-plane
        elevation = np.arcsin(np.clip(y, -1.0, 1.0))
        elevation_deg = np.degrees(elevation)
        
        # Calculate azimuth (angle from +z axis on xz-plane)
        # Project vector onto xz-plane and measure angle from +z axis
        # In y-up system, azimuth is measured from +z axis towards +x axis
        azimuth = np.arctan2(-x, z)
        azimuth_deg = np.degrees(azimuth)
        
        # Ensure azimuth is in [0, 360) range
        if azimuth_deg < 0:
            azimuth_deg += 360.0
        
        return float(elevation_deg), float(azimuth_deg)

    @staticmethod
    def mirror_data(data_dict, side):
        ### TODO
        if type(data_dict[f'{side}o_transf']) is list:
            obj_transf_list_m = []
            for obj_transf in data_dict[f'{side}o_transf']:
                transl_m = obj_transf[:, :3, 3].clone()
                rot_m = obj_transf[:, :3, :3].clone()

                transl_m[..., 0] *= -1
                rot_m = matrix_to_axis_angle(rot_m)
                rot_m[..., [1, 2]] *= -1
                rot_m = axis_angle_to_matrix(rot_m)

                obj_transf_m = torch.eye(4).to(obj_transf.device).repeat(obj_transf.shape[0], 1, 1)
                obj_transf_m[:, :3, :3] = rot_m
                obj_transf_m[:, :3, 3] = transl_m
                obj_transf_list_m.append(obj_transf_m)
            
            obj_points_list_m = []
            for obj_points in data_dict[f'{side}o_points']:
                obj_points_m = obj_points.clone()
                obj_points_m[..., 0] *= -1
                obj_points_list_m.append(obj_points_m)
            
            obj_points_ori_list_m = []
            for obj_points_ori in data_dict[f'{side}o_points_ori']:
                obj_points_ori_m = obj_points_ori.clone()
                obj_points_ori_m[..., 0] *= -1
                obj_points_ori_list_m.append(obj_points_ori_m)
            
            obj_normals_list_m = []
            for obj_normals in data_dict[f'{side}o_normals']:
                obj_normals_m = obj_normals.clone()
                obj_normals_m[..., 0] *= -1
                obj_normals_list_m.append(obj_normals_m)
            
            obj_normals_ori_list_m = []
            for obj_normals_ori in data_dict[f'{side}o_normals_ori']:
                obj_normals_ori_m = obj_normals_ori.clone()
                obj_normals_ori_m[..., 0] *= -1
                obj_normals_ori_list_m.append(obj_normals_ori_m)
            
            obj_features_list_m = [obj_features.clone() for obj_features in data_dict[f'{side}o_features']]
        else:
            obj_transf = data_dict[f'{side}o_transf']
            transl_m = obj_transf[:, :3, 3].clone()
            rot_m = obj_transf[:, :3, :3].clone()

            transl_m[..., 0] *= -1
            rot_m = matrix_to_axis_angle(rot_m)
            rot_m[..., [1, 2]] *= -1
            rot_m = axis_angle_to_matrix(rot_m)

            obj_transf_m = torch.eye(4).to(obj_transf.device).repeat(obj_transf.shape[0], 1, 1)
            obj_transf_m[:, :3, :3] = rot_m
            obj_transf_m[:, :3, 3] = transl_m
            obj_transf_list_m = obj_transf_m

            obj_points_m = data_dict[f'{side}o_points'].clone()
            obj_points_m[..., 0] *= -1
            obj_points_list_m = obj_points_m

            obj_normals_m = data_dict[f'{side}o_normals'].clone()
            obj_normals_m[..., 0] *= -1
            obj_normals_list_m = obj_normals_m

        ### hand
        hand_joints_m = data_dict[f'{side}h_joints'].clone()
        hand_joints_m[..., 0] *= -1
        hand_params:torch.Tensor = data_dict[f'{side}h_params'].clone()
        hand_params[..., -3] *= -1
        hand_rotvec = from_hand_rot6d(hand_params[:, :-3].reshape(hand_params.shape[0], -1, 6), to_rotvec=True)
        hand_rotvec = hand_rotvec.reshape(hand_params.shape[0], -1, 3)
        hand_rotvec[..., [1, 2]] *= -1
        hand_rot6d_m = to_hand_rot6d(hand_rotvec, from_rotvec=True)
        hand_params_m = torch.cat((hand_rot6d_m.flatten(-2), hand_params[..., -3:]), dim=-1)

        side_m = 'l' if side == 'r' else 'r'

        data_dict_m = {
            f'{side_m}h_joints': hand_joints_m,
            f'{side_m}o_transf': obj_transf_list_m,
            f'{side_m}o_points': obj_points_list_m,
            f'{side_m}o_normals': obj_normals_list_m,
            f'{side_m}h_params': hand_params_m,
            f'{side}h_joints': None,
            f'{side}o_transf': None,
            f'{side}o_points': None,
            f'{side}o_normals': None,
            f'{side}h_params': None,
            'contact_indices': copy.deepcopy(data_dict['contact_indices']),
            'mesh_path': copy.deepcopy(data_dict['mesh_path']),
        }
        return data_dict_m

    @abstractmethod
    def _setup_paths(self):
        """Setup dataset-specific paths."""
        pass
    
    @abstractmethod
    def _get_data_list(self) -> np.ndarray:
        """Get list of data items to process."""
        pass
    
    @abstractmethod
    def _load_sequence_data(self, data_item: str) -> List[Dict[str, Any]]:
        """Load raw sequence data for a data item."""
        pass
    
    @abstractmethod
    def _get_hand_info(self, raw_data: Dict[str, Any], side: str, frame_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process hand data to get joints and parameters."""
        pass
    
    @abstractmethod
    def _get_object_info(self, raw_data: Dict[str, Any], frame_indices: List[int]) -> Tuple[List[torch.Tensor], List[str]]:
        """Load object points, transformations, features, and areas."""
        pass
    
    @abstractmethod
    def _get_task_description(self, raw_data: Dict[str, Any]) -> str:
        """Generate task description."""
        pass

    def _apply_coordinate_transform(self, side: str) -> torch.Tensor:
        """Apply coordinate system transformation based on hand side."""
        from scipy.spatial.transform import Rotation as R
        
        yup2xup = torch.eye(4)
        if side == 'l':
            yup2xup[:3, :3] = torch.from_numpy(
                R.from_euler('xyz', [90, 0, -90], degrees=True).as_matrix()
            ).to(torch.float32)
        else:
            yup2xup[:3, :3] = torch.from_numpy(
                R.from_euler('xyz', [90, 0, 90], degrees=True).as_matrix()
            ).to(torch.float32)
        return yup2xup

    def process_sequence(self, raw_data: Dict[str, Any], side: str) -> Optional[HumanSequenceData]:
        
        """Process a single sequence into standardized format."""
        
        # Get frame indices
        frame_indices = raw_data.get('frame_indices', []) 
        if not frame_indices: return None
        
        # Apply coordinate transformation
        yup2xup = self._apply_coordinate_transform(side).to('cuda')
        
        # Process hand data
        hand_tsl, hand_coeffs = self._get_hand_info(raw_data, side, frame_indices)
        if hand_tsl is None: return None
        hand_dict = self.manolayer(hand_coeffs.reshape(-1, 16, 4), torch.zeros((hand_coeffs.shape[0], 10), device=hand_coeffs.device))
        hand_joints = hand_dict.joints + hand_tsl[:, None, :3]  # T X 16 X 3
        
        # Load object data
        object_transf_ls_raw, object_name_ls, object_mesh_path_ls = self._get_object_info(raw_data, frame_indices)
        if object_transf_ls_raw is None: return None
        obj_transf_ls = [yup2xup @ torch.from_numpy(obj_transf).float().to('cuda') 
                        if isinstance(obj_transf, np.ndarray) else yup2xup @ obj_transf.to('cuda') 
                        for obj_transf in object_transf_ls_raw]

        ### render & feature & contact points
        object_points_ls = []
        object_features_ls = []
        contact_indices_ls = []
        start_idx, end_idx = 1e10, 0
        ### get camera orientation
        render_at_tidx = 0
        obj_inv_transf = torch.inverse(obj_transf_ls[0]).to(hand_joints.device)
        hand_inv_joints = hand_joints @ obj_inv_transf[:, :3, :3].transpose(1, 2) + obj_inv_transf[:, :3, 3].unsqueeze(1)
        orientation = hand_inv_joints[render_at_tidx, 0].cpu().numpy()
        orientation[2] += 0.15
        ### get camera orientation
        for obj_part_idx, object_mesh_path in enumerate(object_mesh_path_ls):
            self.renderer.load_mesh(object_mesh_path, scale=0.01)
            elev, azim = self.orientation_to_elev_azim(orientation)
            camera_params = [
                {'elev': elev, 'azim':  azim, 'fov': 60},     # hand orientation
            ]
            R_w2v, T_w2v = self.renderer.setup_cameras(camera_params, auto_distance=True)
            self.renderer.render_views(image_size=256, lighting_type='ambient')
            texture_prompts = ["a wooden shovel, high quality", "a metal shovel, high quality"]
            self.renderer.augment_textures(texture_prompts, num_variations=1, save_dir="test_res")
            self.renderer.extract_features(prompt="a shovel", w_aug_texture=True)
            rgbs, depths, masks, points_ls, features = self.renderer.get_rendered_data()
            # features: (num_views, num_texture, feature_dim, height, width)
            all_points_dff, all_features_dff = get_multiview_dff(points_ls, masks, features,
                                                    n_points=1000)
            
            ### get contact point indices
            obj_points_trans_ori = apply_transformation_pt(all_points_dff, obj_transf_ls[obj_part_idx])
            hand_key_joints = get_key_hand_joints(hand_joints)
            hand_key_features = intepolate_feature(hand_key_joints, all_features_dff[0], obj_points_trans_ori)
            contact_indices = get_contact_pts(all_points_dff, all_features_dff[0], hand_key_features, n_pts=100 // len(object_mesh_path_ls))
            ### get contact point indices
            
            ### get the contact timesteps
            distance = torch.norm(hand_key_joints[..., None, :] - obj_points_trans_ori[..., None, :, :], dim=-1, p=2)
            min_dis, min_dis_idx = torch.min(distance, dim=-1)
            untouch_mask = torch.min(min_dis, dim=-1)[0] > 0.02
            start_idx_, end_idx_ = find_longest_false_substring(untouch_mask)
            start_idx = min(start_idx, start_idx_)
            end_idx = max(end_idx, end_idx_)
            ### get the contact timesteps

            object_points_ls.append(all_points_dff)
            object_features_ls.append(all_features_dff)
            contact_indices_ls.append(contact_indices)
        contact_indices = torch.cat(contact_indices_ls, dim=-1)
        ### render & feature & contact points

        # #### DEBUG CODE
        # w2c = torch.eye(4).unsqueeze(0)
        # w2c[..., :3, :3] = R_w2v.cpu()
        # w2c[..., :3, 3] = T_w2v.cpu()
        # c2w = torch.inverse(w2c)
        # # self.renderer.visualize_all_views()
        # vis_pc_coor_plotly([object_points_ls[0].cpu().numpy(), orientation.reshape(1, 3)], 
        #                    gt_hand_joints=hand_inv_joints[0].cpu().numpy(),
        #                    transformation_ls=c2w, show_axis=True)       
        # #### DEBUG CODE
        
        ### Return KEYS
        ### TODO self.mirror_data
        ### TODO DICT return 
        ### TODO Sanity Check
        # DICT with keys: ['rh_joints', 'ro_transf', 'ro_points', 'ro_normals', 'rh_params', 
        #  'lh_joints', 'lo_transf', 'lo_points', 'lo_normals', 'lh_params', 
        #  'contact_indices', 'mesh_path', 'side', 'task_desc', 'seq_len']
        sequence_data = HumanSequenceData(
            hand_tsls=hand_tsl.cpu(),
            hand_coeffs=hand_coeffs.cpu(),
            side=1 if side == 'r' else 0,
            obj_poses=torch.stack(obj_transf_ls).cpu(),
            object_names=object_name_ls,
            object_mesh_path=object_mesh_path_ls,
            object_points_ls = object_points_ls,
            object_features_ls = object_features_ls,
            
            frame_indices=frame_indices,
            which_dataset=raw_data['which_dataset'],
            which_sequence=raw_data['which_sequence'],
            task_description=self._get_task_description(raw_data),
            extra_info=raw_data.get('extra_info', {}),
        )
        
        return sequence_data

    def process_all_sequences(self, sequence_indices: List[int]) -> List[Dict[str, Any]]:

        """Process all sequences in the dataset."""
        whole_data_ls = []
        bad_seq_num = 0
        
        # for idx in tqdm(range(len(self.data_ls)), desc=f"Processing {self.seq_data_name}"):
        for idx in tqdm(sequence_indices, desc=f"Processing {self.which_dataset}-{self.seq_data_name}"):
            data_item = self.data_ls[idx]
            sequence_list = self._load_sequence_data(data_item) 
            
            for seq_data in sequence_list:
                for side in ['l', 'r']:
                    if seq_data.get(f'{side}_valid', True):  # Check if this side has valid data
                        processed_seq = self.process_sequence(seq_data, side)
                        if processed_seq is not None: whole_data_ls.append(processed_seq)
                    
        
        logging.info(f"Processed {len(whole_data_ls)} sequences, {bad_seq_num} failed")
        return whole_data_ls
 
    def save_processed_data(self, data_list: List[Dict[str, Any]]):
        """Save processed data to file."""
        import joblib
        os.makedirs(os.path.dirname(self.seq_save_path), exist_ok=True)
        with open(self.seq_save_path, 'wb') as f:
            joblib.dump(data_list, f)
        logging.info(f"Saved {len(data_list)} sequences to {self.seq_save_path}")

    def run(self) -> List[Dict[str, Any]]:
        """Main processing pipeline."""
        logging.info(f"Starting {self.__class__.__name__} processing...")
        data_list = self.process_all_sequences(self.sequence_indices)
        self.save_processed_data(data_list)
        return data_list
    

# DatasetRegistry
class DatasetRegistry:
    """Registry for dataset processors."""
    
    _processors = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a processor."""
        def decorator(processor_class):
            cls._processors[name] = processor_class
            return processor_class
        return decorator
    
    @classmethod
    def get_processor(cls, name: str) -> BaseDatasetProcessor:
        """Get a processor by name."""
        if name not in cls._processors:
            raise ValueError(f"Unknown processor: {name}")
        return cls._processors[name]
    
    @classmethod
    def list_processors(cls) -> List[str]:
        """List available processors."""
        return list(cls._processors.keys())


