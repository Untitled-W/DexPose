"""
Base classes for human hand manipulation dataset processing.
Provides a unified framework for loading different datasets into a common format.
"""

import os
import json
import pickle
import logging
import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
from manotorch.manolayer import ManoLayer
try:
    from .vis_utils import vis_pc_coor_plotly
except ImportError:
    from loss import vis_pc_coor_plotly

# Import your existing utility functions (adjust import paths as needed)
try:
    from .utils import (
        apply_transformation_pt, merge_two_parts, get_key_hand_joints,
        get_contact_points, get_contact_pts_from_whole_field_perstep,
        intepolate_feature, farthest_point_sampling, mirror_data
    )
except ImportError:
    try:
        from utils import (
            apply_transformation_pt, merge_two_parts, get_key_hand_joints,
            get_contact_points, get_contact_pts_from_whole_field_perstep,
            intepolate_feature, farthest_point_sampling, mirror_data
        )
    except ImportError:
        logging.error("Utility functions not found. Ensure utils.py is in the correct directory.")
        raise ImportError("Utility functions not found. Ensure utils.py is in the correct directory.")


@dataclass
class SequenceData:
    """Standardized data structure for a manipulation sequence."""
    # Hand data
    hand_joints: torch.Tensor  # T X 21 X 3
    hand_params: torch.Tensor  # T X hand_param_dim
    side: int  # 0 for left, 1 for right
    
    # Object data
    obj_points: torch.Tensor  # N X 3
    obj_points_ori: Optional[torch.Tensor]  # N_ori X 3 (higher resolution)
    obj_normals: torch.Tensor  # N X 3
    obj_normals_ori: Optional[torch.Tensor]  # N_ori X 3
    obj_features: Optional[torch.Tensor]  # N X feature_dim
    obj_transformations: torch.Tensor  # N_obj X T X 4 X 4
    obj_num: torch.Tensor # N_obj
    obj_num_ori: torch.Tensor # N_obj_ori
    
    # Contact data
    contact_indices: torch.Tensor  # T X n_contact
    
    # Metadata
    mesh_path: str
    mesh_norm_transformation: torch.Tensor  # 4 X 4
    task_description: str
    seq_len: int
    extra_info: Optional[Dict[str, Any]] = None

  
class BaseDatasetProcessor(ABC):
    """Base class for dataset processors."""
    
    def __init__(
        self,
        root_path: str,
        task_interval: int = 1,
        pt_n: int = 100,
        cpt_n: int = 100,
        seq_data_name: str = "dataset",
        contact_threshold: float = 0.1,
        **kwargs
    ):
        self.root_path = root_path
        self.task_interval = task_interval
        self.pt_n = pt_n
        self.cpt_n = cpt_n
        self.seq_data_name = seq_data_name
        self.contact_threshold = contact_threshold
        
        self.mano_layer_left = ManoLayer(center_idx=0, side='left', rot_mode="quat", use_pca=False).cuda()
        self.mano_layer_right = ManoLayer(center_idx=0, side='right', rot_mode="quat", use_pca=False).cuda()
        
        # Setup paths
        self._setup_paths()
        
        # Initialize data list
        self.data_ls = self._get_data_list()
        
        # Setup save path
        try:
            from .config import WORKING_DIR
        except ImportError:
            try:
                from config import WORKING_DIR
            except ImportError:
                WORKING_DIR = './'  # Default fallback
        self.seq_save_path = f'{WORKING_DIR}data/seq_data/seq_{seq_data_name}_{task_interval}.p'
    
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
    def _process_hand_data(self, raw_data: Dict[str, Any], side: str, frame_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process hand data to get joints and parameters."""
        pass
    
    @abstractmethod
    def _load_object_data(self, raw_data: Dict[str, Any], frame_indices: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[float]]:
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
    
    def _downsample_points(self, points_ls: List[torch.Tensor], area_ls: List[float], target_points: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Downsample points based on surface area."""
        whole_area = sum(area_ls)
        n_pt_ls = [int(target_points * area / whole_area) for area in area_ls]
        n_pt_ls[-1] += target_points - sum(n_pt_ls)
        
        points_idx_ls = [
            farthest_point_sampling(points.unsqueeze(0), n_pt)[:n_pt] 
            for points, n_pt in zip(points_ls, n_pt_ls)
        ]
        points_ds = [points[idx] for points, idx in zip(points_ls, points_idx_ls)]
        
        return points_ds, points_idx_ls
    
    def _compute_normals(self, points_ls: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute normals for point clouds."""
        normals_ls = []
        for points in points_ls:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points.cpu().numpy())
            pc.estimate_normals()
            normals = np.asarray(pc.normals)
            normals_ls.append(torch.from_numpy(normals).float().to('cuda'))
        return normals_ls
    
    def _get_contact_data(self, hand_joints: torch.Tensor, obj_points_trans: torch.Tensor, 
                         features: torch.Tensor, obj_points_ori: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
        """Get contact points and indices."""
        hand_key_joints = get_key_hand_joints(hand_joints)
        hand_key_features = intepolate_feature(hand_key_joints, features, obj_points_trans)
        
        ### 0701!!! Unmatched function call
        contact_points = get_contact_points(
            hand_joints, obj_points_ori, 
            # method='neighbor',  ###0701 no these arg here!
            # n_pts=self.cpt_n, 
            # only_traj_cut=True
        )
        '''
        contact_points: torch.Size([2494, 700, 3])
        obj_points_ori.shape: torch.Size([2494, 1000, 3])
        hand_joints.shape: torch.Size([2494, 21, 3])
        hand_key_joints.shape: torch.Size([2494, 6, 3])
        hand_key_features.shape: torch.Size([2494, 6, 2048])
        '''

        ### Get start and end indices for the sequence
        s_idx, e_idx = 0, 0
        
        if s_idx == -1:
            return -1, -1, None
            
        contact_indices = get_contact_pts_from_whole_field_perstep(
            obj_points_trans[s_idx:e_idx:10].cpu(),  # Convert to CPU for processing
            features.cpu(), 
            hand_key_features[s_idx:e_idx:10].cpu()
        )
        
        return s_idx, e_idx, contact_indices
    
    def process_sequence(self, raw_data: Dict[str, Any], side: str) -> Optional[SequenceData]:
        """Process a single sequence into standardized format."""
        # Get frame indices
        frame_indices = raw_data.get('frame_indices', []) 
        if not frame_indices:
            return None
        
        raw_data['extra_info'] = raw_data.get('extra_info', {})
        raw_data['extra_info']['seq_name'] = raw_data.get('seq_name', 'unknown')
        raw_data['extra_info']['task_name'] = raw_data.get('task_name', 'unknown')

        
        # Apply coordinate transformation
        yup2xup = self._apply_coordinate_transform(side).to('cuda')
        
        # Process hand data
        hand_joints, hand_params = self._process_hand_data(raw_data, side, frame_indices)
        if hand_joints is None:  return None
        
        # Load object data
        points_ls, obj_transf_ls, features_ls, area_ls = self._load_object_data(raw_data, frame_indices, hand_joints,
                                                                                disable_feature=False)
        if points_ls is None: return None

        # Convert to tensors and apply transformations
        points_ls = [torch.from_numpy(points).float().to('cuda') if isinstance(points, np.ndarray) 
                    else points.to('cuda') for points in points_ls]
        obj_transf_ls = [yup2xup @ torch.from_numpy(obj_transf).float().to('cuda') 
                        if isinstance(obj_transf, np.ndarray) else yup2xup @ obj_transf.to('cuda') 
                        for obj_transf in obj_transf_ls]
        
        # Compute normals
        normals_ls = self._compute_normals(points_ls)
        
        # Downsample points
        points_ds, points_idx_ls = self._downsample_points(points_ls, area_ls, self.pt_n)
        points_1000, points_idx_1000 = self._downsample_points(points_ls, area_ls, 1000)

        # Downsample features and normals

        features_ds = [features[idx] for features, idx in zip(features_ls, points_idx_ls)]
        normals_ds = [normals[idx] for normals, idx in zip(normals_ls, points_idx_ls)]
        normals_1000 = [normals[idx] for normals, idx in zip(normals_ls, points_idx_1000)]
        
        # Transform points
        obj_points_trans_ori = merge_two_parts([
            apply_transformation_pt(points, transform) 
            for points, transform in zip(points_1000, obj_transf_ls)
        ])
        obj_points_trans = merge_two_parts([
            apply_transformation_pt(points, transform) 
            for points, transform in zip(points_ds, obj_transf_ls)
        ])


        VIS_DEBUG = False
        if VIS_DEBUG == True:
            ii = int(.5 * len(hand_joints))
            vis_pc_coor_plotly([obj_points_trans_ori[ii].cpu().numpy()], 
                            gt_hand_joints=hand_joints[ii].cpu().numpy(), 
                            show_axis=True, filename=f"visualize_0705_test_dataset/debug_camera_view/{raw_data['extra_info']['task_name']}_{raw_data['extra_info']['seq_name']}_{ii}")
            import sys; sys.exit(0)

        # Get contact data

        # s_idx, e_idx, contact_indices = self._get_contact_data(hand_joints, obj_points_trans, torch.cat(features_ds, dim=0), obj_points_trans_ori)
        
        s_idx, e_idx, contact_indices = 0, len(hand_joints), None  # Default to full sequence for now
        if s_idx == -1 or e_idx - s_idx < 60:
            return None
        
        sequence_data = SequenceData(
            hand_joints=hand_joints[s_idx:e_idx].cpu(),
            hand_params=hand_params[s_idx:e_idx].cpu(),
            side=1 if side == 'r' else 0,
            
            obj_points=torch.cat([points.cpu() for points in points_ds], dim=0),
            obj_points_ori=torch.cat([points.cpu() for points in points_1000], dim=0),
            obj_normals=torch.cat([normals.cpu() for normals in normals_ds], dim=0),
            obj_normals_ori=torch.cat([normals.cpu() for normals in normals_1000], dim=0),
            obj_features=torch.cat([features.cpu() for features in features_ds], dim=0) if features_ds else None,
            obj_num=torch.tensor([points.shape[0] for points in points_ds]),  # Number of points per object
            obj_num_ori=torch.tensor([points.shape[0] for points in points_1000]),  # Number of points per object in original resolution
            
            obj_transformations=torch.stack([obj_transf.cpu()[s_idx:e_idx] for obj_transf in obj_transf_ls]),
            contact_indices=contact_indices.cpu() if contact_indices is not None else torch.empty(0),
            
            mesh_path=raw_data.get('mesh_path', ''),
            mesh_norm_transformation=torch.eye(4).cpu().to(torch.float32),
            task_description=self._get_task_description(raw_data),
            seq_len=e_idx - s_idx,
            extra_info=raw_data.get('extra_info', {})
        )

        
        return sequence_data

    
    def process_all_sequences(self) -> List[Dict[str, Any]]:
        """Process all sequences in the dataset."""
        whole_data_ls = []
        bad_seq_num = 0
        
        # for idx in tqdm(range(len(self.data_ls)), desc=f"Processing {self.seq_data_name}"):
        for idx in tqdm(range(10), desc=f"Processing {self.seq_data_name}"):
            # try:
            data_item = self.data_ls[idx]
            sequence_list = self._load_sequence_data(data_item) ###0630
            
            for seq_data in sequence_list:
                for side in ['l', 'r']:
                    if seq_data.get(f'{side}_valid', True):  # Check if this side has valid data

                        processed_seq = self.process_sequence(seq_data, side)  ###0630(error here!)
                        if processed_seq is not None:
                            # Convert to dict format for compatibility
                            seq_dict = self._sequence_data_to_dict(processed_seq)
                            whole_data_ls.append(seq_dict)
                            
                            # Create mirrored version if needed
                            if self._should_create_mirror(seq_data, side):
                                mirrored_dict = mirror_data(seq_dict, side)
                                mirrored_dict['side'] = 1 - seq_dict['side']
                                whole_data_ls.append(mirrored_dict)
                
                        
            # except Exception as e:
            #     logging.error(f"Error processing {data_item}: {e}")
            #     bad_seq_num += 1
        
        logging.info(f"Processed {len(whole_data_ls)} sequences, {bad_seq_num} failed")
        return whole_data_ls
    
    def _sequence_data_to_dict(self, seq_data: SequenceData) -> Dict[str, Any]:
        """Convert SequenceData to dictionary format for backward compatibility."""
        side_str = 'r' if seq_data.side == 1 else 'l'
        other_side = 'l' if seq_data.side == 1 else 'r'
        
        return {
            f'{side_str}h_joints': seq_data.hand_joints,
            f'{side_str}h_params': seq_data.hand_params,
            
            f'{side_str}o_points': seq_data.obj_points,
            f'{side_str}o_points_ori': seq_data.obj_points_ori,
            f'{side_str}o_normals': seq_data.obj_normals,
            f'{side_str}o_normals_ori': seq_data.obj_normals_ori,
            f'{side_str}o_features': seq_data.obj_features,
            f'{side_str}o_transf': seq_data.obj_transformations,
            f'{side_str}o_num': seq_data.obj_num,
            f'{side_str}o_num_ori': seq_data.obj_num_ori,
            
            f'{other_side}h_joints': None,
            f'{other_side}h_params': None,
            f'{other_side}o_points': None,
            f'{other_side}o_points_ori': None,
            f'{other_side}o_normals': None,
            f'{other_side}o_normals_ori': None,
            f'{other_side}o_features': None,
            f'{other_side}o_transf': None,
            
            'contact_indices': seq_data.contact_indices,
            'mesh_path': seq_data.mesh_path,
            'mesh_norm_transformation': seq_data.mesh_norm_transformation,
            'task_desc': seq_data.task_description,
            'side': seq_data.side,
            'seq_len': seq_data.seq_len,
            'extra_desc': seq_data.extra_info
        }
    
    def _should_create_mirror(self, seq_data: Dict[str, Any], side: str) -> bool:
        """Determine if a mirrored version should be created."""
        # Override in subclasses if needed
        return False
    
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
        data_list = self.process_all_sequences()
        self.save_processed_data(data_list)
        return data_list


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


def load_multiple_datasets(processor_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Load and combine multiple datasets."""
    all_data = []
    
    for config in processor_configs:
        processor_name = config.pop('processor_name')
        processor_class = DatasetRegistry.get_processor(processor_name)
        processor = processor_class(**config)
        data = processor.run()
        all_data.extend(data)
    
    logging.info(f"Combined {len(all_data)} sequences from {len(processor_configs)} datasets")
    return all_data


