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


ORIGIN_DATA_PATH = {
    "Taco": "/home/qianxu/Desktop/Project/interaction_pose/data/Taco",
    "Oakinkv2": '/home/qianxu/Desktop/New_Folder/OakInk2/OakInk-v2-hub',
}

HUMAN_SEQ_PATH = {
    "Taco": "/home/qianxu/Desktop/Project/interaction_pose/data/Taco/human_save",
    "Oakinkv2": "/home/qianxu/Desktop/Project/interaction_pose/data/Oakinkv2/human_save",
}

DEX_SEQ_PATH = {
    "Taco": "/home/qianxu/Desktop/Project/interaction_pose/data/Taco/dex_save",
    "Oakinkv2": "/home/qianxu/Desktop/Project/interaction_pose/data/Oakinkv2/dex_save",
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
    # Hand data
    hand_type: str
    hand_poses: torch.Tensor  # T X n_dof
    side: int  # 0 for left, 1 for right

    # Object data
    obj_poses: torch.Tensor  # K X T X 4 X 4
    obj_point_clouds: torch.Tensor  # K X N X 3
    obj_feature: torch.Tensor  # K X N X d

    # Metadata
    object_mesh_path: List[str]
    frame_indices: List[int]  # Frame indices in the original sequence
    task_description: str
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
        
        self.mano_layer_left = ManoLayer(center_idx=0, side='left', rot_mode="quat", use_pca=False).cuda()
        self.mano_layer_right = ManoLayer(center_idx=0, side='right', rot_mode="quat", use_pca=False).cuda()
        
        # Setup paths
        self._setup_paths()
        
        # Initialize data list
        self.data_ls = self._get_data_list()
        self.sequence_indices = sequence_indices if sequence_indices is not None else list(range(len(self.data_ls)))

        self.seq_save_path = f'{save_path}/seq_{seq_data_name}_{task_interval}.p'


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

        # Load object data
        object_transf_ls, object_name_ls, object_mesh_path_ls = self._get_object_info(raw_data, frame_indices)
        if object_transf_ls is None: return None

        # Convert to tensors and apply transformations
        obj_transf_ls = [yup2xup @ torch.from_numpy(obj_transf).float().to('cuda') 
                        if isinstance(obj_transf, np.ndarray) else yup2xup @ obj_transf.to('cuda') 
                        for obj_transf in object_transf_ls]
        
        sequence_data = HumanSequenceData(
            hand_tsls=hand_tsl,
            hand_coeffs=hand_coeffs,
            side=1 if side == 'r' else 0,
            obj_poses=torch.stack(obj_transf_ls),
            object_names=object_name_ls,
            object_mesh_path=object_mesh_path_ls,
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

            DEBUG = False

            if DEBUG:
                data_item = self.data_ls[idx]
                sequence_list = self._load_sequence_data(data_item) 
                
                for seq_data in sequence_list:
                    for side in ['l', 'r']:
                        if seq_data.get(f'{side}_valid', True):  # Check if this side has valid data

                            processed_seq = self.process_sequence(seq_data, side)
                            if processed_seq is not None: whole_data_ls.append(processed_seq)
            else:
                try:
                    data_item = self.data_ls[idx]
                    sequence_list = self._load_sequence_data(data_item) 
                    
                    for seq_data in sequence_list:
                        for side in ['l', 'r']:
                            if seq_data.get(f'{side}_valid', True):  # Check if this side has valid data

                                processed_seq = self.process_sequence(seq_data, side)
                                if processed_seq is not None: whole_data_ls.append(processed_seq)
                        
                except Exception as e:
                    logging.error(f"Error processing {data_item}: {e}")
                    bad_seq_num += 1
        
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


