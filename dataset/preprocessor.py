"""
Specific dataset processors for OAKINKv2 and TACO datasets.
"""

import os
import ast
import json
import random
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import (
    quaternion_to_matrix, matrix_to_quaternion, axis_angle_to_quaternion, quaternion_to_axis_angle
)
from pytransform3d import transformations as pt
from manotorch.manolayer import ManoLayer

from .base_structure import BaseDatasetProcessor, DatasetRegistry, HumanSequenceData, ORIGIN_DATA_PATH, HUMAN_SEQ_PATH
from utils.tools import apply_transformation_pt
from utils.vis_utils import visualize_human_sequence
from utils.dexycb_dataset import DexYCBVideoDataset, YCB_CLASSES


# Oakinkv2Processor
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
                            'which_dataset': 'Oakinkv2',
                            'which_sequence': task_name,
                            'task_name': task_name,
                            'seg': seg,
                            'side': 'l',
                            'frame_indices': list(range(seg[0][0], seg[0][1], self.task_interval)),
                            'obj_names': l_obj_names,
                            'primitive': lh_primitive,
                            'description': l_desc,
                            'extra_info': description,
                            'anno': anno,
                            'l_valid': True,
                            'r_valid': False
                        })
                    
                    # Process right hand
                    if rh_primitive is not None and not r_skip:
                        r_obj_names = program[str(seg)]['obj_list_rh']
                        r_desc = self._get_hand_description(program[str(seg)], rh_primitive, 'rh')
                        
                        sequences.append({
                            'which_dataset': 'Oakinkv2',
                            'which_sequence': task_name,
                            'task_name': task_name,
                            'seg': seg,
                            'side': 'r',
                            'frame_indices': list(range(seg[1][0], seg[1][1], self.task_interval)),
                            'obj_names': r_obj_names,
                            'primitive': rh_primitive,
                            'description': r_desc,
                            'extra_info': description,
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
    
    def _get_task_description(self, raw_data: Dict[str, Any]) -> str:
        """Get task description for OAKINKv2."""
        return raw_data.get('description', '')

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

    def _apply_transformation_pt(self, points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """Apply transformation to points."""
        return apply_transformation_pt(points, transform)

    def _get_hand_info(self, raw_data: Dict[str, Any], side: str, frame_indices: List[int], pre_trans: torch.Tensor = None, device = torch.device('cuda')) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process OAKINKv2 hand data."""
        
        h_coeffs, h_tsl, h_betas = self._extract_hand_coeffs(raw_data, side, frame_indices)
        
        yup2xup = self._apply_coordinate_transform(side).to('cuda')
        
        # Transform hand pose
        hand_transformation = torch.eye(4).to('cuda').unsqueeze(0).repeat((h_coeffs.shape[0], 1, 1))
        hand_rotation_quat = h_coeffs[:, 0]
        hand_transformation[:, :3, :3] = quaternion_to_matrix(hand_rotation_quat)
        h_coeffs[:, 0] = matrix_to_quaternion(yup2xup[:3, :3] @ hand_transformation[:, :3, :3])
        h_tsl = self._apply_transformation_pt(h_tsl, yup2xup)

        return h_tsl.squeeze(), h_coeffs

    def _get_object_info(self, raw_data: Dict[str, Any], frame_indices: List[int]) -> Tuple[List[torch.Tensor], List[str]]:
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
            return None, None
        
        obj_transf_list = [
            np.stack([obj_transf_all[frame] for frame in frame_indices if frame in obj_transf_all])
            for obj_transf_all in obj_transf_all_list
        ]

        # Store mesh path info
        raw_data['mesh_path'] = '$'.join(mesh_paths)
        
        return obj_transf_list, obj_names, mesh_paths


# TacoProcessor
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
                        'which_dataset': 'TACO',
                        'which_sequence': f'{task_name}-{seq_name}',
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
                        'which_dataset': 'TACO',
                        'which_sequence': f'{task_name}-{seq_name}',
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
    
    def _get_hand_info(self, raw_data: Dict[str, Any], side: str, frame_indices: List[int], pre_trans:torch.Tensor=None, device = torch.device('cuda')) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process TACO hand data. The ONLY transformation applied is the pre_trans."""
        hand_pose_dir = raw_data['hand_pose_dir']
        
        # Load hand poses
        if side == 'r':
            hand_pkl = pickle.load(open(os.path.join(hand_pose_dir, "right_hand.pkl"), "rb"))
            hand_beta = pickle.load(open(os.path.join(hand_pose_dir, "right_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().to(device)
        else:
            hand_pkl = pickle.load(open(os.path.join(hand_pose_dir, "left_hand.pkl"), "rb"))
            hand_beta = pickle.load(open(os.path.join(hand_pose_dir, "left_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().to(device)
        
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
        hand_transform = torch.eye(4).to(device).unsqueeze(0).repeat((hand_thetas_raw.shape[0], 1, 1))
        hand_transform[:, :3, :3] = quaternion_to_matrix(hand_thetas_raw[:, 0])
        hand_transform[:, :3, 3] = hand_trans_raw
        hand_thetas_raw[:, 0] = matrix_to_quaternion(hand_transform[:, :3, :3])
        hand_trans_raw = hand_transform[:, :3, 3]

        if side == 'r':
            output = self.manolayer_right(hand_thetas_raw, hand_beta.unsqueeze(0).repeat(hand_thetas_raw.shape[0], 1))
            hand_joints = output.joints + hand_trans_raw[...,None,:]
            hand_thetas, hand_trans, hand_joints = self.optimize_hand_joints(hand_thetas_raw, hand_trans_raw, hand_joints, self.manolayer_right)
        else:
            output = self.manolayer_left(hand_thetas_raw, hand_beta.unsqueeze(0).repeat(hand_thetas_raw.shape[0], 1))
            hand_joints = output.joints + hand_trans_raw[...,None,:]
            hand_thetas, hand_trans, hand_joints = self.optimize_hand_joints(hand_thetas_raw, hand_trans_raw, hand_joints, self.manolayer_left)

        return hand_trans[frame_indices], hand_thetas[frame_indices], hand_joints[frame_indices]
    

    def _get_object_info(self, raw_data: Dict[str, Any], frame_indices: List[int], pre_trans:torch.Tensor=None, device = torch.device('cuda')) -> Tuple[List[torch.Tensor], List[str]]:
        """Load TACO object data"""
        obj_name = raw_data['obj_name']
        obj_poses = raw_data['obj_poses']
        
        # Load object mesh
        obj_path = os.path.join(self.obj_dir, obj_name + "_cm.obj")

        # Apply coordinate transformation
        if pre_trans is None:
            pre_trans = torch.eye(4).to(device)
        obj_poses_transformed = pre_trans @ torch.from_numpy(obj_poses).to(device).float()
        obj_transf_subset = obj_poses_transformed[frame_indices]

        # Store mesh path
        raw_data['mesh_path'] = obj_path

        return [obj_transf_subset], [obj_name], [obj_path]        
        
    def _get_task_description(self, raw_data: Dict[str, Any]) -> str:
        """Get task description for TACO."""
        verb = raw_data['action_verb']
        obj_name = raw_data['obj_real_name']
        return f"{verb} the {obj_name}"


# DexYCBProcessor
@DatasetRegistry.register('dexycb')
class DexYCBProcessor(BaseDatasetProcessor):
    def _setup_paths(self):
        self.dataset = DexYCBVideoDataset(self.root_path)
    
    def _get_data_list(self):
        return range(len(self.dataset))
    
    def _apply_transformation_pt(self, points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """Apply transformation to points."""
        return apply_transformation_pt(points, transform)

    def _load_sequence_data(self, data_item):
        seq_data =  self.dataset[data_item]

        start_frame = 0
        for i in range(0, len(seq_data['hand_pose'])):
            hand_pose_frame = seq_data['hand_pose'][i]
            if np.abs(hand_pose_frame).sum() > 1e-5:
                start_frame = i
                break

        return [{
                'which_dataset': 'DexYCB',
                'which_sequence': seq_data['capture_name'],
                'side': 'r',
                'obj_names': YCB_CLASSES[seq_data['ycb_ids'][0]],
                'frame_indices': list(range(0, len(seq_data['hand_pose'])-start_frame, self.task_interval)),
                'hand_pose': seq_data['hand_pose'][start_frame:],
                'object_pose': seq_data['object_pose'][start_frame:],
                'extrinsics': seq_data['extrinsics'],
                'object_mesh_file': seq_data['object_mesh_file'][0],
                'description': "",
                # 'extra_info': {'extrinsics': seq_data['extrinsics']},
                'anno': "",
                'l_valid': False,
                'r_valid': True
            }]

    def _get_hand_info(self, raw_data: Dict[str, Any], side: str, frame_indices: List[int], pre_trans: torch.Tensor = None, device = torch.device('cuda')) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get hand tsl and coeffs information."""
        hand_pose_frame = raw_data['hand_pose'][frame_indices]
        p = torch.from_numpy(hand_pose_frame[..., : 48].astype(np.float32))

        from manopth.manolayer import ManoLayer as _ManoLayer
        xx = _ManoLayer(
            flat_hand_mean=False,
            ncomps=45,
            side="left" if side == 'l' else 'right',
            use_pca=True,
        )
        p[...,3:] = p[...,3:] @ xx.th_selected_comps + xx.th_hands_mean
        t_offset = torch.stack([xx(pp)[1].cpu()[0] * 0.001 for pp in p], axis=0).numpy()[:,0,:]

        p = p.reshape(-1, 16, 3)  # Reshape to (T, 16, 3)
        h_coeffs = axis_angle_to_quaternion(p)
        t = torch.from_numpy(hand_pose_frame[..., 48:51].astype(np.float32)).reshape(-1, 3) + t_offset

        yup2xup = np.linalg.inv(raw_data['extrinsics'])
        yup2xup = torch.from_numpy(yup2xup).float()
        # yup2xup = torch.eye(4)
        
        # Transform hand pose
        hand_transformation = torch.eye(4).unsqueeze(0).repeat((p.shape[0], 1, 1))
        hand_rotation_quat = h_coeffs[:, 0]
        hand_transformation[:, :3, :3] = quaternion_to_matrix(hand_rotation_quat)
        hand_transformation[:, :3, 3] = t
        hand_transformation = yup2xup @ hand_transformation
        h_coeffs[:, 0] = matrix_to_quaternion(hand_transformation[:, :3, :3])
        h_tsl = hand_transformation[:, :3, 3]

        # h_coeffs[:, 0] = matrix_to_quaternion(yup2xup[:3, :3] @ hand_transformation[:, :3, :3])
        # h_tsl = self._apply_transformation_pt(t, yup2xup)

        return h_tsl, h_coeffs
    
    def _get_task_description(self, raw_data):
        return ""

    def _get_object_info(self, raw_data: Dict[str, Any], frame_indices: List[int]) -> Tuple[List[torch.Tensor], List[str]]:
        """Get object transformation and mesh path."""
        object_pose = raw_data['object_pose'][frame_indices]
        # extrinsic_mat = raw_data['extrinsics']
        
        # pose_vec = pt.pq_from_transform(extrinsic_mat)
        # camera_pose = torch.eye(4)
        # camera_pose[:3, :3] = quaternion_to_matrix(torch.from_numpy(pose_vec[3:7]))
        # camera_pose[:3, 3] = torch.from_numpy(pose_vec[0:3])
        # camera_pose = camera_pose.numpy()
        # camera_pose = np.linalg.inv(camera_pose)
        camera_pose = np.linalg.inv(raw_data['extrinsics'])
    
        # Apply extrinsics transformation to object poses
        obj_transf = []
        for i in range(object_pose.shape[0]):
            pose = torch.from_numpy(object_pose[i])
            transformation = torch.eye(4)
            transformation[:3, :3] = quaternion_to_matrix(torch.cat((pose[3:4], pose[:3])))
            transformation[:3, 3] = pose[4:]
            obj_transf.append(camera_pose @ transformation.numpy())
        obj_transf = np.stack(obj_transf, axis=0)
        yup2xup = self._apply_coordinate_transform(raw_data['side']).T
        obj_poses_transformed = yup2xup.cpu().numpy().astype(np.float32) @ obj_transf

        # Get object names
        object_names = raw_data['obj_names']
        # Get mesh path
        mesh_path = raw_data['object_mesh_file']

        return [obj_poses_transformed], [object_names], [mesh_path]


# Dataset configurations
DATASET_CONFIGS = {
    'oakinkv2': {
        'processor_name': 'oakinkv2',
        'root_path': ORIGIN_DATA_PATH['Oakinkv2'],
        'save_path': HUMAN_SEQ_PATH['Oakinkv2'],
        'task_interval': 20,
        'which_dataset': 'Oakinkv2',
        'seq_data_name': 'feature',
        'sequence_indices': list([20,30,40,50])  # Example sequence indices for processing
    },
    
    'taco': {
        'processor_name': 'taco',
        'root_path': ORIGIN_DATA_PATH['Taco'],
        'save_path': HUMAN_SEQ_PATH['Taco'],
        'task_interval': 1,
        'which_dataset': 'Taco',
        'seq_data_name': 'feature',
        'sequence_indices': None  # Example sequence indices for processing
    },

    'dexycb': {
        'processor_name': 'dexycb',
        'root_path': ORIGIN_DATA_PATH['DexYCB'],
        'save_path': HUMAN_SEQ_PATH['DexYCB'],
        'task_interval': 1,
        'which_dataset': 'DexYCB',
        'seq_data_name': 'feature',
        'sequence_indices': list([20,30,40,50])  # Example sequence indices for 
    }
}

def setup_logging(level=logging.ERROR):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler()
        ]
    )

def process_single_dataset(dataset_name: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Process a single dataset.
    
    Args:
        dataset_name: Name of the dataset ('oakinkv2', 'taco', etc.)
        **kwargs: Additional parameters to override default config
    
    Returns:
        List of processed sequences
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name].copy()
    config.update(kwargs)
    
    processor_class = DatasetRegistry.get_processor(config['processor_name'])
    processor = processor_class(**config)
    
    return processor.run()

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

def process_multiple_datasets(
    dataset_names: List[str], 
    configs: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Process multiple datasets and combine them.
    
    Args:
        dataset_names: List of dataset names to process
        configs: Optional list of config overrides for each dataset
    
    Returns:
        Combined list of processed sequences
    """
    if configs is None:
        configs = [{}] * len(dataset_names)
    
    if len(configs) != len(dataset_names):
        raise ValueError("configs length must match dataset_names length")
    
    processor_configs = []
    for name, config_override in zip(dataset_names, configs):
        if name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {name}")
        
        config = DATASET_CONFIGS[name].copy()
        config.update(config_override)
        processor_configs.append(config)
    
    return load_multiple_datasets(processor_configs)

def show_human_statistics(human_data: List[HumanSequenceData]):
    
    # separate data from different datasets
    dataset_data = {}
    for data in human_data:
        dataset_name = data.which_dataset
        if dataset_name not in dataset_data:
            dataset_data[dataset_name] = []
        dataset_data[dataset_name].append(data)
    num_datasets = len(dataset_data)

    ### 1 - seq_len in each dataset
    print(f"Number of datasets: {num_datasets}")
    print(f"Total sequences: {len(human_data)}")
    for dataset_name, data_list in dataset_data.items():
        seq_lens = [len(d.frame_indices) if hasattr(d, 'frame_indices') else 0 for d in data_list]
        print(f"Dataset {dataset_name} has {len(data_list)} sequences")
        print(f"  Mean sequence length: {np.mean(seq_lens):.2f}, Std: {np.std(seq_lens):.2f}")
    
    GRAPH = True
    if GRAPH:
        fig, axes = plt.subplots(1, num_datasets + 1, figsize=(6 * (num_datasets + 1), 6), squeeze=False)
        all_seq_lens = []
        for idx, (dataset_name, data_list) in enumerate(dataset_data.items()):
            seq_lens = [len(d.frame_indices) if hasattr(d, 'frame_indices') else 0 for d in data_list]
            # seq_lens = [d.seq_len for d in data_list]
            all_seq_lens.extend(seq_lens)
            ax = axes[0, idx]
            ax.hist(seq_lens, bins=30, alpha=0.7)
            ax.set_title(f'Sequence Length Distribution in {dataset_name}')
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('Count')
            ax.grid(True)
        # Add subplot for all datasets combined
        ax = axes[0, -1]
        ax.hist(all_seq_lens, bins=30, alpha=0.7, color='gray')
        ax.set_title('Sequence Length Distribution (All Datasets)')
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Count')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    ### 2 & 3 - objects number and sequence length in each dataset (bar chart with subplot)
    fig, axes = plt.subplots(1, num_datasets * 2, figsize=(3 * (num_datasets * 2), 20), squeeze=False)

    for idx, (dataset_name, data_list) in enumerate(dataset_data.items()):

        # Flatten all object names for this dataset
        object_names = []
        object_seq_lens = {}
        for d in data_list:
            objs = d.object_names
            object_names.extend(objs)
            # Collect sequence lengths for each object
            for obj in objs:
                if obj not in object_seq_lens:
                    object_seq_lens[obj] = []
                object_seq_lens[obj].append(len(d.frame_indices))
        
        # Count occurrences
        obj_counter = Counter(object_names)
        ax = axes[0, 2 * idx]
        obj_keys = list(obj_counter.keys())
        obj_vals = list(obj_counter.values())
        ax.barh(obj_keys, obj_vals, alpha=0.7)
        ax.set_title(f'Object Distribution in {dataset_name}', fontsize=10)
        ax.set_ylabel('Object Name', fontsize=10)
        ax.set_xlabel('Count', fontsize=10)
        ax.grid(True, axis='x')
        ax.tick_params(axis='both', which='major', labelsize=8)

        # Add subplot for mean and std of sequence lengths per object (order matches previous subplot)
        means = [np.mean(object_seq_lens[obj]) for obj in obj_keys]
        stds = [np.std(object_seq_lens[obj]) for obj in obj_keys]
        ax = axes[0, 2 * idx + 1]
        ax.barh(obj_keys, means, xerr=stds, alpha=0.7, color='lightgreen')
        ax.set_title('Mean and Std of Sequence Lengths per Object', fontsize=10)
        ax.set_xlabel('Mean Sequence Length', fontsize=10)
        ax.set_ylabel('Object Name', fontsize=10)
        ax.grid(True, axis='x')
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    plt.show()

    print()

def check_data_correctness_by_vis(human_data: List[HumanSequenceData]):
    """
    Check data correctness by visualizing a few sequences.
    
    Args:
        human_data: List of HumanSequenceData objects
    """
    
    dataset_data = {}
    for data in human_data:
        dataset_name = data.which_dataset
        if dataset_name not in dataset_data:
            dataset_data[dataset_name] = []
        dataset_data[dataset_name].append(data)

    for dataset_name, data_list in dataset_data.items():
        print(f"Dataset {dataset_name} has {len(data_list)} sequences")
        # Sample a few sequences for visualization
        # sampled_data = random.sample(data_list, 2)
        sampled_data = data_list
        for d in sampled_data:
            print(f"Visualizing sequence {d.which_sequence}")
            visualize_human_sequence(d, f'/home/qianxu/Desktop/Project/DexPose/dataset/vis_results/{d.which_dataset}_{d.which_sequence}.html')

if __name__ == "__main__":

    # dataset_names = ['dexycb',  'oakinkv2']
    dataset_names = ['taco']
    processed_data = []
    
    GENERATE = True
    if GENERATE:
        processed_data = process_multiple_datasets(dataset_names)
    else:
        for dataset_name in dataset_names:
            file_path = os.path.join(DATASET_CONFIGS[dataset_name]['save_path'],f'seq_{DATASET_CONFIGS[dataset_name]["seq_data_name"]}_{DATASET_CONFIGS[dataset_name]["task_interval"]}.p')
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                processed_data.extend(data)

    # show_human_statistics(processed_data)
    
    # check_data_correctness_by_vis(processed_data)