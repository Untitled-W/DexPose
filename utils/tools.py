import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from PIL import Image
import open3d as o3d
from pytorch3d.io import load_obj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_cluster import fps
from sklearn.cluster import DBSCAN
from pytorch3d.transforms import (
    quaternion_to_matrix, 
    matrix_to_quaternion, 
    axis_angle_to_quaternion, 
    quaternion_to_axis_angle
)
from manopth.manolayer import ManoLayer

from dataset.base_structure import HumanSequenceData, DexSequenceData

def apply_transformation_pt(points: torch.Tensor, transformation_matrix: torch.Tensor):
    """Apply transformation matrix to points.
    
    Args:
        points: N X 3 or T X N X 3
        transformation_matrix: T x 4 X 4 or 4 X 4
    """
    # points: N X 3, transformation_matrix: T x 4 X 4
    if transformation_matrix.dim() == 4:
        points = torch.cat([points, torch.ones((*points.shape[:-1], 1)).to(points.device)], dim=-1)
        if points.dim() == 3:
            points = points.unsqueeze(-3)
        transformation_points = torch.matmul(transformation_matrix, points.transpose(-1, -2)).transpose(-1, -2)[..., :3]
        return transformation_points
    if points.dim() == 2 and transformation_matrix.dim() == 2:
        points = torch.cat([points, torch.ones((points.shape[0], 1)).to(points.device)], dim=1)
        transformation_points = torch.matmul(transformation_matrix, points.transpose(0, 1)).transpose(0, 1)[:, :3]
        return transformation_points
    if points.dim() == 2:
        points = torch.cat([points, torch.ones((points.shape[0], 1)).to(points.device)], dim=1)
        points = points[None, ...] # 1 X N X 4
    else:
        points = torch.cat([points, torch.ones((points.shape[0], points.shape[1], 1)).to(points.device)], dim=2)
    transformed_points = torch.matmul(transformation_matrix, points.transpose(1, 2)).transpose(1, 2)[:, :, :3] # T X N X 3
    return transformed_points

def cosine_similarity(a, b):
    if len(a) > 30000:
        return cosine_similarity_batch(a, b, batch_size=30000)
    dot_product = torch.mm(a, b.t())
    norm_a = torch.norm(a, dim=1, keepdim=True)
    norm_b = torch.norm(b, dim=1, keepdim=True)
    similarity = dot_product / (norm_a * norm_b.t())

    return similarity


def cosine_similarity_batch(a, b, batch_size=30000):
    num_a, dim_a = a.size()
    num_b, dim_b = b.size()
    similarity_matrix = torch.empty(num_a, num_b, device="cpu")
    for i in tqdm(range(0, num_a, batch_size)):
        a_batch = a[i:i+batch_size]
        for j in range(0, num_b, batch_size):
            b_batch = b[j:j+batch_size]
            dot_product = torch.mm(a_batch, b_batch.t())
            norm_a = torch.norm(a_batch, dim=1, keepdim=True)
            norm_b = torch.norm(b_batch, dim=1, keepdim=True)
            similarity_batch = dot_product / (norm_a * norm_b.t())
            similarity_matrix[i:i+batch_size, j:j+batch_size] = similarity_batch.cpu()
    return similarity_matrix



def farthest_point_sampling(pos: torch.FloatTensor, n_sampling: int):
    bz, N = pos.size(0), pos.size(1)
    feat_dim = pos.size(-1)
    device = pos.device
    sampling_ratio = float(n_sampling / N)
    pos_float = pos.float()

    batch = torch.arange(bz, dtype=torch.long).view(bz, 1).to(device)
    mult_one = torch.ones((N,), dtype=torch.long).view(1, N).to(device)

    batch = batch * mult_one
    batch = batch.view(-1)
    pos_float = pos_float.contiguous().view(-1, feat_dim).contiguous() # (bz x N, 3)
    # sampling_ratio = torch.tensor([sampling_ratio for _ in range(bz)], dtype=torch.float).to(device)
    # batch = torch.zeros((N, ), dtype=torch.long, device=device)
    sampled_idx = fps(pos_float, batch, ratio=sampling_ratio, random_start=True)
    # shape of sampled_idx?
    return sampled_idx


def pt_transform(points, transformation):
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation @ points_homogeneous.T).T
    return transformed_points[:, :3]


def get_point_clouds_from_dexycb(data: dict):
    ycb_mesh_files = data["object_mesh_file"] # a list of .obj file path
    meshes = []
    for mesh_file in ycb_mesh_files:
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        meshes.append(mesh)
    
    original_pc = [np.asarray(mesh.vertices) for mesh in meshes if mesh is not None]

    original_pc_ls = [
            farthest_point_sampling(torch.from_numpy(points).unsqueeze(0), 1000)[:1000]  for points in original_pc
        ]
    pc_ds = [pc[pc_idx] for pc, pc_idx in zip(original_pc, original_pc_ls)]

    return pc_ds[0] # only 1 object in the dataset


def get_point_clouds_from_human_data(seq_data: HumanSequenceData, ds_num=1000):
    obj_mesh = []
    for mesh_path in seq_data.object_mesh_path:
        obj_mesh.append(o3d.io.read_triangle_mesh(mesh_path))
    original_pc = [np.asarray(mesh.vertices) for mesh in obj_mesh if mesh is not None]

    original_pc_ls = [
            farthest_point_sampling(torch.from_numpy(points).unsqueeze(0), ds_num)[:ds_num]  for points in original_pc
        ]
    pc_ds = [pc[pc_idx] for pc, pc_idx in zip(original_pc, original_pc_ls)]

    if seq_data.which_dataset == 'TACO':
        for i in range(len(pc_ds)):
            pc_ds[i] *= 0.01

    return pc_ds


def apply_transformation_human_data(points: List[torch.tensor], transformation: torch.tensor) -> torch.tensor:
    obj_pc = [] # should be (T, N, 3) of len k
    for pc, obj_trans in zip(points, transformation):
        t_frame_pc = []
        for t_trans in obj_trans:
            t_frame_pc.append(pt_transform(pc, t_trans.cpu().numpy()))
        obj_pc.append(np.array(t_frame_pc))
    pc_ls = []
    for t in range(transformation.shape[1]):
        pc_ls.append(np.concatenate([pc[t] for pc in obj_pc], axis=0))
    return pc_ls


def extract_hand_points_and_mesh(hand_tsls, hand_coeffs, side):
    if side == 0:
        mano_layer = ManoLayer(center_idx=0, side='left', use_pca=False).cuda()
    else:
        mano_layer = ManoLayer(center_idx=0, side='right', use_pca=False).cuda()


    hand_verts, hand_joints = mano_layer(quaternion_to_axis_angle(hand_coeffs.to('cuda')).reshape(-1, 48)) # manopth use axis_angle and should be (B, 48)
    hand_joints = hand_joints.cpu().numpy() * 0.001 # if using manotorch, you don't need that!
    hand_verts = hand_verts.cpu().numpy() * 0.001
    hand_joints += hand_tsls.cpu().numpy()[...,None,:]

    return hand_joints, hand_verts


def compute_hand_geometry(hand_pose_frame, mano_layer):
    # pose parameters all zero, no hand is detected
    if np.abs(hand_pose_frame).sum() < 1e-5:
        return None
    p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
    t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
    vertex, joint = mano_layer(p, t)
    vertex = vertex.cpu().numpy()[0]
    joint = joint.cpu().numpy()[0]

    return joint