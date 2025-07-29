import torch
import numbers
import numpy as np
import argparse
from typing import List, Dict, Optional, Tuple
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from PIL import Image
from pytorch3d.io import load_obj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys
from manotorch.manolayer import ManoLayer, MANOOutput
from torch_cluster import fps
import copy
from sklearn.cluster import DBSCAN
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy()"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, numbers.Number):
        return np.array(tensor)
    else:
        raise NotImplementedError


def to_tensor(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    elif isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray, numbers.Number):
        return torch.tensor(ndarray)
    else:
        raise NotImplementedError


def convert_trimesh_to_torch_mesh(tm, device, is_tosca=True):
    verts_1, faces_1 = torch.tensor(tm.vertices, dtype=torch.float32), torch.tensor(
        tm.faces, dtype=torch.float32
    )
    if is_tosca:
        verts_1 = verts_1 / 10
    verts_rgb = torch.ones_like(verts_1)[None] * 0.8
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts_1], faces=[faces_1], textures=textures)
    mesh = mesh.to(device)
    return mesh

def convert_mesh_container_to_torch_mesh(tm, device, is_tosca=True):
    verts_1, faces_1 = torch.tensor(tm.vert, dtype=torch.float32), torch.tensor(
        tm.face, dtype=torch.float32
    )
    if is_tosca:
        verts_1 = verts_1 / 10
    verts_rgb = torch.ones_like(verts_1)[None] * 0.8
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts_1], faces=[faces_1], textures=textures)
    mesh = mesh.to(device)
    return mesh

def load_textured_mesh(mesh_path, device):
    verts, faces, aux = load_obj(mesh_path)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_maps = aux.texture_images

    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...]  # (1, H, W, 3)

    # Create a textures object
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

    # Initialise the mesh with textures
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=tex)
    mesh = mesh.to(device)
    return mesh

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


def hungarian_correspondence(similarity_matrix):
    # Convert similarity matrix to a cost matrix by negating the similarity values
    cost_matrix = -similarity_matrix.cpu().numpy()

    # Use the Hungarian algorithm to find the best assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create a binary matrix with 1s at matched indices and 0s elsewhere
    num_rows, num_cols = similarity_matrix.shape
    match_matrix = np.zeros((num_rows, num_cols), dtype=int)
    match_matrix[row_indices, col_indices] = 1
    match_matrix = torch.from_numpy(match_matrix).cuda()
    return match_matrix


def gmm(a, b):
    # Compute Gram matrices
    gram_matrix_a = torch.mm(a, a.t())
    gram_matrix_b = torch.mm(b, b.t())

    # Expand dimensions to facilitate broadcasting
    gram_matrix_a = gram_matrix_a.unsqueeze(1)
    gram_matrix_b = gram_matrix_b.unsqueeze(0)

    # Compute Frobenius norm for each pair of vertices using vectorized operations
    correspondence_matrix = torch.norm(gram_matrix_a - gram_matrix_b, p='fro', dim=2)

    return correspondence_matrix

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
  
  
def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


def get_contact_points(hand_pts:torch.Tensor, obj_pts:torch.Tensor, obj_normals:torch.Tensor=None, n_cpt:int=700,  dist_threshold=0.01) -> torch.Tensor:
    """
    Param:
    hand_pts: torch.Tensor, shape=(..., N, 3)
    obj_pts: torch.Tensor, shape=(..., M, 3)

    Return:
    contact_points: torch.Tensor, shape=(..., n_cpt, 3)
        
        
    """

    # object_normal = data['f4'].reshape(self.window_size, -1, 3).astype(np.float32)
    # (..., N, M)
    start_idx = 1
    dist_hand_joints_to_obj_pc = (hand_pts[..., None, :] - obj_pts[..., None, :, :]).norm(dim=-1)
    # index of object points which is the most nearest points to each hand keypoints
    # (..., N)
    _, minn_dists_joints_obj_idx = torch.min(dist_hand_joints_to_obj_pc, dim=-1) # num_frames x nn_hand_verts 
    # (..., N, 3)
    nearest_obj_pts = batched_index_select(values=obj_pts, indices=minn_dists_joints_obj_idx, dim=start_idx)
    # (..., M, N)
    dist_object_pc_nearest_pts = (obj_pts[..., None, :] - nearest_obj_pts[..., None, :, :]).norm(dim=-1)
    # (..., M)
    dist_object_pc_nearest_pts, _ = torch.min(dist_object_pc_nearest_pts, dim=-1) # nf x nn_obj_pcs
    
    # (M,) the minimum distance between object points and ANY hand keypoints
    for i in range(start_idx):
       dist_object_pc_nearest_pts, _ = torch.min(dist_object_pc_nearest_pts, dim=0) # nn_obj_pcs #
    
    
    # (M) The qualified poitns:
    # Points within the threshold distance to **ANY** hand key pts' nearest obj points in **ANY** time-step
    base_pts_mask = (dist_object_pc_nearest_pts <= dist_threshold)
    # (N_basepts, 3)
    if start_idx == 2:
      base_pts = obj_pts[0][0][base_pts_mask]
    elif start_idx == 1:
      base_pts = obj_pts[0][base_pts_mask]
    else:
      raise ValueError("start_idx should be 1 or 2")
    # # # base_pts_bf_sampling = base_pts.clone()
    # base_normals = object_normal_th[0][base_pts_mask]
    
    # nn_base_pts = self.nn_base_pts
    base_pts_idxes = farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=n_cpt)
    base_pts_idxes = base_pts_idxes[:n_cpt]
    # # if self.debug:
    # #     print(f"base_pts_idxes: {base_pts.size()}, nn_base_sampling: {nn_base_pts}")
    
    # # ### get base points ### # base_pts and base_normals #
    base_pts = obj_pts[..., base_pts_mask, :][..., base_pts_idxes, :] # nn_base_sampling x 3 #
    if obj_normals is not None:
        base_normals = obj_normals[..., base_pts_mask, :][..., base_pts_idxes, :]
        return base_pts, base_normals
    # base_normals = base_normals[base_pts_idxes]
    return base_pts


# Additional utility functions migrated from dataset.py and tools.py


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


def merge_two_parts(verts_list):
    """Merge multiple vertex lists into one tensor."""
    if len(verts_list) == 1:
        return verts_list[0]
    
    verts_num = 0
    merged_verts_list = []
    for p_idx in range(len(verts_list)):
        # part_verts = torch.from_numpy(verts_list[p_idx]) # T X Nv X 3 
        part_verts = verts_list[p_idx] # T X Nv X 3 

        if p_idx == 0:
            merged_verts_list.append(part_verts)
        else:
            merged_verts_list.append(part_verts)

        verts_num += part_verts.shape[1] 

    # merged_verts = torch.cat(merged_verts_list, dim=1).data.cpu().numpy()
    merged_verts = torch.cat(merged_verts_list, dim=-2)

    return merged_verts


def get_key_hand_joints(hand_joints: torch.Tensor) -> torch.Tensor:
    """Extract key hand joints: Thumb, Index, Middle, Ring, Little, Rigid.
    
    Parameters:
        hand_joints (... X 21 X 3):
    Returns:
        hand_key_joints (... X 6 X 3):
    """
    # Thumb = hand_joints[..., 2:5, :].mean(dim=-2, keepdim=True)
    # Index = hand_joints[..., 6:9, :].mean(dim=-2, keepdim=True)
    # Middle = hand_joints[..., 10:13, :].mean(dim=-2, keepdim=True)
    # Ring = hand_joints[..., 14:17, :].mean(dim=-2, keepdim=True)
    # Little = hand_joints[..., 18:21, :].mean(dim=-2, keepdim=True)
    # Rigid = hand_joints[..., [0, 1, 5, 9, 13, 17], :].mean(dim=-2, keepdim=True)
    Thumb = hand_joints[..., 4, :].unsqueeze(-2)
    Index = hand_joints[..., 8, :].unsqueeze(-2)
    Middle = hand_joints[..., 12, :].unsqueeze(-2)
    Ring = hand_joints[..., 16, :].unsqueeze(-2)
    Little = hand_joints[..., 20, :].unsqueeze(-2)
    Rigid = hand_joints[..., [1, 5, 9, 13, 17], :].mean(dim=-2, keepdim=True)
    hand_key_joints = torch.cat((Thumb, Index, Middle, Ring, Little, Rigid), dim=-2)
    return hand_key_joints


def intepolate_feature(query_points: torch.Tensor, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Interpolates features at query points using inverse distance weighting.

    Args:
        query_points: (..., num, 3)
        features: (..., n, feat_dim)
        points: (..., n, 3)
    """
    points_exp = points[..., None, :, :]
    query_points_exp = query_points[..., :, None, :]
    
    dists = torch.norm((points_exp - query_points_exp), dim=-1) # (..., num, n)
    if dists.isnan().any():
        raise ValueError('nan in dists')

    # weights = 1 / (torch.abs(dists) + 1e-10) # (..., num, n)
    weights = 1 / (dists + 1e-5) ** 2 # (..., num, n)
    weights = weights / torch.sum(weights, dim=-1, keepdim=True) # (..., num, n)
    interpolated_features = torch.matmul(weights, features)# (..., num, feat_dim)
    return interpolated_features


def get_largest_cluster_mask_dbscan(data, eps=0.5, min_samples=5, device='cpu'):
    """Get mask for largest cluster using DBSCAN clustering."""
    # Convert to numpy if tensor
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = data
    
    # Reshape for clustering
    original_shape = data_np.shape
    data_reshaped = data_np.reshape(-1, data_np.shape[-1])
    
    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_reshaped)
    labels = clustering.labels_
    
    # Find largest cluster
    unique_labels = np.unique(labels)
    largest_cluster_label = -1
    largest_cluster_size = 0
    
    for label in unique_labels:
        if label != -1:  # Ignore noise points
            cluster_size = np.sum(labels == label)
            if cluster_size > largest_cluster_size:
                largest_cluster_size = cluster_size
                largest_cluster_label = label
    
    # Create mask
    mask = (labels == largest_cluster_label)
    mask = mask.reshape(original_shape[:-1])
    
    if isinstance(data, torch.Tensor):
        mask = torch.tensor(mask, device=device)
    
    return mask


def get_contact_pts_from_whole_field_perstep(pts: torch.Tensor, feats: torch.Tensor, hk_feats: torch.Tensor, n_pts: int = 100, nd_thre: int = 0.02) -> torch.Tensor:
    """Get contact points from whole field per step.

    Args:
        pts (torch.Tensor): (N, 3)
        feats (torch.Tensor): (N, F_dim)
        hk_feats (torch.Tensor): (T, 6, F_dim)
        transforms (torch.Tensor): (T, 4, 4)
        n_pts (int, optional): _description_. Defaults to 100.
        nd_thre (int, optional): _description_. Defaults to 0.02.

    Returns:
        torch.Tensor: _description_
    """
    lx, rx = pts[..., 0].min() - max(abs(pts[..., 0].min() * 0.1), 0.05), pts[..., 0].max() + max(abs(pts[..., 0].max() * 0.1), 0.05)
    ly, ry = pts[..., 1].min() - max(abs(pts[..., 1].min() * 0.1), 0.05), pts[..., 1].max() +  max(abs(pts[..., 1].max() * 0.1), 0.05)
    lz, rz = pts[..., 2].min() - max(abs(pts[..., 2].min() * 0.1), 0.05), pts[..., 2].max() +  max(abs(pts[..., 2].max() * 0.1), 0.05)
    scale = max(rx - lx, ry - ly, rz - lz)
    step = scale / 30

    X, Y, Z = torch.meshgrid(torch.arange(lx, rx+step, step), torch.arange(ly, ry+step, step), torch.arange(lz, rz+step, step), indexing='ij')
    grid = torch.stack((X, Y, Z), dim=0).reshape(3, -1).T.to(pts.device)
    grid_weight = (1 / (torch.norm(pts[None, ...] - grid[:, None, ...], dim=-1) + 1e-6) ** 2) # (n, f1)
    grid_feat = torch.matmul(grid_weight, feats) / grid_weight.sum(dim=-1, keepdims=True) # (n, 768)
    
    T, N, F = hk_feats.shape
    index_ls = []
    selected_pts_ls = []
    selected_pts_indices_ls = []
    for t_idx in range (T):
        all_feat_dis = torch.norm(grid_feat - hk_feats[t_idx, ...].unsqueeze(-2), dim=-1) # (n, )
        sp_ls = []
        sp_idx_ls = []
        for f_idx in range(N):
            feat_dis = all_feat_dis[f_idx]
            top_pts, top_n_idx = torch.topk(feat_dis, 4, largest=False, dim=-1)
            selected_pts = grid[top_n_idx]
            sp_ls.append(selected_pts)
            sp_idx_ls.append(top_n_idx)
    
        selected_pts = torch.concatenate(sp_ls, dim=0)
        selected_pts_ls.append(selected_pts)
        selected_pts_indices_ls.append(torch.concatenate(sp_idx_ls, dim=0))
    
    selected_points = torch.stack(selected_pts_ls)
    selected_points_indices = torch.stack(selected_pts_indices_ls)
    masks = get_largest_cluster_mask_dbscan(selected_points.reshape(1, -1, 3), eps=step.item() * 5 , device=pts.device)
    masks = masks.reshape(T, -1)
    # voxel_dict_example = {
    #                         'grid_centers': grid.cpu().numpy(),
    #                         'selected_points': selected_points[2].cpu().numpy(),
    #                         'vis_empty': True,               # show outlines for non-selected
    #                         'auto_calculate_size': True      # automatically compute (dx, dy, dz)
    #                         # 'voxel_size': (1.0, 1.0, 1.0)  # if you want to fix the size yourself
    #                     }
    # vis_pc_coor_plotly([selected_points[2].cpu().numpy()], [pts.cpu().numpy()], 
    #                 voxel_dict=voxel_dict_example,)
    print((torch.count_nonzero(masks) / (masks.shape[0] * masks.shape[1])).item())
    
    for t_idx in range (T):
        sp = selected_points[t_idx]
        mask = masks[t_idx]
        idx_count = get_contact_points(sp[mask], pts, not_prune=True, return_idx=True, nd_thre=nd_thre)
        # vis_pc_coor_plotly([contact_points.reshape(-1, 3).cpu().numpy()], [pts.cpu().numpy()])
        
        contact_indices = torch.topk(idx_count, min(int(2 * n_pts), torch.count_nonzero(idx_count)), largest=True)[1].to(torch.long)
        contact_d_index = farthest_point_sampling(pts[contact_indices].unsqueeze(0), n_pts)[:n_pts]
        index_ls.append(contact_indices[contact_d_index])
    return torch.stack(index_ls)


def mirror_data(data_dict, side):
    """Mirror data for data augmentation."""
    # Note: This function requires additional imports that may not be available
    # from hand_model_utils import from_hand_rot6d, to_hand_rot6d
    # For now, we'll provide a simplified version that handles basic mirroring
    
    hand_joints_m = data_dict[f'{side}h_joints'].clone()
    hand_joints_m[..., 0] *= -1

    if type(data_dict[f'{side}o_transf']) is list:
        obj_transf_list_m = []
        for obj_transf in data_dict[f'{side}o_transf']:
            # transl_m = obj_transf[:, :3, 3].clone()
            # rot_m = obj_transf[:, :3, :3].clone()
            transl_m = obj_transf[..., :3, 3].clone()
            rot_m = obj_transf[..., :3, :3].clone()

            transl_m[..., 0] *= -1
            rot_m = matrix_to_axis_angle(rot_m)
            rot_m[..., [1, 2]] *= -1
            rot_m = axis_angle_to_matrix(rot_m)

            obj_transf_m = torch.eye(4).to(obj_transf.device).repeat(obj_transf.shape[0], 1, 1)
            # obj_transf_m[:, :3, :3] = rot_m
            # obj_transf_m[:, :3, 3] = transl_m
            obj_transf_m[..., :3, :3] = rot_m
            obj_transf_m[..., :3, 3] = transl_m
            obj_transf_list_m.append(obj_transf_m)
        
        obj_points_list_m = []
        for obj_points in data_dict[f'{side}o_points']:
            obj_points_m = obj_points.clone()
            obj_points_m[..., 0] *= -1
            obj_points_list_m.append(obj_points_m)
        
        obj_normals_list_m = []
        for obj_normals in data_dict[f'{side}o_normals']:
            obj_normals_m = obj_normals.clone()
            obj_normals_m[..., 0] *= -1
            obj_normals_list_m.append(obj_normals_m)

        obj_points_ori_list_m = []
        for obj_points_ori in data_dict[f'{side}o_points_ori']:
            obj_points_ori_m = obj_points_ori.clone()
            obj_points_ori_m[..., 0] *= -1
            obj_points_ori_list_m.append(obj_points_ori_m)
        
        obj_normals_ori_list_m = []
        for obj_normals_ori in data_dict[f'{side}o_normals_ori']:
            obj_normals_ori_m = obj_normals_ori.clone()
            obj_normals_ori_m[..., 0] *= -1
            obj_normals_ori_list_m.append(obj_normals_ori_m)
        
        obj_features_list_m = [obj_features.clone() for obj_features in data_dict[f'{side}o_features']]

    else:
        obj_transf = data_dict[f'{side}o_transf'] # torch.Tensor, shape=(N, T, 4, 4)
        # transl_m = obj_transf[:, :3, 3].clone()
        # rot_m = obj_transf[:, :3, :3].clone()
        transl_m = obj_transf[..., :3, 3].clone() # torch.Tensor, shape=(N, T, 3)
        rot_m = obj_transf[..., :3, :3].clone() # torch.Tensor, shape=(N, T, 3, 3)

        transl_m[..., 0] *= -1
        rot_m = matrix_to_axis_angle(rot_m) # torch.Tensor, shape=(N, T, 3, 3)
        rot_m[..., [1, 2]] *= -1
        rot_m = axis_angle_to_matrix(rot_m) # torch.Tensor, shape=(N, T, 3, 3)

        # fix bug here!
        obj_transf_m = torch.eye(4).to(obj_transf.device).repeat(obj_transf.shape[0], obj_transf.shape[1], 1, 1) # (N, T, 4, 4)
        # obj_transf_m[:, :3, :3] = rot_m
        # obj_transf_m[:, :3, 3] = transl_m
        obj_transf_m[..., :3, :3] = rot_m
        obj_transf_m[..., :3, 3] = transl_m
        obj_transf_list_m = obj_transf_m

        obj_points_m = data_dict[f'{side}o_points'].clone()
        obj_points_m[..., 0] *= -1
        obj_points_list_m = obj_points_m

        obj_normals_m = data_dict[f'{side}o_normals'].clone()
        obj_normals_m[..., 0] *= -1
        obj_normals_list_m = obj_normals_m

        obj_points_ori_m = data_dict[f'{side}o_points_ori'].clone()
        obj_points_ori_m[..., 0] *= -1
        obj_points_ori_list_m = obj_points_ori_m

        obj_normals_ori_m = data_dict[f'{side}o_normals_ori'].clone()
        obj_normals_ori_m[..., 0] *= -1
        obj_normals_ori_list_m = obj_normals_ori_m

        obj_features_list_m = [obj_features.clone() for obj_features in data_dict[f'{side}o_features']]

    # Simplified hand parameter mirroring (commented out complex parts)
    # hand_params: torch.Tensor = data_dict[f'{side}h_params'].clone()
    # hand_params[..., -3] *= -1
    # For now, we'll skip the complex hand parameter transformations
    # hand_rotvec = from_hand_rot6d(hand_params[:, :-3].reshape(hand_params.shape[0], -1, 6), to_rotvec=True)
    # hand_rotvec = hand_rotvec.reshape(hand_params.shape[0], -1, 3)
    # hand_rotvec[..., [1, 2]] *= -1
    # hand_rot6d_m = to_hand_rot6d(hand_rotvec, from_rotvec=True)
    # hand_params_m = torch.cat((hand_rot6d_m.flatten(-2), hand_params[..., -3:]), dim=-1)

    mesh_norm_transformation = data_dict['mesh_norm_transformation'].clone()
    mesh_norm_transformation[0] *= -1

    side_m = 'l' if side == 'r' else 'r'

    data_dict_m = {
        f'{side_m}h_joints': hand_joints_m,
        # f'{side_m}h_params': hand_params_m,  # Commented out due to complex dependencies
        
        f'{side_m}o_points': obj_points_list_m,
        f'{side_m}o_points_ori': obj_points_ori_list_m,
        f'{side_m}o_normals': obj_normals_list_m,
        f'{side_m}o_normals_ori': obj_normals_ori_list_m,
        f'{side_m}o_features': obj_features_list_m,
        f'{side_m}o_transf': obj_transf_list_m,
        f'{side_m}o_num': data_dict[f'{side}o_num'].clone(),
        f'{side_m}o_num_ori': data_dict[f'{side}o_num_ori'].clone(),

        f'{side}h_joints': None,
        f'{side}h_params': None,
        f'{side}o_points': None,
        f'{side}o_points_ori': None,
        f'{side}o_normals': None,
        f'{side}o_normals_ori': None,
        f'{side}o_features': None,
        f'{side}o_transf': None,

        'contact_indices': copy.deepcopy(data_dict['contact_indices']) if 'contact_indices' in data_dict else None,
        'mesh_path': copy.deepcopy(data_dict['mesh_path']) if 'mesh_path' in data_dict else None,
        'mesh_norm_transformation': mesh_norm_transformation,
        'task_desc': copy.deepcopy(data_dict['task_desc']) if 'task_desc' in data_dict else None,
        'side': side_m,
        'seq_len': data_dict['seq_len'] if 'seq_len' in data_dict else None,
        'extra_desc': copy.deepcopy(data_dict['extra_desc']) if 'extra_desc' in data_dict else {},
    }

    return data_dict_m

def apply_transformation_pt(points:torch.Tensor, transformation_matrix:torch.Tensor):
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