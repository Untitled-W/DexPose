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
from hdbscan import HDBSCAN
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix

def apply_transformation_pt(points: torch.Tensor, transformation_matrix: torch.Tensor):
    """Apply transformation matrix to points.
    
    Args:
        points: N X 3
        transformation_matrix: T x 4 X 4
    """
    # points: N X 3, transformation_matrix: T x 4 X 4
    if points.dim() == 2:
        points = torch.cat([points, torch.ones((points.shape[0], 1)).to(points.device)], dim=1)
        points = points[None, ...] # 1 X N X 4
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

def intepolate_feature(query_points:torch.Tensor, features:torch.Tensor, points:torch.Tensor) -> torch.Tensor:
    """
    Interpolates features at query points using inverse distance weighting.

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
    """
    使用 DBSCAN 对 (b, n, 3) 数据做批量聚类，返回 (b, n) 的最大簇掩码。
    """
    b, n, _ = data.shape
    mask = torch.zeros((b, n), dtype=torch.bool, device=device)
    
    data_cpu = data.detach().cpu().numpy()

    for i in range(b):
        points = data_cpu[i]  # (n, 3)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(points)
        
        # 排除 -1 噪声
        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            continue
        counts = np.bincount(valid_labels)
        largest_cluster_label = np.argmax(counts)
        largest_cluster_mask = (labels == largest_cluster_label)
        
        mask[i, largest_cluster_mask] = True
    
    return mask

def get_largest_cluster_mask(data, 
                             min_cluster_size=5, 
                             min_samples=None,
                             device='cuda'):
    """
    使用 HDBSCAN 对 (b, n, 3) 的数据进行逐 batch 聚类，
    并返回 (b, n) 的掩码，表示每个点是否在最大簇中。

    参数：
    -------
    data: torch.Tensor, shape = (b, n, 3)
        待聚类的批量数据
    min_cluster_size: int
        HDBSCAN中的最小簇大小
    min_samples: int 或 None
        HDBSCAN中的最小采样点，若None则默认与min_cluster_size相同
    device: str
        CPU/GPU 设备 (若要在 GPU 上跑，需将 data 转到 CPU numpy 再转回)

    返回：
    -------
    mask: torch.BoolTensor, shape = (b, n)
        对应的布尔掩码，True 表示该点在最大簇，False 表示非最大簇或噪声
    """
    b, n, _ = data.shape
    # 初始化 (b, n) 的布尔张量
    mask = torch.zeros((b, n), dtype=torch.bool, device=device)
    
    # 如果数据还在 GPU，就先转到 CPU 做聚类
    # （HDBSCAN/DBSCAN 通常是基于 numpy/scipy 实现的，需要 CPU）
    data_cpu = data.detach().cpu().numpy()  # shape: (b, n, 3)

    for i in range(b):
        points = data_cpu[i]  # shape: (n, 3)

        # 1) 使用 HDBSCAN 聚类
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples)
        
        # fit_predict 返回每个点的标签：-1 表示噪声，>=0 表示簇索引
        labels = clusterer.fit_predict(points)  # shape: (n,)

        # 2) 寻找数量最多的那个簇（排除 -1）
        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            # 全部都是噪声点，没有任何簇
            continue
        
        # 找到出现次数最多的标签
        counts = np.bincount(valid_labels)  # 按标签统计频次
        largest_cluster_label = np.argmax(counts)

        # 3) 构建这个 batch 的掩码：属于最大簇的置 True
        largest_cluster_mask = (labels == largest_cluster_label)

        # 存入最终的 mask (需要先转为 torch，再赋值)
        mask[i, largest_cluster_mask] = True

    return mask

def find_longest_false_substring(bool_tensor: torch.Tensor):
    int_tensor = (~bool_tensor).int()   
    max_length = 0
    start_idx = -1
    end_idx = -1
    current_start = None
    current_length = 0
    
    for i in range(len(int_tensor)):
        if int_tensor[i] == 1:
            if current_start is None:
                current_start = i
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                start_idx = current_start
                end_idx = i

            current_start = None
            current_length = 0
    
    if current_length > max_length:
        max_length = current_length
        start_idx = current_start
        end_idx = len(int_tensor)
    
    return start_idx, end_idx

def get_key_hand_joints(hand_joints:torch.Tensor) -> torch.Tensor:
    """
    Thumb, Index, Middle, Ring, Lttle, Rigid
    
    Parameters:
        hand_joints (... X 21 X 3):
    Returns:
        hand_key_joints (... X 6 X 3):
    """
    Thumb = hand_joints[..., 4, :].unsqueeze(-2)
    Index = hand_joints[..., 8, :].unsqueeze(-2)
    Middle = hand_joints[..., 12, :].unsqueeze(-2)
    Ring = hand_joints[..., 16, :].unsqueeze(-2)
    Little = hand_joints[..., 20, :].unsqueeze(-2)
    Rigid = hand_joints[..., [1, 5, 9, 13, 17], :].mean(dim=-2, keepdim=True)
    hand_key_joints = torch.cat((Thumb, Index, Middle, Ring, Little, Rigid), dim=-2)
    return hand_key_joints

def get_contact_points_pertimestep(hand_key_joints:torch.Tensor, obj_pts:torch.Tensor, 
                       method:str='neighbor', nd_thre:int=0.05):
    """
    Get the contact points
    Args:
        hand_joints (N_h X 3): Hand joints
        obj_pts (N, 3): Object points
        method (str): The method to get the contact points
            - sphere: Use the 6 sphere to get the contact points
            - neighbor: Use the neighbor points to get the contact points
    """
    distance = torch.norm(hand_key_joints[..., None, :] - obj_pts[..., None, :, :], dim=-1, p=2)
    if method == 'sphere':
        contact_threshold = 0.1
        contact_mask = (distance < contact_threshold).sum(-2).bool()
    elif method == 'neighbor':
        min_dis, min_dis_idx = torch.min(distance, dim=-1)
        nearest_neighbor = obj_pts.gather(-2, min_dis_idx[..., None].expand(*min_dis_idx.shape, 3))
        # vis_pc_coor_plotly([hand_key_joints.cpu().numpy(), nearest_neighbor.cpu().numpy()], [obj_pts.cpu().numpy()])
        neighbor_distance = torch.norm(nearest_neighbor[..., None, :] - obj_pts[..., None, :, :], dim=-1)
        contact_mask = (neighbor_distance < nd_thre).sum(-2)
    else:
        raise ValueError(f"Method {method} is not supported")

    return contact_mask


def get_contact_pts(pts:torch.Tensor, feats:torch.Tensor, hk_feats:torch.Tensor, n_pts:int=100, nd_thre:int=0.02) -> torch.Tensor:
    """

    Args:
        pts (torch.Tensor): (N_obj, 3)
        feats (torch.Tensor): (N_obj, F_dim)
        hk_feats (torch.Tensor): (T, N, F_dim)
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
    # from utils.vis_utils import vis_pc_coor_plotly
    # vis_pc_coor_plotly([selected_points[2].cpu().numpy()], [pts.cpu().numpy()], 
    #                 voxel_dict=voxel_dict_example,)
    # print((torch.count_nonzero(masks) / (masks.shape[0] * masks.shape[1])).item())
    for t_idx in range (T):
        sp = selected_points[t_idx]
        mask = masks[t_idx]
        idx_count = get_contact_points_pertimestep(sp[mask], pts,nd_thre=nd_thre)
        # vis_pc_coor_plotly([contact_points.reshape(-1, 3).cpu().numpy()], [pts.cpu().numpy()])
        
        contact_indices = torch.topk(idx_count, min(int(2 * n_pts), torch.count_nonzero(idx_count)), largest=True)[1].to(torch.long)
        contact_d_index = farthest_point_sampling(pts[contact_indices].unsqueeze(0), n_pts)[:n_pts]
        index_ls.append(contact_indices[contact_d_index])
    return torch.stack(index_ls)
    # return torch.stack(index_ls), grid, selected_points_indices

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


def get_point_clouds_from_human_data(seq_data, ds_num=1000):
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