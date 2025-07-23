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


