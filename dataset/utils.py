import torch

# visualize a HumanSequenceData

# visualize a DexSequenceData

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

