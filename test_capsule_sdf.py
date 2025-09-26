import torch
import numpy as np
import open3d as o3d
import trimesh
from torchsdf import compute_sdf, index_vertices_by_faces
import plotly.graph_objects as go

def fit_capsule_to_points(points: torch.Tensor, padding_factor: float = 1.05):
    """
    Fits an arbitrarily oriented capsule to a point cloud using PCA.
    Returns dict with *tensor* fields so gradient flow is preserved.
    """
    if points.shape[0] < 3:
        center = points.mean(dim=0, keepdim=True) if points.shape[0] > 0 else torch.zeros(1, 3, device=points.device)
        return {'start': center, 'end': center, 'radius': torch.tensor(0.01, device=points.device)}

    mean = points.mean(dim=0)
    centered = points - mean
    _, _, Vh = torch.linalg.svd(centered)
    direction = Vh.T[:, 0]                      # PCA 主方向

    projections = centered @ direction
    min_proj, max_proj = projections.min(), projections.max()
    p1 = mean + min_proj * direction
    p2 = mean + max_proj * direction

    line_vec = p2 - p1
    line_len_sq = torch.dot(line_vec, line_vec)

    if line_len_sq < 1e-8:                      # 退化→球
        radius = (points - p1).norm(dim=1).max()
    else:
        t = ((points - p1) @ line_vec).clamp(0, 1)
        closest = p1 + t.unsqueeze(1) * line_vec
        radius = (points - closest).norm(dim=1).max()

    return {'start': p1.unsqueeze(0),
            'end': p2.unsqueeze(0),
            'radius': radius * padding_factor} 

def _rotation_matrix_between_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the rotation matrix that rotates unit vector 'a' to unit vector 'b'.

    This implementation is based on Rodrigues' rotation formula and is robust
    to parallel and anti-parallel vectors.

    Args:
        a (torch.Tensor): The starting unit vector (shape: 3).
        b (torch.Tensor): The target unit vector (shape: 3).

    Returns:
        torch.Tensor: The 3x3 rotation matrix that performs the transformation.
    """
    # Ensure inputs are on the same device and are normalized
    a = torch.nn.functional.normalize(a, dim=0)
    b = torch.nn.functional.normalize(b, dim=0)
    
    # The axis of rotation is the cross product of the two vectors
    v = torch.cross(a, b)
    
    # The sine of the angle is the norm of the cross product
    s = torch.linalg.norm(v)
    
    # The cosine of the angle is the dot product
    c = torch.dot(a, b)
    
    # --- Handle Edge Cases ---
    
    # If the vectors are nearly parallel (cross product is close to zero)
    if s < 1e-8:
        # If they are pointing in the same direction (cosine is 1)
        if c > 0:
            # The rotation is the identity matrix
            return torch.eye(3, device=a.device, dtype=a.dtype)
        # If they are pointing in opposite directions (cosine is -1)
        else:
            # The rotation is 180 degrees. A simple way to represent this
            # is with a matrix that inverts the vector.
            # A more general 180-degree rotation can be found, but -I is a valid one.
            # For a more stable 180-degree rotation matrix that works for any axis:
            # Find an arbitrary perpendicular vector 'u'
            u = torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype)
            if torch.abs(torch.dot(a, u)) > 0.99:
                 u = torch.tensor([0.0, 1.0, 0.0], device=a.device, dtype=a.dtype)
            u = torch.nn.functional.normalize(torch.cross(a, u), dim=0)
            # Rodrigues' formula for 180 degrees simplifies to 2*u*u^T - I
            return 2 * torch.outer(u, u) - torch.eye(3, device=a.device, dtype=a.dtype)

    # --- Standard Case (Rodrigues' Rotation Formula) ---
    
    # Skew-symmetric cross-product matrix of v
    vx = torch.tensor([[ 0.0, -v[2],  v[1]],
                       [ v[2],  0.0, -v[0]],
                       [-v[1],  v[0],  0.0]], device=a.device, dtype=a.dtype)
                       
    # Rodrigues' formula: R = I + sin(θ)*K + (1-cos(θ))*K^2
    # Where K is the skew-symmetric matrix of the *unit* axis k = v/s.
    # A more direct formula using v, s, and c is:
    # R = I + vx + vx^2 * ( (1-c) / s^2 )
    
    R = torch.eye(3, device=a.device, dtype=a.dtype) + vx + (vx @ vx) * ((1 - c) / (s**2))
    
    return R

def create_watertight_capsule_trimesh(capsule_params, sections: int = 16) -> trimesh.Trimesh:
    """
    Creates a guaranteed watertight capsule mesh using Trimesh's boolean union operations.
    The capsule is oriented along the Z-axis and centered at the origin.

    Args:
        sections (int): The number of sections for the cylinder and spheres.

    Returns:
        trimesh.Trimesh: A single, watertight Trimesh object representing the capsule.
    """
    # 1. Create the three primitive components as separate Trimesh objects
    # Create a cylinder. Ensure it has caps for the boolean operation to work.
    radius = capsule_params['radius'].squeeze()
    p_start = capsule_params['start'].squeeze()
    p_end = capsule_params['end'].squeeze()
    height = torch.linalg.norm(p_end - p_start).item()
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)

    # Create two spheres for the end caps
    sphere1 = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere2 = sphere1.copy()

    # 2. Position the spheres correctly at the ends of the cylinder
    sphere1.apply_translation([0, 0, height / 2])
    sphere2.apply_translation([0, 0, -height / 2])
    
    # 3. Perform a boolean union of the three parts.
    #    Trimesh's boolean operations are robust and will correctly
    #    remove all internal faces and stitch the geometry together,
    #    resulting in a watertight mesh.
    
    # Union the cylinder with the first sphere
    capsule = trimesh.boolean.union([cylinder, sphere1])
    
    # Union the result with the second sphere
    capsule = trimesh.boolean.union([capsule, sphere2])
    
    # The result of a boolean operation is typically watertight, but a final
    # process() call can clean up any minor artifacts.
    capsule.process()
    capsule_mesh_local_z = capsule
    
    # 3. Calculate the transformation to align and position the capsule
    transform_matrix = torch.eye(4)
    if height > 1e-6:
        # Find rotation
        z_axis = torch.tensor([0.0, 0.0, 1.0])
        target_axis = torch.nn.functional.normalize(p_end - p_start, dim=0)
        rotation_matrix = _rotation_matrix_between_vectors(z_axis, target_axis)
        transform_matrix[:3, :3] = rotation_matrix
    
    # Find translation (center of the capsule)
    center_translation = (p_start + p_end) / 2.0
    transform_matrix[:3, 3] = center_translation

    # 4. Apply the transform to the local Z-aligned capsule
    capsule_mesh_local_z.apply_transform(transform_matrix.cpu().numpy())
    
    # 5. Store the final collision geometry
    c_verts = torch.from_numpy(capsule_mesh_local_z.vertices).to(dtype=torch.float)
    c_faces = torch.from_numpy(capsule_mesh_local_z.faces).to(dtype=torch.long)
    return c_verts, c_faces

def sdf_capsule_analytical_torch(
    query_points: torch.Tensor, capsule_params: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the analytical Signed Distance Field (SDF) for an arbitrarily oriented capsule.
    This function is fully differentiable with respect to all inputs.

    Args:
        query_points (torch.Tensor): A tensor of points to query, shape (N, 3).
        p_start (torch.Tensor): The start point of the capsule's core segment, shape (1, 3) or (3,).
        p_end (torch.Tensor): The end point of the capsule's core segment, shape (1, 3) or (3,).
        radius (torch.Tensor): The radius of the capsule, a scalar tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - sdf_values (torch.Tensor): The signed distance for each query point.
              Convention: Negative inside, positive outside. Shape (N,).
            - signs (torch.Tensor): The sign of the distance for each query point.
              -1 for inside, +1 for outside. Shape (N,).
    """
    p_start = capsule_params['start'].squeeze()
    p_end = capsule_params['end'].squeeze()
    radius = capsule_params['radius'].squeeze()
    line_vec = p_end - p_start
    line_len_sq = torch.dot(line_vec, line_vec)
    
    # Project each query point onto the infinite line defined by p_start and line_vec.
    # The parameter 't' represents the normalized position along the line.
    # t = dot(query - start, line_vec) / dot(line_vec, line_vec)
    t = torch.matmul(query_points - p_start, line_vec) / line_len_sq
    
    # Clamp 't' to the range [0, 1] to find the closest point on the *segment*.
    t_clamped = torch.clamp(t, 0.0, 1.0)
    
    # Calculate the closest point on the line segment for each query point.
    # This uses broadcasting: (1, 3) + (N, 1) * (1, 3) -> (N, 3)
    closest_points_on_line = p_start.unsqueeze(0) + t_clamped.unsqueeze(1) * line_vec.unsqueeze(0)
    
    # The distance from each query point to the line segment is the norm of the difference vector.
    dist_to_segment = torch.linalg.norm(query_points - closest_points_on_line, dim=-1)
    
    # The final SDF is the distance to the segment, minus the radius.
    sdf_values = dist_to_segment - radius
    
    # The sign is simply the sign of the SDF value.
    signs = torch.sign(sdf_values)
    
    return sdf_values

def sdf_capsule_analytical_batch_torch(
    query_points: torch.Tensor, capsule_params: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the analytical Signed Distance Field (SDF) for an arbitrarily oriented capsule,
    batched for query_points. This function is fully differentiable with respect to all inputs.

    Args:
        query_points (torch.Tensor): A tensor of points to query, shape (N, 3).
        capsule_params (dict): A dictionary containing capsule parameters.
            - 'start' (torch.Tensor): The start point of the capsule's core segment, shape (1, 3) or (3,).
            - 'end' (torch.Tensor): The end point of the capsule's core segment, shape (1, 3) or (3,).
            - 'radius' (torch.Tensor): The radius of the capsule, a scalar tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - sdf_values (torch.Tensor): The signed distance for each query point.
              Convention: Negative inside, positive outside. Shape (N,).
            - signs (torch.Tensor): The sign of the distance for each query point.
              -1 for inside, +1 for outside. Shape (N,).
    """
    p_start = capsule_params['start'].squeeze()  # Shape (3,)
    p_end = capsule_params['end'].squeeze()      # Shape (3,)
    radius = capsule_params['radius'].squeeze()  # Shape ()

    # Ensure p_start and p_end are broadcastable with query_points
    # p_start, p_end will be (1, 3) for broadcasting
    p_start_unsqueeze = p_start.unsqueeze(0)
    p_end_unsqueeze = p_end.unsqueeze(0)

    line_vec = p_end_unsqueeze - p_start_unsqueeze  # Shape (1, 3)
    line_len_sq = torch.sum(line_vec * line_vec, dim=-1, keepdim=True) # Shape (1, 1)

    # Project each query point onto the infinite line defined by p_start and line_vec.
    # t = dot(query - start, line_vec) / dot(line_vec, line_vec)
    # (N, 3) - (1, 3) -> (N, 3)
    # (N, 3) @ (3, 1) -> (N, 1) if line_vec was (3,1), but it's (1,3)
    # Using element-wise product and sum for dot product with broadcasting
    t = torch.sum((query_points - p_start_unsqueeze) * line_vec, dim=-1, keepdim=True) / line_len_sq # Shape (N, 1)

    # Clamp 't' to the range [0, 1] to find the closest point on the *segment*.
    t_clamped = torch.clamp(t, 0.0, 1.0) # Shape (N, 1)

    # Calculate the closest point on the line segment for each query point.
    # (1, 3) + (N, 1) * (1, 3) -> (N, 3)
    closest_points_on_line = p_start_unsqueeze + t_clamped * line_vec # Shape (N, 3)

    # The distance from each query point to the line segment is the norm of the difference vector.
    # (N, 3) - (N, 3) -> (N, 3)
    # torch.linalg.norm(..., dim=-1) -> (N,)
    dist_to_segment = torch.linalg.norm(query_points - closest_points_on_line, dim=-1) # Shape (N,)

    # The final SDF is the distance to the segment, minus the radius.
    # (N,) - () -> (N,) (radius broadcasts)
    sdf_values = dist_to_segment - radius

    # The sign is simply the sign of the SDF value.
    signs = torch.sign(sdf_values) # Shape (N,)

    return sdf_values

if __name__ == "__main__":

    # 1. Fit a capsule to the finger segment's visual vertices
    data = []

    link_vertices = torch.cat([(torch.rand(100, 1) - 0.5) * 0.05, (torch.rand(100, 1) - 0.5) * 0.005, (torch.rand(100, 1) - 0.5) * 0.01], dim=1)
    capsule_params = fit_capsule_to_points(link_vertices)

    # 2. Create a watertight, Z-axis aligned capsule mesh using our helper
    c_verts, c_faces = create_watertight_capsule_trimesh(capsule_params, sections=16)
    c_face_verts = index_vertices_by_faces(c_verts, c_faces)
    v_numpy = c_verts.numpy()
    f_numpy = c_faces.numpy()
    mesh_item = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(v_numpy),
            triangles=o3d.utility.Vector3iVector(f_numpy)
        )
    verts = np.asarray(mesh_item.vertices)
    faces = np.asarray(mesh_item.triangles if hasattr(mesh_item, "triangles") else mesh_item.faces)
    data.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        color="#914E02", opacity=0.5, showlegend=True))

    rrange = 0.1
    interval = 0.01
    grid_range = np.arange(-rrange, rrange+interval, interval)
    grid_x, grid_y, grid_z = np.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
    grid_points_local_np = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    grid_points_local_torch = torch.from_numpy(grid_points_local_np).to(dtype=torch.float)
    grid_points_local_torch = grid_points_local_torch.expand(10, -1, -1)
    grid_points_local_torch = grid_points_local_torch.contiguous()

    sdf_values = sdf_capsule_analytical_batch_torch(grid_points_local_torch, capsule_params)
    sdf_values = sdf_values[3]

    # from torchsdf import compute_sdf
    # dist_sq, signs, _, _ = compute_sdf(grid_points_local_torch.to('cuda'), c_face_verts.to('cuda'))
    # sdf_values = torch.sqrt(dist_sq.clamp(min=1e-8)) * (signs)
    
    sdf_values = sdf_values.detach().cpu().numpy()

    # 4. Normalize the SDF values to a [0, 1] opacity scale
    opacities = np.zeros_like(sdf_values)
    positive_mask = sdf_values > 0
    negative_mask = sdf_values < 0
    # Normalize positive values: The largest positive SDF value will have opacity 1.
    if np.any(positive_mask):
        positive_sdfs = sdf_values[positive_mask]
        max_pos_sdf = np.max(positive_sdfs)
        if max_pos_sdf > 1e-6: # Avoid division by zero
            opacities[positive_mask] = (positive_sdfs / max_pos_sdf) ** 3
    # Normalize negative values: The most negative SDF value will have opacity 1.
    if np.any(negative_mask):
        negative_sdfs = sdf_values[negative_mask]
        # The "largest" negative value is the minimum value. Its absolute is the max distance inside.
        max_abs_neg_sdf = np.abs(np.min(negative_sdfs))
        if max_abs_neg_sdf > 1e-6: # Avoid division by zero
            opacities[negative_mask] = (np.abs(negative_sdfs) / max_abs_neg_sdf) ** 3

    # 5. Plot the points with their calculated colors and opacities
    # Plot positive (outside) points as blue
    if np.any(positive_mask):
        rgba_colors_positive = [
            f"rgba(0, 0, 255, {op:.8f})"  for op in opacities[positive_mask]
        ]
        data.append(go.Scatter3d(
            x=grid_points_local_np[positive_mask, 0],
            y=grid_points_local_np[positive_mask, 1],
            z=grid_points_local_np[positive_mask, 2],
            mode='markers',
            marker=dict(
                # Pass the list of RGBA strings to the color property
                color=rgba_colors_positive,
                size=3
            ),
            name='SDF (Outside)'
        ))

    # Plot negative (inside) points as red with variable opacity
    if np.any(negative_mask):
        rgba_colors_negative = [
            f"rgba(255, 0, 0, {op:.8f})"  for op in opacities[negative_mask]
        ]
        data.append(go.Scatter3d(
            x=grid_points_local_np[negative_mask, 0],
            y=grid_points_local_np[negative_mask, 1],
            z=grid_points_local_np[negative_mask, 2],
            mode='markers',
            marker=dict(
                # Pass the list of RGBA strings to the color property
                color=rgba_colors_negative,
                size=3
            ),
            name='SDF (Inside)'
        ))
    
    # --- NEW: Calculate and Visualize Face Normals ---
    # Your face_verts_world is shape (B, F, 3, 3). Since qpos was 1D, B=1.
    # We can squeeze it to (F, 3, 3) for easier handling.
    faces = c_face_verts.squeeze(0) # Shape: (F, 3, 3)

    # Calculate face normals using the cross product of two edges
    v1 = faces[:, 1, :] - faces[:, 0, :] # Edge P0 -> P1
    v2 = faces[:, 2, :] - faces[:, 0, :] # Edge P0 -> P2
    face_normals = torch.cross(v1, v2, dim=1)
    
    # Normalize to get unit vectors
    norms = torch.linalg.norm(face_normals, dim=1, keepdim=True)
    unit_normals = face_normals / (norms + 1e-8)

    # Calculate the center of each face (centroid) to be the starting point of the normal vector
    face_centers = torch.mean(faces, dim=1)

    # Define the length of the visualized normal vectors
    normal_length_scale = 0.005
    end_points = face_centers + unit_normals * normal_length_scale

    # Convert to NumPy for plotting
    centers_np = face_centers.detach().cpu().numpy()
    ends_np = end_points.detach().cpu().numpy()

    # Prepare coordinate lists for a single efficient Scatter3d trace
    lines_x, lines_y, lines_z = [], [], []
    for i in range(len(centers_np)):
        lines_x.extend([centers_np[i, 0], ends_np[i, 0], None])
        lines_y.extend([centers_np[i, 1], ends_np[i, 1], None])
        lines_z.extend([centers_np[i, 2], ends_np[i, 2], None])

    data.append(go.Scatter3d(
        x=lines_x,
        y=lines_y,
        z=lines_z,
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Face Normals',
    ))
    
    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'  # Ensures equal scaling
        ),
        title='3D SDF Visualization with Face Normals'
    )
    fig = go.Figure(data=data, layout=layout)
    fig.write_html("sdf_capsule_analytical.html")