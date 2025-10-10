import open3d as o3d
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import trimesh

import os

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
    v = torch.cross(a, b, dim=-1)
    
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


def visualize(mesh: o3d.geometry.TriangleMesh, mesh_pca: o3d.geometry.TriangleMesh, mesh_approx: o3d.geometry.TriangleMesh, surface_points, filename=None):
    """
    Open3D mesh -> [Plotly Mesh3d, Scatter3d(edges)]
    返回 list 可直接 unpack 进 go.Figure(data=[...])
    """

    data = []
    # --------- 面片 ---------
    v = pd.DataFrame(np.asarray(mesh.vertices), columns=['x', 'y', 'z'])
    f = np.asarray(mesh.triangles)
    face_trace = go.Mesh3d(
        x=v['x'], y=v['y'], z=v['z'],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color="#86B8EA",
        flatshading=True,
        opacity=0.9,
        name='faces',
        showlegend=True
    )
    data.append(face_trace)

    v = pd.DataFrame(np.asarray(mesh_pca.vertices), columns=['x', 'y', 'z'])
    f = np.asarray(mesh_pca.triangles)
    approx_face_trace = go.Mesh3d(
        x=v['x'], y=v['y'], z=v['z'],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color="#E5E841",
        flatshading=True,
        opacity=0.3,
        name='approx_faces',
        showlegend=True
    )
    data.append(approx_face_trace)

    v = pd.DataFrame(np.asarray(mesh_approx.vertices), columns=['x', 'y', 'z'])
    f = np.asarray(mesh_approx.triangles)
    approx_face_trace = go.Mesh3d(
        x=v['x'], y=v['y'], z=v['z'],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color="#C286EA",
        flatshading=True,
        opacity=0.3,
        name='approx_faces',
        showlegend=True
    )
    data.append(approx_face_trace)


    # --------- 非流形顶点 ---------
    if surface_points is not None:
        if len(surface_points) == 0:
            nm_v = pd.DataFrame([[np.nan, np.nan, np.nan]], columns=['x', 'y', 'z'])
        else:
            nm_v = pd.DataFrame(np.asarray(surface_points), columns=['x', 'y', 'z'])
        non_manifold_trace = go.Scatter3d(
            x=nm_v['x'], y=nm_v['y'], z=nm_v['z'],
            mode='markers',
            marker=dict(color='red', size=3),
            name='non-manifold vertices',
            hoverinfo='skip'
        )
        data.append(non_manifold_trace)

    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        title=filename,
        showlegend=False
    )
    if filename is not None:
        out_path = os.path.join("output", f"{filename}.html")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.write_html(out_path)
        print(f"Saved visualization to {out_path}")
    else:
        fig.show()


def fit_capsule_to_points_PCA(points: torch.Tensor, padding_factor: float = 1.05):
    """
    Fits an arbitrarily oriented capsule to a point cloud using PCA.
    (This is the same robust function from previous answers)
    """
    if points.shape[0] < 3:
        center = torch.mean(points, dim=0, keepdim=True) if points.shape[0] > 0 else torch.zeros(1, 3, device=points.device)
        return {'start': center, 'end': center, 'radius': 0.01}
    
    mean = torch.mean(points, dim=0)
    centered_points = points - mean
    U, S, Vh = torch.linalg.svd(centered_points)
    direction = Vh.T[:, 0]
    projections = torch.matmul(centered_points, direction)
    min_proj, max_proj = torch.min(projections), torch.max(projections)
    p1 = mean + min_proj * direction
    p2 = mean + max_proj * direction
    
    line_vec = p2 - p1
    line_len_sq = torch.dot(line_vec, line_vec)
    if line_len_sq < 1e-8:
        dists = torch.linalg.norm(points - p1, dim=1)
        radius = torch.max(dists)
    else:
        t = torch.clamp(torch.matmul(points - p1, line_vec) / line_len_sq, 0.0, 1.0)
        closest_points = p1.unsqueeze(0) + t.unsqueeze(1) * line_vec.unsqueeze(0)
        dists = torch.linalg.norm(points - closest_points, dim=1)
        radius = torch.max(dists)

    return {'start': p1.unsqueeze(0), 'end': p2.unsqueeze(0), 'radius': radius * padding_factor}


def distance_point_to_segment_actual(points: torch.Tensor, s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """
    Calculates the actual Euclidean distance from a batch of points to a line segment.

    Args:
        points: A torch.Tensor of shape (N, 3) for N points.
        s: A torch.Tensor of shape (3,) representing the segment's start point.
        e: A torch.Tensor of shape (3,) representing the segment's end point.

    Returns:
        A torch.Tensor of shape (N,) containing the distances.
    """
    v = e - s  # Vector representing the line segment (3,)
    l2 = torch.dot(v, v) # Squared length of the segment

    if l2 < 1e-8:  # If segment is too short, treat it as a single point 's'
        return torch.linalg.norm(points - s, dim=1) # (N,)

    # Calculate 't' parameter: projection of (point - s) onto v, normalized by l2
    # This finds where along the infinite line the point's projection lies.
    t = torch.sum((points - s) * v, dim=1) / l2  # (N,)

    # Clamp 't' to [0, 1] to ensure the closest point is on the segment itself
    t_clamped = torch.clamp(t, 0.0, 1.0)  # (N,)
    
    # Calculate the closest points on the line segment
    closest_points_on_segment = s + t_clamped.unsqueeze(-1) * v  # (N, 3)

    # Return the Euclidean distance from each point to its closest point on the segment
    return torch.linalg.norm(points - closest_points_on_segment, dim=1) # (N,)


def fit_capsule_to_points_optimization(points: torch.Tensor, padding_factor: float = 1.05,
                                      num_iterations: int = 1000, learning_rate: float = 0.01,
                                      lambda_r: float = 1e-4, lambda_len: float = 1e-4, alpha = 1e-3):
    """
    Fits an arbitrarily oriented capsule to a point cloud using an optimization-based method.
    它最小化点到胶囊表面的距离，使得点云尽可能地贴合在胶囊表面。
    包含正则化项以稳定优化过程。

    Args:
        points: 一个形状为 (N, 3) 的 torch.Tensor，表示点云。
        padding_factor: 初始半径的乘法因子。有助于提供一个良好的初始猜测。
        num_iterations: 优化迭代次数。
        learning_rate: Adam 优化器的学习率。
        lambda_r: 半径正则化强度。
        lambda_len: 胶囊轴长正则化强度。

    Returns:
        一个字典，包含:
            'start': torch.Tensor (1, 3) - 优化后的胶囊轴起点。
            'end': torch.Tensor (1, 3) - 优化后的胶囊轴终点。
            'radius': torch.Tensor (1,) - 优化后的胶囊半径。
    """
    # 处理退化情况（少于 3 个点）
    if points.shape[0] < 3:
        center = torch.mean(points, dim=0, keepdim=True) if points.shape[0] > 0 else torch.zeros(1, 3, device=points.device)
        return {'start': center, 'end': center, 'radius': torch.tensor(0.01, device=points.device)}

    # 1. 使用 PCA 方法初始化胶囊参数
    initial_capsule = fit_capsule_to_points_PCA(points, padding_factor=padding_factor)
    
    # 提取参数，移动到正确的设备，并启用梯度跟踪
    # s = torch.zeros_like(initial_capsule['start']).squeeze(0).clone().detach().requires_grad_(True)
    # e = torch.zeros_like(initial_capsule['end']).squeeze(0).clone().detach().requires_grad_(True)
    # r = torch.zeros_like(initial_capsule['radius']).clone().detach().requires_grad_(True)

    s = initial_capsule['start'].squeeze(0).clone().detach().requires_grad_(True)
    e = initial_capsule['end'].squeeze(0).clone().detach().requires_grad_(True)
    r = initial_capsule['radius'].clone().detach().requires_grad_(True)
    
    # 确保半径从一开始就是非负的
    with torch.no_grad():
        r.clamp_min_(0.0)

    # 初始化优化器 (Adam 是一个很好的通用选择)
    optimizer = optim.Adam([s, e, r], lr=learning_rate)
    
    # 初始化学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # 每100步将学习率降低为原来的0.1

    # 2. 优化循环
    for i in range(num_iterations):
        optimizer.zero_grad() # 清除上次迭代的梯度

        # 计算所有点到当前胶囊轴（线段 s-e）的距离
        dists_to_segment = distance_point_to_segment_actual(points, s, e)

        # --- 关键改动在这里：新的损失函数定义 ---
        # 目标是让 dists_to_segment 尽可能等于 r
        # 因此，惩罚 (距离 - 半径) 的平方
        point_loss = torch.sum((dists_to_segment - r)**2) # 标量损失

        # 正则化损失 (依然重要，防止胶囊崩溃或过度膨胀)
        # 这项确保了胶囊会收缩，防止损失在胶囊过大时变为零。
        dists_to_s = torch.linalg.norm(points - s, dim=1)  # Distances from points to start
        dists_to_e = torch.linalg.norm(points - e, dim=1)  # Distances from points to end
        axis_length_sq = torch.min(dists_to_s) + torch.min(dists_to_e)  # Use the minimum distance as loss
        radius_sq = r**2 # 使用半径的平方进行正则化
        
        regularization_loss = lambda_r * radius_sq + lambda_len * axis_length_sq
        
        
        # Calculate distance between current s/e/r and initial_capsule_params
        current_distance_start = torch.linalg.norm(s - initial_capsule['start'].squeeze(0))  # Distance from current start to initial start
        current_distance_end = torch.linalg.norm(e - initial_capsule['end'].squeeze(0))  # Distance from current end to initial end
        current_radius_difference = r - initial_capsule['radius']  # Difference in radius
        regularization_loss += alpha * (current_distance_start + current_distance_end + current_radius_difference**2)
        
        # 总损失 (标量)
        loss = point_loss + regularization_loss

        # 反向传播：计算损失对 s、e、r 的梯度
        loss.backward() # 现在对标量 'loss' 调用 backward()

        # 使用优化器更新参数
        optimizer.step()
        
        # 更新学习率
        # scheduler.step()

        # 每次更新后强制半径非负
        with torch.no_grad(): # 此块内的操作不跟踪梯度
            r.clamp_min_(0.0) # 半径必须非负

        # # 可选：打印进度
        # if (i + 1) % (num_iterations // 10) == 0 or i == 0:
        #     current_axis_len = torch.linalg.norm(e-s).item()
        #     print(f"Iteration {i+1}/{num_iterations}, Total Loss: {loss.item():.6f}, "
        #           f"Point Loss: {point_loss.item():.6f}, Reg Loss: {regularization_loss.item():.6f}, "
        #           f"Radius: {r.item():.4f}, Axis Length: {current_axis_len:.4f}")

    # 返回优化后的胶囊参数，将 start/end 恢复为 (1,3)
    return {'start': s.unsqueeze(0).detach(), 'end': e.unsqueeze(0).detach(), 'radius': r.detach()}


def create_watertight_capsule_trimesh(capsule_params, sections: int = 16) -> trimesh.Trimesh:
    """
    Creates a guaranteed watertight capsule mesh using Trimesh's boolean union operations.
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


if __name__ == "__main__":

    robot_name = 'allegro_hand'

    obj_root = f"E:/2025Codes/DexProjects/DexPose_0919/thirdparty/dex-retargeting/assets/robots/hands/{robot_name}/meshes"

    obj_files = [
        ### Schunk Hand
        # "collision/d13.obj", "collision/finger_tip.obj", "collision/f11.obj", "collision/f21.obj"
        ### Inspire Hand
        # ["right_thumb_distal.obj",[-0.002, 0.019, 4],[-0.01,-0.002, 3],ray_from_y],
        # ["right_index_intermediate.obj",[0.019,0.041,6],[-0.007,-0.002,2],ray_from_x_neg],
        # ["right_middle_intermediate.obj",[0.022,0.043,6],[-0.007,-0.002,2],ray_from_x_neg],
        # ["right_pinky_intermediate.obj",[0.012,0.034,6],[-0.007,-0.002,2],ray_from_x_neg],
        ### Allgero Hand3
        "collision/link_tip.obj", "visual/link_1.0.obj", "visual/link_14.0.obj", "visual/link_15.0.obj"
    ]

    for oo in obj_files:
        mesh_path = os.path.join(obj_root, oo)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = torch.from_numpy(np.asarray(mesh.vertices)).to(dtype=torch.float)
        faces = torch.from_numpy(np.asarray(mesh.triangles)).to(dtype=torch.long)
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=100) # 采样点云
        points = np.asarray(pcd.points)
        
        capsule_params = fit_capsule_to_points_PCA(torch.from_numpy(points).to(dtype=torch.float))
        c_verts_pca, c_faces_pca = create_watertight_capsule_trimesh(capsule_params, sections=16)
        v_numpy_pca = c_verts_pca.detach().cpu().numpy()
        f_numpy_pca = c_faces_pca.detach().cpu().numpy()
        mesh_pca = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(v_numpy_pca),
            triangles=o3d.utility.Vector3iVector(f_numpy_pca)
        )

        capsule_params = fit_capsule_to_points_optimization(torch.from_numpy(points).to(dtype=torch.float))
        c_verts, c_faces = create_watertight_capsule_trimesh(capsule_params, sections=16)
        v_numpy = c_verts.detach().cpu().numpy()
        f_numpy = c_faces.detach().cpu().numpy()
        mesh_approx = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(v_numpy),
            triangles=o3d.utility.Vector3iVector(f_numpy)
        )

        visualize(mesh, mesh_pca, mesh_approx, points)
