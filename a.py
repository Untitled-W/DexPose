import torch
import numpy as np
import open3d as o3d
from torchsdf import compute_sdf

# =========================================================================
# 1. HELPER FUNCTIONS
# =========================================================================

import open3d as o3d
import numpy as np

def create_capsule_mesh_o3d(radius: float = 0.5,
                            height: float = 2.0,
                            resolution: int = 32) -> o3d.geometry.TriangleMesh:
    """
    生成 **水密** 胶囊体（Z 轴对称，原点居中）
    1. 圆柱
    2. 上下半球（切除 z<0 或 z>0 的冗余部分）
    3. 顶点级合并 + 移除退化和重复元素
    """
    # ---------- 1. 圆柱 ----------
    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=height, resolution=resolution)
    # Open3D 圆柱默认中心在原点，Z 轴对称，无需平移

    # ---------- 2. 上半球（只保留 z >= 0） ----------
    sp_top = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=resolution)
    # 切掉下半球
    top_v = np.asarray(sp_top.vertices)
    top_f = np.asarray(sp_top.triangles)
    mask = top_v[:, 2] >= 0.0
    top_v = top_v[mask]
    remap = np.full(mask.shape[0], -1, dtype=int)
    remap[mask] = np.arange(mask.sum())
    top_f = remap[top_f[(mask[top_f]).all(axis=1)]]
    hemisphere_top = o3d.geometry.TriangleMesh()
    hemisphere_top.vertices = o3d.utility.Vector3dVector(top_v)
    hemisphere_top.triangles = o3d.utility.Vector3iVector(top_f)
    hemisphere_top.translate((0, 0, height / 2))  # 移到圆柱顶端

    # ---------- 3. 下半球（只保留 z <= 0） ----------
    sp_bot = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=resolution)
    bot_v = np.asarray(sp_bot.vertices)
    bot_f = np.asarray(sp_bot.triangles)
    mask = bot_v[:, 2] <= 0.0
    bot_v = bot_v[mask]
    remap = np.full(mask.shape[0], -1, dtype=int)
    remap[mask] = np.arange(mask.sum())
    bot_f = remap[bot_f[(mask[bot_f]).all(axis=1)]]
    hemisphere_bottom = o3d.geometry.TriangleMesh()
    hemisphere_bottom.vertices = o3d.utility.Vector3dVector(bot_v)
    hemisphere_bottom.triangles = o3d.utility.Vector3iVector(bot_f)
    hemisphere_bottom.translate((0, 0, -height / 2))  # 移到圆柱底端

    # ---------- 4. 合并并保证水密 ----------
    capsule = cyl + hemisphere_top + hemisphere_bottom
    capsule.merge_close_vertices(eps=1e-6)
    capsule.remove_duplicated_triangles()
    capsule.remove_duplicated_vertices()
    capsule.remove_non_manifold_edges()
    capsule.remove_degenerate_triangles()
    return capsule


def sdf_capsule_analytical(points: torch.Tensor, radius: float, height: float) -> torch.Tensor:
    """
    Calculates the true, analytical SDF for a capsule centered at the origin
    and aligned with the Z-axis.
    
    Convention: Negative inside, positive outside.
    """
    h = height / 2.0
    # Points' xy coordinates
    p_xy = points[:, :2]
    # Points' z coordinate
    p_z = points[:, 2]
    
    # Clamp the z-coordinate to the line segment of the capsule's core [-h, h]
    c = torch.clamp(p_z, -h, h)
    
    # Calculate the distance from each point to the closest point on the line segment
    # This is the distance in the XY plane + the distance along Z from the clamped point
    dist_to_segment = torch.sqrt(torch.sum(p_xy**2, dim=1) + (p_z - c)**2)
    
    # The final SDF is the distance to the segment, minus the radius
    return dist_to_segment - radius

def sdf_capsule_analytical_torch(
    query_points: torch.Tensor, 
    p_start: torch.Tensor, 
    p_end: torch.Tensor, 
    radius: torch.Tensor
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
    p_start = p_start.squeeze()
    p_end = p_end.squeeze()
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


# =========================================================================
# 2. MAIN VERIFICATION LOGIC
# =========================================================================

if __name__ == '__main__':
    # --- Step 1: Define a simple capsule for our test ---
    CAPSULE_RADIUS = 0.1
    CYLINDER_HEIGHT = 0.4
    
    # The capsule's core is a line segment on the Z-axis from -0.2 to +0.2.
    # The spherical caps extend from z=-0.3 to -0.2 and z=0.2 to 0.3.
    
    print(f"--- Testing SDF with a capsule: Radius={CAPSULE_RADIUS}, Cylinder Height={CYLINDER_HEIGHT} ---")
    
    # --- Step 2: Create the Open3D mesh for the capsule ---
    capsule_mesh = create_capsule_mesh_o3d(CAPSULE_RADIUS, CYLINDER_HEIGHT)
    
    # Get vertices and faces as NumPy arrays
    verts_np = np.asarray(capsule_mesh.vertices)
    faces_np = np.asarray(capsule_mesh.triangles)

    # `compute_sdf` needs the faces represented as a list of vertices (F, 3, 3)
    face_verts_np = verts_np[faces_np]
    
    # Convert to PyTorch tensors for the function call
    face_verts_torch = torch.from_numpy(face_verts_np).to(dtype=torch.float)

    # --- Step 3: Create 10 specific test points ---
    test_points = torch.tensor([
        # Point, Expected Location, Expected Analytical SDF
        [0.0,    0.0,    0.0],    # 0: Deep inside, at the center.          SDF = -0.1
        [0.05,   0.0,    0.1],    # 1: Inside, cylindrical part.            SDF = -0.05
        [0.0,    0.0,    0.25],   # 2: Inside, top spherical cap.           SDF = -0.05
        [0.1,    0.0,    0.0],    # 3: Exactly on surface, cylindrical part. SDF = 0.0
        [0.0,    0.0,    0.3],    # 4: Exactly on surface, top cap.          SDF = 0.0
        [0.15,   0.0,    0.0],    # 5: Outside, near cylinder.               SDF = +0.05
        [0.0,    0.0,    0.4],    # 6: Outside, above top cap.               SDF = +0.1
        [0.1,    0.0,    0.25],   # 7: Outside, diagonal to top cap.        SDF = sqrt(0.1^2 + 0.05^2) - 0.1 = +0.0118
        [-0.05,  -0.05,  0.0],    # 8: Inside, diagonal in cylinder.        SDF = sqrt(0.05^2 + 0.05^2) - 0.1 = -0.0293
        [0.5,    0.0,    0.0],    # 9: Far outside.                          SDF = +0.4
    ], dtype=torch.float)
    
    # --- Step 4: Run the user's `compute_sdf` interface ---
    # Move data to GPU as the original code does
    points_cuda = test_points.to('cuda')
    face_verts_cuda = face_verts_torch.to('cuda')

    # Call the function under test
    dist_sq, signs, _, _ = compute_sdf(points_cuda, face_verts_cuda)

    # Process the output exactly as in the original code
    # Convention: "inside" should be positive
    sdf_from_compute_sdf = torch.sqrt(dist_sq.clamp(min=1e-8)) * (-signs)

    # --- Step 5: Run the manual, analytical SDF calculation for ground truth ---
    # Convention: "inside" should be negative, so we multiply by -1 at the end
    # sdf_analytical_pytorch = sdf_capsule_analytical(test_points, CAPSULE_RADIUS, CYLINDER_HEIGHT) * -1.0
    sdf_analytical_pytorch = sdf_capsule_analytical_torch(
        test_points, 
        torch.tensor([0.0, 0.0, -CYLINDER_HEIGHT/2]), 
        torch.tensor([0.0, 0.0, CYLINDER_HEIGHT/2]), 
        torch.tensor(CAPSULE_RADIUS)
    ) * -1.0

    # --- Step 6: Compare the results ---
    print("\n--- Comparison of SDF Results ---")
    print(f"{'Index':<6} | {'Point Coordinates':<25} | {'compute_sdf Result':>20} | {'Ground Truth (Manual)':>25}")
    print("-" * 80)
    
    results_compute_sdf_cpu = sdf_from_compute_sdf.cpu().numpy()
    results_analytical_cpu = sdf_analytical_pytorch.cpu().numpy()

    for i in range(len(test_points)):
        pt_str = f"({test_points[i,0]:.2f}, {test_points[i,1]:.2f}, {test_points[i,2]:.2f})"
        res_comp = results_compute_sdf_cpu[i]
        res_anal = results_analytical_cpu[i]
        
        # Check if the signs match
        sign_match = "✅" if (res_comp * res_anal >= 0) or (abs(res_comp) < 1e-5 and abs(res_anal) < 1e-5) else "❌"

        print(f"{i:<6} | {pt_str:<25} | {res_comp:>20.4f} | {res_anal:>25.4f}   {sign_match}")
        
    print("-" * 80)
    print("\nConclusion:")
    if np.allclose(results_compute_sdf_cpu, results_analytical_cpu, atol=1e-3):
        print("✅ The `compute_sdf` function and your processing logic appear to be working correctly!")
    else:
        print("❌ The `compute_sdf` results DO NOT match the ground truth.")
        print("   This strongly suggests that the `signs` tensor returned by `compute_sdf` is incorrect for this mesh,")
        print("   likely because the mesh is non-watertight or has inconsistent face windings, causing the 'inside'/'outside' test to fail.")
        