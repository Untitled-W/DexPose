import open3d as o3d
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os

def visualize(mesh: o3d.geometry.TriangleMesh, contact_points, filename=None):
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
        name='faces'
    )
    data.append(face_trace)


    # --------- 非流形顶点 ---------
    if contact_points is not None:
        if len(contact_points) == 0:
            nm_v = pd.DataFrame([[np.nan, np.nan, np.nan]], columns=['x', 'y', 'z'])
        else:
            nm_v = pd.DataFrame(np.asarray(contact_points), columns=['x', 'y', 'z'])
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

def ray_from_y(mesh: o3d.geometry.TriangleMesh, x: list, z: list) -> np.ndarray:
    """
    该函数从Y+方向向mesh打光（光线方向为Y-），
    在X和Z坐标构成的网格上发射光线，计算光线与面片的第一个交点。

    Args:
        mesh (o3d.geometry.TriangleMesh): 输入的3D网格模型。
        x (list): 一个包含三个元素的列表 [x_start, x_end, x_count]，
                  定义X坐标的起始值、结束值和列数。x_count应为整数。
        z (list): 一个包含三个元素的列表 [z_start, z_end, z_count]，
                  定义Z坐标的起始值、结束值和列数。z_count应为整数。

    Returns:
        np.ndarray: 一个 (N, 3) 的NumPy数组，包含所有有效的交点。
                    如果没有光线击中mesh，则返回空数组。
    """
    # 1. 确保mesh有数据
    if not mesh.has_vertices() or not mesh.has_triangles():
        print("Warning: Input mesh has no vertices or triangles. Returning empty array.")
        return np.array([])

    # 2. 确定光线起点Y坐标
    # 为了确保所有光线都从mesh上方开始，我们找到mesh的Y轴最大边界，并在此基础上增加一个偏移量。
    max_bound = mesh.get_max_bound()
    ray_start_y = max_bound[1] + 0.1  # 在mesh最高点之上0.1单位，确保从上方发射

    # 3. 生成X和Z坐标网格
    # 使用np.linspace在指定的X和Z范围内生成均匀分布的坐标点
    x_coords = np.linspace(x[0], x[1], int(x[2]))
    z_coords = np.linspace(z[0], z[1], int(z[2]))

    # 使用meshgrid生成所有(x, z)组合，构成一个二维网格
    X_grid, Z_grid = np.meshgrid(x_coords, z_coords)
    
    # 将网格展平，形成光线原点的X和Z分量
    grid_x_origins = X_grid.flatten()
    grid_z_origins = Z_grid.flatten()

    num_rays = len(grid_x_origins) # 光线总数

    # 4. 构造光线原点和方向
    # 创建一个用于存储所有光线原点的NumPy数组
    origins = np.zeros((num_rays, 3), dtype=np.float32)
    origins[:, 0] = grid_x_origins       # 设置X坐标
    origins[:, 1] = ray_start_y          # 设置Y坐标（所有光线起点Y值相同）
    origins[:, 2] = grid_z_origins       # 设置Z坐标

    # 创建一个用于存储所有光线方向的NumPy数组
    directions = np.zeros((num_rays, 3), dtype=np.float32)
    directions[:, 1] = -1.0 # 光线方向沿Y轴负方向 (从Y+向Y-)

    # 5. 使用Open3D的RaycastingScene进行高效光线投射
    scene = o3d.t.geometry.RaycastingScene()
    
    # 将Open3D的传统mesh对象转换为Open3D Tensor mesh对象，并添加到光线投射场景中
    # RaycastingScene需要t.geometry类型的mesh
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh_t) # 将mesh添加到场景中进行光线求交

    # 将光线原点和方向组合成一个 (N, 6) 的Open3D Tensor，这是cast_rays函数所需的格式
    # 前三列是原点，后三列是方向
    rays = o3d.core.Tensor(np.concatenate((origins, directions), axis=1), dtype=o3d.core.Dtype.Float32)

    # 执行光线投射，intersections是一个字典，包含各种交点信息
    intersections = scene.cast_rays(rays)
    
    t_hit_values = intersections['t_hit'].numpy()
    hit_mask = np.isfinite(t_hit_values)

    # 筛选出击中mesh的光线的原点、方向和t_hit值
    hit_origins = origins[hit_mask]
    hit_directions = directions[hit_mask]
    valid_t_hits = t_hit_values[hit_mask]

    # 根据公式 P = O + t * D 计算交点位置
    # P = hit_origins + valid_t_hits[:, np.newaxis] * hit_directions
    # np.newaxis 用于将 valid_t_hits 变成 (N, 1) 形状，以便广播乘法
    valid_hit_positions = hit_origins + valid_t_hits[:, np.newaxis] * hit_directions

    return valid_hit_positions

def ray_from_x_neg(mesh: o3d.geometry.TriangleMesh, y: list, z: list) -> np.ndarray:
    """
    该函数从X-方向向mesh打光（光线方向为X+），
    在Y和Z坐标构成的网格上发射光线，计算光线与面片的第一个交点。

    Args:
        mesh (o3d.geometry.TriangleMesh): 输入的3D网格模型。
        y (list): 一个包含三个元素的列表 [y_start, y_end, y_count]，
                  定义Y坐标的起始值、结束值和列数。y_count应为整数。
        z (list): 一个包含三个元素的列表 [z_start, z_end, z_count]，
                  定义Z坐标的起始值、结束值和列数。z_count应为整数。

    Returns:
        np.ndarray: 一个 (N, 3) 的NumPy数组，包含所有有效的交点。
                    如果没有光线击中mesh，则返回空数组。
    """
    # 1. 确保mesh有数据
    if not mesh.has_vertices() or not mesh.has_triangles():
        print("Warning: Input mesh has no vertices or triangles. Returning empty array.")
        return np.array([])

    # 2. 确定光线起点X坐标
    # 为了确保所有光线都从mesh右侧（X-方向）开始，我们找到mesh的X轴最小边界，并在此基础上增加一个偏移量。
    max_bound = mesh.get_min_bound()
    ray_start_x = max_bound[0] - 0.1  # 在mesh最左侧X-之上0.1单位，确保从X-方向发射

    # 3. 生成Y和Z坐标网格
    # 使用np.linspace在指定的Y和Z范围内生成均匀分布的坐标点
    y_coords = np.linspace(y[0], y[1], int(y[2]))
    z_coords = np.linspace(z[0], z[1], int(z[2]))

    # 使用meshgrid生成所有(y, z)组合，构成一个二维网格
    Y_grid, Z_grid = np.meshgrid(y_coords, z_coords)
    
    # 将网格展平，形成光线原点的Y和Z分量
    grid_y_origins = Y_grid.flatten()
    grid_z_origins = Z_grid.flatten()

    num_rays = len(grid_y_origins) # 光线总数

    # 4. 构造光线原点和方向
    # 创建一个用于存储所有光线原点的NumPy数组
    origins = np.zeros((num_rays, 3), dtype=np.float32)
    origins[:, 0] = ray_start_x          # 设置X坐标（所有光线起点X值相同）
    origins[:, 1] = grid_y_origins       # 设置Y坐标
    origins[:, 2] = grid_z_origins       # 设置Z坐标

    # 创建一个用于存储所有光线方向的NumPy数组
    directions = np.zeros((num_rays, 3), dtype=np.float32)
    directions[:, 0] = 1.0 # 光线方向沿X轴正方向 (从X-向X+)

    # 5. 使用Open3D的RaycastingScene进行高效光线投射
    scene = o3d.t.geometry.RaycastingScene()
    
    # 将Open3D的传统mesh对象转换为Open3D Tensor mesh对象，并添加到光线投射场景中
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh_t) 

    # 将光线原点和方向组合成一个 (N, 6) 的Open3D Tensor
    rays = o3d.core.Tensor(np.concatenate((origins, directions), axis=1), dtype=o3d.core.Dtype.Float32)

    # 执行光线投射
    intersections = scene.cast_rays(rays)
    
    t_hit_values = intersections['t_hit'].numpy()
    hit_mask = np.isfinite(t_hit_values)

    # 筛选出击中mesh的光线的原点、方向和t_hit值
    hit_origins = origins[hit_mask]
    hit_directions = directions[hit_mask]
    valid_t_hits = t_hit_values[hit_mask]

    # 根据公式 P = O + t * D 计算交点位置
    # np.newaxis 用于将 valid_t_hits 变成 (N, 1) 形状，以便广播乘法
    valid_hit_positions = hit_origins + valid_t_hits[:, np.newaxis] * hit_directions

    return valid_hit_positions

def ray_from_x(mesh: o3d.geometry.TriangleMesh, y: list, z: list) -> np.ndarray:
    """
    该函数从X-方向向mesh打光（光线方向为X+），
    在Y和Z坐标构成的网格上发射光线，计算光线与面片的第一个交点。

    Args:
        mesh (o3d.geometry.TriangleMesh): 输入的3D网格模型。
        y (list): 一个包含三个元素的列表 [y_start, y_end, y_count]，
                  定义Y坐标的起始值、结束值和列数。y_count应为整数。
        z (list): 一个包含三个元素的列表 [z_start, z_end, z_count]，
                  定义Z坐标的起始值、结束值和列数。z_count应为整数。

    Returns:
        np.ndarray: 一个 (N, 3) 的NumPy数组，包含所有有效的交点。
                    如果没有光线击中mesh，则返回空数组。
    """
    # 1. 确保mesh有数据
    if not mesh.has_vertices() or not mesh.has_triangles():
        print("Warning: Input mesh has no vertices or triangles. Returning empty array.")
        return np.array([])

    # 2. 确定光线起点X坐标
    # 为了确保所有光线都从mesh右侧（X-方向）开始，我们找到mesh的X轴最小边界，并在此基础上增加一个偏移量。
    max_bound = mesh.get_max_bound()
    ray_start_x = max_bound[0] + 0.1  # 在mesh最左侧X-之上0.1单位，确保从X-方向发射

    # 3. 生成Y和Z坐标网格
    # 使用np.linspace在指定的Y和Z范围内生成均匀分布的坐标点
    y_coords = np.linspace(y[0], y[1], int(y[2]))
    z_coords = np.linspace(z[0], z[1], int(z[2]))

    # 使用meshgrid生成所有(y, z)组合，构成一个二维网格
    Y_grid, Z_grid = np.meshgrid(y_coords, z_coords)
    
    # 将网格展平，形成光线原点的Y和Z分量
    grid_y_origins = Y_grid.flatten()
    grid_z_origins = Z_grid.flatten()

    num_rays = len(grid_y_origins) # 光线总数

    # 4. 构造光线原点和方向
    # 创建一个用于存储所有光线原点的NumPy数组
    origins = np.zeros((num_rays, 3), dtype=np.float32)
    origins[:, 0] = ray_start_x          # 设置X坐标（所有光线起点X值相同）
    origins[:, 1] = grid_y_origins       # 设置Y坐标
    origins[:, 2] = grid_z_origins       # 设置Z坐标

    # 创建一个用于存储所有光线方向的NumPy数组
    directions = np.zeros((num_rays, 3), dtype=np.float32)
    directions[:, 0] = -1.0 # 光线方向沿X轴正方向 (从X-向X+)

    # 5. 使用Open3D的RaycastingScene进行高效光线投射
    scene = o3d.t.geometry.RaycastingScene()
    
    # 将Open3D的传统mesh对象转换为Open3D Tensor mesh对象，并添加到光线投射场景中
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh_t) 

    # 将光线原点和方向组合成一个 (N, 6) 的Open3D Tensor
    rays = o3d.core.Tensor(np.concatenate((origins, directions), axis=1), dtype=o3d.core.Dtype.Float32)

    # 执行光线投射
    intersections = scene.cast_rays(rays)
    
    t_hit_values = intersections['t_hit'].numpy()
    hit_mask = np.isfinite(t_hit_values)

    # 筛选出击中mesh的光线的原点、方向和t_hit值
    hit_origins = origins[hit_mask]
    hit_directions = directions[hit_mask]
    valid_t_hits = t_hit_values[hit_mask]

    # 根据公式 P = O + t * D 计算交点位置
    # np.newaxis 用于将 valid_t_hits 变成 (N, 1) 形状，以便广播乘法
    valid_hit_positions = hit_origins + valid_t_hits[:, np.newaxis] * hit_directions

    return valid_hit_positions


if __name__ == "__main__":

    ### Schunk Hand
    # obj_root = r"E:\2025Codes\DexProjects\DexPose_0919\thirdparty\dex-retargeting\assets\robots\hands\schunk_svh_hand\meshes\visual"

    ### Inspire Hand
    # obj_root = r"E:\2025Codes\DexProjects\DexPose_0919\thirdparty\dex-retargeting\assets\robots\hands\inspire_hand\meshes\collision"

    ### Allgero Hand
    obj_root = r"E:\2025Codes\DexProjects\DexPose_0919\thirdparty\dex-retargeting\assets\robots\hands\allegro_hand\meshes\visual"

    obj_files = [
        ### Schunk Hand
        #     ["d13.obj",[0.008, 0.024, 4],[-0.004,0.004, 3],ray_from_y],
        #     ["finger_tip.obj",[0.008,0.020,4],[-0.004,0.004,3],ray_from_y]
        ### Inspire Hand
        # ["right_thumb_distal.obj",[-0.002, 0.019, 4],[-0.01,-0.002, 3],ray_from_y],
        # ["right_index_intermediate.obj",[0.019,0.041,6],[-0.007,-0.002,2],ray_from_x_neg],
        # ["right_middle_intermediate.obj",[0.022,0.043,6],[-0.007,-0.002,2],ray_from_x_neg],
        # ["right_pinky_intermediate.obj",[0.012,0.034,6],[-0.007,-0.002,2],ray_from_x_neg],
        ### Allgero Hand3
        # ["link_tip.obj", [-0.002,0.008,3],[-0.004,0.004,3],ray_from_x]
    ]

    fig = go.Figure()

    for (obj_file,arg1,arg2,ff) in obj_files:
        obj_path = os.path.join(obj_root, obj_file)
        mesh = o3d.io.read_triangle_mesh(obj_path)
        mesh.compute_vertex_normals()

        points = ff(mesh, arg1, arg2)
        print(obj_file)
        for pp in points:
            print(f"\t\t[{pp[0]:.6f}, {pp[1]:.6f}, {pp[2]:.6f}],")

        visualize(mesh, points)

