import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
import open3d as o3d
from torch_cluster import fps
from pytorch3d.transforms import quaternion_to_axis_angle
import plotly.graph_objects as go
from plotly.colors import get_colorscale
import matplotlib.pyplot as plt

from .hand_model import load_robot


def farthest_point_sampling(pos: torch.FloatTensor, n_sampling: int):
    bz, N = pos.size(0), pos.size(1)
    if N == 0:
        return torch.tensor([], dtype=torch.long, device=pos.device)
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


def get_object_meshes_from_human_data(seq_data: Dict[str, Any]) -> List[o3d.geometry.TriangleMesh]:
    """
    从序列数据中加载对象网格。
    
    特别处理：如果数据集是 'taco'，则将网格缩小1000倍，
    因为TACO数据集的单位通常是毫米（mm），而其他数据集是米（m）。

    Args:
        seq_data: 包含序列信息的字典，应有 "object_mesh_path" 和 "which_dataset" 键。

    Returns:
        一个包含加载并经过适当缩放的 o3d.geometry.TriangleMesh 对象的列表。
    """
    obj_meshes = []
    
    # 检查数据集是否为 'taco'
    is_taco_dataset = (seq_data.get("which_dataset", "").lower() == "taco")

    for mesh_path in seq_data["object_mesh_path"]:
        # 读取网格文件
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        # 仅当网格有效时（有顶点）才进行处理
        if mesh and mesh.has_vertices():
            
            # 如果是 TACO 数据集，则进行缩放
            if is_taco_dataset:
                center_point = np.zeros(3) #!!! 这里是中心点
                mesh.scale(0.01, center=center_point)
            
            obj_meshes.append(mesh)
            
    return obj_meshes


def get_point_clouds_from_human_data(seq_data, ds_num=1000, return_norm=False):
    """
    从序列数据中加载对象网格并提取下采样后的点云。

    Args:
        seq_data (dict): 包含对象网格路径等信息的字典。
        ds_num (int, optional): 下采样后的点云数量。默认为 1000。
        return_norm (bool, optional): 如果为 True，则额外返回与点云对应的法向量。默认为 False。

    Returns:
        list[np.ndarray]: 下采样后的点云列表。
        OR
        tuple[list[np.ndarray], list[np.ndarray]]: 如果 return_norm 为 True，则返回一个元组，
                                                     包含点云列表和法向量列表。
    """
    obj_meshes = []
    for mesh_path in seq_data["object_mesh_path"]:
        # 读取网格文件
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        # 仅当网格有效时（有顶点）才添加
        if mesh and mesh.has_vertices():
            obj_meshes.append(mesh)

    # 提取原始顶点数据
    original_pc = [np.asarray(mesh.vertices, dtype=np.float32) for mesh in obj_meshes]

    original_normals = []
    if return_norm:
        # 如果需要返回法向量，则计算并提取它们
        for mesh in obj_meshes:
            mesh.compute_vertex_normals()  # 确保法向量已计算
            original_normals.append(np.asarray(mesh.vertex_normals, dtype=np.float32))

    # 使用最远点采样（FPS）获取下采样点的索引
    # 注意：FPS 在原始点云上操作，返回的是索引
    original_pc_ls = [
            farthest_point_sampling(torch.from_numpy(points).unsqueeze(0), ds_num)
            for points in original_pc
        ]
    
    # 使用索引从原始点云中选出下采样后的点
    pc_ds = [pc[pc_idx] for pc, pc_idx in zip(original_pc, original_pc_ls)]

    # TACO 数据集的特殊处理：缩放点云
    # 注意：法向量是方向向量，不应被缩放
    if seq_data["which_dataset"].lower() == 'taco':
        for i in range(len(pc_ds)):
            pc_ds[i] *= 0.01

    if return_norm:
        # 如果需要，也使用相同的索引来采样法向量
        normals_ds = [normals[pc_idx] for normals, pc_idx in zip(original_normals, original_pc_ls)]
        return pc_ds, normals_ds
    else:
        # 默认只返回点云
        return pc_ds


def pt_transform(points, transformation):
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation @ points_homogeneous.T).T
    return transformed_points[:, :3]


def norm_transform(normals: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """
    对一组3D法向量（或任何方向向量）应用一个4x4变换矩阵的旋转部分。
    """
    # 提取3x3的旋转矩阵
    rotation_matrix = transformation[:3, :3]
    # 应用旋转
    transformed_normals = normals @ rotation_matrix.T
    
    # (可选但推荐) 重新归一化法向量，以防旋转操作引入浮点误差
    norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
    # 避免除以零
    safe_norms = np.where(norms == 0, 1, norms)
    
    return transformed_normals / safe_norms


def apply_transformation_human_data(
    points: List[torch.Tensor], 
    transformation: torch.Tensor, 
    norm: Optional[List[torch.Tensor]] = None
):
    '''
    对点云列表应用一系列变换。
    如果提供了法向量，则它们也会被相应地变换。

    Args:
        points: k个点云的列表。每个元素是形状为 (N_i, 3) 的张量。
        transformation: 形状为 (k, T, 4, 4) 的变换张量。
        norm: (可选) 与点对应的k个法向量列表。
              每个元素的形状为 (N_i, 3)。默认为 None。

    Returns:
        如果 norm is None:
            返回一个(np.ndarray), (T, N, 3) 。
        如果 norm is not None:
            返回一个元组 (pc_ls, norm_ls)，分别包含变换后的点云和法向量列表。
    '''
    obj_pc = []  # 存储每个对象随时间变换后的点云
    obj_norm = [] if norm is not None else None

    if norm is not None and len(points) != len(norm):
        raise ValueError("点云和法向量的列表长度必须相同。")

    # 遍历每个对象
    for i in range(len(points)):
        pc_np = points[i].cpu().numpy() if type(points[i]) is torch.Tensor else points[i]
        obj_trans_seq = transformation[i]

        t_frame_pc = []
        t_frame_norm = [] if norm is not None else None

        if norm is not None:
            current_norm_np = norm[i].cpu().numpy() if type(norm[i]) is torch.Tensor else norm[i]
        else:
            current_norm_np = None

        # 遍历该对象的时间序列变换
        for t_trans in obj_trans_seq:
            trans_np = t_trans.cpu().numpy()
            
            # 变换点
            t_frame_pc.append(pt_transform(pc_np, trans_np))
            
            # 如果提供了法向量，则变换法向量
            if current_norm_np is not None:
                t_frame_norm.append(norm_transform(current_norm_np, trans_np))

        obj_pc.append(np.array(t_frame_pc, dtype=np.float32))
        if obj_norm is not None:
            obj_norm.append(np.array(t_frame_norm, dtype=np.float32))

    # 对于每个时间帧，拼接所有对象的点云（和法向量）
    pc_ls = []
    norm_ls = [] if obj_norm is not None else None
    
    num_time_frames = transformation.shape[1]
    for t in range(num_time_frames):
        pc_ls.append(np.concatenate([pc[t] for pc in obj_pc], axis=0))
        if norm_ls is not None:
            norm_ls.append(np.concatenate([n[t] for n in obj_norm], axis=0))

    if norm is not None:
        return np.asarray(pc_ls), np.asarray(norm_ls)
    else:
        return np.asarray(pc_ls)


def apply_transformation_on_object_mesh(
    meshes: List[o3d.geometry.TriangleMesh],
    transformation: torch.Tensor
) -> List[List[o3d.geometry.TriangleMesh]]:
    """
    对一系列Open3D网格应用一系列变换。

    Args:
        meshes: k个Open3D网格对象的列表。
        transformation: 形状为 (k, T, 4, 4) 的变换张量，
                      其中 k 是对象的数量，T 是时间帧的数量。

    Returns:
        一个长度为 k 的列表。每个元素是另一个列表，
        包含了该物体在T个时间帧变换后的 o3d.geometry.TriangleMesh 对象。
        例如: result[k][t] 是第 k 个对象在时间 t 的网格。
    """
    num_objects = transformation.shape[0]
    num_time_frames = transformation.shape[1]

    if len(meshes) != num_objects:
        raise ValueError(
            f"网格列表的长度 ({len(meshes)}) 必须与变换张量的第一个维度 ({num_objects}) 相同。"
        )

    # 初始化结果列表，外层代表物体
    # transformed_meshes_per_object[k] 将存储第 k 个对象在所有时间帧的网格
    transformed_meshes_per_object = [[] for _ in range(num_objects)]

    # 遍历每个对象
    for i in range(num_objects):
        # 遍历每个时间帧
        for t in range(num_time_frames):
            # 获取原始网格和对应的变换矩阵
            original_mesh = meshes[i]
            trans_matrix_torch = transformation[i, t]

            # 将 PyTorch 张量转换为 NumPy 数组以供 Open3D 使用
            trans_matrix_np = trans_matrix_torch.cpu().numpy()

            # 创建原始网格的副本以进行变换
            # 这是为了确保原始网格在每个时间步的变换都是基于最原始的状态
            transformed_mesh = o3d.geometry.TriangleMesh()
            transformed_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(original_mesh.vertices))
            transformed_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(original_mesh.triangles))
            
            # 保持法线和颜色信息（如果有的话）
            if original_mesh.has_vertex_normals():
                transformed_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(original_mesh.vertex_normals))
            if original_mesh.has_vertex_colors():
                transformed_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(original_mesh.vertex_colors))

            # 应用变换
            transformed_mesh.transform(trans_matrix_np)

            # 将变换后的网格添加到对应物体的列表中
            transformed_meshes_per_object[i].append(transformed_mesh)

    return transformed_meshes_per_object


def extract_hand_points_and_mesh(hand_tsls, hand_coeffs, side):
    from manotorch.manolayer import ManoLayer

    # it should be true
    CENTER = True

    if CENTER:
        if side == 0:
            mano_layer = ManoLayer(center_idx=0, side='left', use_pca=False).cuda()
        else:
            mano_layer = ManoLayer(center_idx=0, side='right', use_pca=False).cuda()
    else:
        if side == 0:
            mano_layer = ManoLayer(side='left', use_pca=False).cuda()
        else:
            mano_layer = ManoLayer(side='right', use_pca=False).cuda()


    output = mano_layer(quaternion_to_axis_angle(hand_coeffs.to('cuda')).reshape(-1, 48)) # manopth use axis_angle and should be (B, 48)
    hand_verts, hand_joints = output.verts, output.joints
    hand_joints = hand_joints.cpu().numpy()
    hand_verts = hand_verts.cpu().numpy()
    hand_joints += hand_tsls.cpu().numpy()[...,None,:]

    # print("Hand tsl")
    # for i in hand_tsls[30]:
    #     print(i)
    # print("Hand coeffs")
    # for i in hand_coeffs[30]:
    #     print(i)
    # print("Hand joints")
    # for i in hand_joints[30]:
    #     print(i)

    # print("side:", side)

    return hand_joints, hand_verts


def get_subitem(ls:List[np.ndarray], idx:int, not_list=False):
    if ls is None:
        return None
    if type(ls) is not list or not_list:
        return ls[idx]
    return [item[idx] for item in ls]



def get_vis_hand_keypoints_with_color_gradient_and_lines(
    gt_posi_pts: np.ndarray, 
    marker_size: float = 5, # 添加了 marker_size 参数
    line_width: float = 2, # 添加了 line_width 参数
    color_scale='Viridis', 
    finger_groups=None, 
    emphasize_idx=None):
    """
    Visualize the 21 hand key points with different colors for different fingers,
    decreasing opacity based on the distance from the root point, and add lines
    connecting points within each finger and from the root to the root of each finger.

    Args:
        gt_posi_pts (np.ndarray): Ground truth position points (21 hand keypoints)
        color_scale (str): The color scale to use (e.g., 'Viridis')
    """

    # Define the groupings of the hand keypoints based on fingers
    if finger_groups is None:
        if gt_posi_pts.shape[-2] == 21:
            finger_groups = {
                'thumb': [0, 1, 2, 3, 4],
                'index': [5, 6, 7, 8],
                'middle': [9, 10, 11, 12],
                'ring': [13, 14, 15, 16],
                'pinky': [17, 18, 19, 20]
            }
        elif gt_posi_pts.shape[-2] == 16:
            finger_groups = {
                'thumb': [0, 1, 2, 3],
                'index': [4, 5, 6],
                'middle': [7, 8, 9],
                'ring': [10, 11, 12],
                'pinkie': [13, 14, 15]
            }

    # Get colors from the specified color scale
    color_scale_vals = get_colorscale(color_scale)

    data = []
    
    for i, (finger_name, indices) in enumerate(finger_groups.items()):
        # Get a color for this finger from the color scale
        finger_color = color_scale_vals[i % len(color_scale_vals)]

        # Add points and lines within each finger
        for j, idx in enumerate(indices):
            # Calculate the opacity based on the distance from the root (first point in the group)
            opacity = 1.0 - (j / (len(indices)))  # Linear decrease in opacity

            # Add the keypoint with the appropriate color and opacity
            if emphasize_idx is not None and idx == emphasize_idx:
                data.append(go.Scatter3d(
                    x=[gt_posi_pts[idx, 0]], 
                    y=[gt_posi_pts[idx, 1]], 
                    z=[gt_posi_pts[idx, 2]], 
                    mode='markers', 
                    marker=dict(size=marker_size, color='green', opacity=1), 
                    name=f"{finger_name} {j+1}",
                    showlegend=False
                ))
            else:
                data.append(go.Scatter3d(
                    x=[gt_posi_pts[idx, 0]], 
                    y=[gt_posi_pts[idx, 1]], 
                    z=[gt_posi_pts[idx, 2]], 
                    mode='markers', 
                    marker=dict(size=marker_size, color=finger_color[1], opacity=opacity), 
                    name=f"{finger_name} {j+1}",
                    showlegend=False
                ))

            # Add lines between points within the finger
            if j > 0:
                prev_idx = indices[j - 1]
                data.append(go.Scatter3d(
                    x=[gt_posi_pts[prev_idx, 0], gt_posi_pts[idx, 0]], 
                    y=[gt_posi_pts[prev_idx, 1], gt_posi_pts[idx, 1]], 
                    z=[gt_posi_pts[prev_idx, 2], gt_posi_pts[idx, 2]], 
                    mode='lines', 
                    line=dict(color=finger_color[1], width=line_width),
                    name=f"{finger_name} Line {j}",
                    showlegend=False
                ))

        # Add lines from the root (index 0) to the root of each finger (index 5, 9, 13, 17)
        if indices[0] != 0:
            data.append(go.Scatter3d(
                x=[gt_posi_pts[0, 0], gt_posi_pts[indices[0], 0]], 
                y=[gt_posi_pts[0, 1], gt_posi_pts[indices[0], 1]], 
                z=[gt_posi_pts[0, 2], gt_posi_pts[indices[0], 2]], 
                mode='lines', 
                line=dict(color=finger_color[1], width=line_width,), 
                name=f"Root to {finger_name}",
                showlegend=False
            ))
    return data



def visualize_hand_and_joints(
    mano_joint: np.array = None, 
    robot_keypoints: Dict[str, torch.Tensor] = None,
    # robot_hand_mesh: o3d.geometry.TriangleMesh = None,
    robot_link_mesh: Dict[str, o3d.geometry.TriangleMesh] = None,
    robot_approx_mesh: Dict[str, o3d.geometry.TriangleMesh] = None,
    human_keypoints: np.array = None,
    contact_points: np.array = None,
    penetration_keypoints: np.array = None,
    filename: str = None
):
    """
    Visualizes MANO joints, Robot keypoints, and an optional robot hand mesh.
    Uses the detailed `get_vis_hand_keypoints_with_color_gradient_and_lines` 
    to render the human skeleton.
    """
    data_traces = []

    # # --- 1. Add Robot Hand Mesh ---
    # if robot_hand_mesh is not None:
    #     verts = np.asarray(robot_hand_mesh.vertices)
    #     faces = np.asarray(robot_hand_mesh.triangles)
    #     mesh_trace = go.Mesh3d(
    #         x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
    #         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    #         color='lightgrey', opacity=0.5, name="Robot Hand Mesh"
    #     )
    #     data_traces.append(mesh_trace)

    # --- 2. Add Labeled MANO Joint Points ---
    if mano_joint is not None:
        mano_np = mano_joint
        for i in range(mano_np.shape[0]):
            point = mano_np[i]
            trace = go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[point[2]],
                mode='markers+text', marker=dict(size=4, color='red', symbol='circle'),
                text=[str(i)], textfont=dict(color='darkred', size=10),
                name=f"MANO Joint {i}"
            )
            data_traces.append(trace)
    if contact_points is not None:
        data_traces.append(go.Scatter3d(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2],
                        mode="markers", marker=dict(size=1, color="orange"), 
                        name="Contact Points (Robot)"))
    if penetration_keypoints is not None and penetration_keypoints.shape[0] > 0:
        data_traces.append(go.Scatter3d(
            x=penetration_keypoints[:, 0],
            y=penetration_keypoints[:, 1],
            z=penetration_keypoints[:, 2],
            mode="markers",
            marker=dict(size=3, color="#ACC313"),
            name="Penetration Keypoints"
        ))

    # --- 3. Add Labeled Robot Joint Points ---
    if robot_keypoints is not None:
        for i, (joint_name, point_tensor) in enumerate(robot_keypoints.items()):
            point = point_tensor.squeeze().detach().cpu().numpy()
            trace = go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[point[2]],
                mode='markers+text', marker=dict(size=4, color='blue', symbol='diamond'),
                text=[str(i)], textfont=dict(color='darkblue', size=10),
                name=f"Robot Joint {i} ({joint_name})"
            )
            data_traces.append(trace)

    # --- 4. Add Human/MANO Skeleton using YOUR provided function ---
    if human_keypoints is not None:
        human_keypoints_np = human_keypoints
        
        # Call your function to get the list of detailed skeleton traces
        # Using 'Reds' colorscale to visually connect skeleton to the red MANO points
        skeleton_traces = get_vis_hand_keypoints_with_color_gradient_and_lines(
            human_keypoints_np, 
            color_scale='Reds'
        )
        
        # Add all the generated traces to our main data list
        data_traces.extend(skeleton_traces)
        
    # --- 5. Add Robot Link Mesh ---
    if robot_link_mesh is not None:
        for i, (link_name, link_mesh) in enumerate(robot_link_mesh.items()):
            verts = np.asarray(link_mesh.vertices)
            faces = np.asarray(link_mesh.triangles)
            mesh_trace = go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='lightgrey', opacity=0.8, name=f"Robot Link {i} ({link_name})", showlegend=True
            )
            data_traces.append(mesh_trace)
    if robot_approx_mesh is not None:
        for i, (link_name, link_mesh) in enumerate(robot_approx_mesh.items()):
            verts = np.asarray(link_mesh.vertices)
            faces = np.asarray(link_mesh.triangles)
            mesh_trace = go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='#9DEBEB', opacity=0.2, name=f"Robot Link {i} ({link_name})", showlegend=True
            )
            data_traces.append(mesh_trace)

    # --- 6. Create and show/save the figure ---
    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title="MANO vs. Robot Joint and Mesh Comparison",
        scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        legend_title_text='Joints & Meshes',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    if filename is not None:
        fig.write_html(f"{filename}.html")
        print(f"Visualization saved to {filename}.html")
    else:
        fig.show()


def vis_frames_plotly(
    pc_ls: List[np.ndarray] = None,
    hand_pts_ls: List[np.ndarray] = None,
    transformation_ls: List[np.ndarray] = None,
    gt_transformation_ls: List[np.ndarray] = None,
    gt_posi_pts: np.ndarray = None,
    posi_pts_ls: List[np.ndarray] = None,
    hand_joints_ls: List[np.ndarray] = None,
    gt_hand_joints: np.ndarray = None,
    hand_mesh=None,
    obj_mesh=None,
    hand_mesh_ls=None,
    object_mesh_ls=None,
    hand_name_ls: List[str] = None,
    show_axis: bool = False,
    show_line: bool = False, 
    filename: str = None,
):
    """
    使用 Plotly 将包含时间序列的3D数据可视化为可播放的动画帧。

    此函数作为 `vis_pc_coor_plotly` 的动画封装器。它接收具有时间维度的数据，
    为每个时间步创建一个帧（Frame），并将它们组合成一个带有播放/暂停按钮
    和滑动条的交互式HTML动画。

    Args:
        pc_ls (List[np.ndarray], optional):
            - 输入格式: 一个 NumPy 数组的列表。列表长度代表不同类型的点云数量。
                        列表中的每个元素是一个形状为 `(T, N, 3)` 的 NumPy 数组，其中 `T` 是
                        时间帧的数量，`N` 是点的数量，`3` 代表 (x, y, z) 坐标。

        hand_pts_ls (List[np.ndarray], optional):
            - 输入格式: 与 `pc_ls` 类似，一个形状为 `(T, N, 3)` 的手部点云数组的列表。

        transformation_ls (List[np.ndarray], optional):
            - 输入格式: 一个 NumPy 数组的列表。每个元素是形状为 `(T, 4, 4)` 的数组，
                        代表一个坐标系随 `T` 帧变化的齐次变换矩阵。

        gt_transformation_ls (List[np.ndarray], optional):
            - 输入格式: 与 `transformation_ls` 类似，一个 `(T, 4, 4)` 变换矩阵数组的列表。

        gt_posi_pts (np.ndarray, optional):
            - 输入格式: 一个形状为 `(T, N, 3)` 的 NumPy 数组，代表一组真值位置点随时间的变化。

        posi_pts_ls (List[np.ndarray], optional):
            - 输入格式: 一个形状为 `(T, N, 3)` 的位置点数组的列表。

        hand_joints_ls (List[np.ndarray], optional):
            - 输入格式: 一个 NumPy 数组的列表。列表长度代表要动画显示的手的数量。
                        每个元素是形状为 `(T, K, 3)` 的数组，`K` 是关节点数量（如21或16）。

        gt_hand_joints (np.ndarray, optional):
            - 输入格式: 一个形状为 `(T, K, 3)` 的 NumPy 数组，代表真值手部关节点随时间的变化。

        hand_mesh (List, optional):
            - 输入格式: 一个网格对象的列表，列表长度为 `T`。每个元素是单帧的手部网格。
                        （注意：此参数用于单个手部网格的时间序列）。

        obj_mesh (List, optional):
            - 输入格式: 一个网格对象的列表，列表长度为 `T`。每个元素是单帧的物体网格。
                        （注意：此参数用于单个物体网格的时间序列）。

        hand_mesh_ls (List[List], optional):
            - 输入格式: 一个列表的列表 `List[List[Mesh]]`。外层列表的长度代表要显示的不同
                        手的数量。内层列表的长度为 `T`，包含了该手在每个时间帧的网格对象。

        object_mesh_ls (List[List], optional):
            - 输入格式: 一个列表的列表 `List[List[Mesh]]`。外层列表的长度代表物体数量。
                        内层列表的长度为 `T`，包含了该物体在每个时间帧的网格对象。

        hand_name_ls (List[str], optional):
            - 输入格式: 字符串列表，长度应与 `hand_mesh_ls` 的外层列表匹配。
            - 表现形式: 为 `hand_mesh_ls` 中的每个手部网格序列在图例中提供名称。

        show_axis (bool, optional):
            - 输入格式: 布尔值。
            - 表现形式: 如果为 `True`，则显示场景的X, Y, Z坐标轴；否则隐藏。

        show_line (bool, optional): 
            - 如果为 True，将为 `gt_posi_pts` 和 `posi_pts_ls` 中的每对对应点绘制连接线

        filename (str, optional):
            - 输入格式: 字符串，代表文件名（例如 "output"，没有后缀）。
            - 表现形式: 控制参数。如果提供，则不会在浏览器中显示动画，而是将其保存为HTML文件。
                        如果为 `None`，则会生成一个临时HTML文件并在浏览器中打开。
    """

    # Determine the number of frames, T, from the first available animated list.
    T = pc_ls[0].shape[0] if pc_ls is not None else gt_hand_joints.shape[0]

    initial_data = vis_pc_coor_plotly(
        pc_ls=get_subitem(pc_ls, 0),
        hand_pts_ls=get_subitem(hand_pts_ls, 0),
        transformation_ls=get_subitem(transformation_ls, 0),
        gt_transformation_ls=get_subitem(gt_transformation_ls, 0),
        gt_posi_pts=get_subitem(gt_posi_pts, 0, not_list=True),
        posi_pts_ls=get_subitem(posi_pts_ls, 0),
        hand_joints_ls=get_subitem(hand_joints_ls, 0),
        gt_hand_joints=get_subitem(gt_hand_joints, 0),
        hand_mesh=get_subitem(hand_mesh, 0, not_list=True),
        obj_mesh=get_subitem(obj_mesh, 0, not_list=True),
        hand_mesh_ls=get_subitem(hand_mesh_ls, 0),
        obj_mesh_ls=get_subitem(object_mesh_ls, 0),
        hand_name_ls=hand_name_ls,
        return_data=True,
        show_line=show_line,  
    )

    frames = []
    for t in tqdm(range(T), desc="Creating frames"):
        data = vis_pc_coor_plotly(
            pc_ls=get_subitem(pc_ls, t),
            hand_pts_ls=get_subitem(hand_pts_ls, t),
            transformation_ls=get_subitem(transformation_ls, t),
            gt_transformation_ls=get_subitem(gt_transformation_ls, t),
            gt_posi_pts=get_subitem(gt_posi_pts, t, not_list=True),
            posi_pts_ls=get_subitem(posi_pts_ls, t),
            hand_joints_ls=get_subitem(hand_joints_ls, t),
            gt_hand_joints=get_subitem(gt_hand_joints, t),
            hand_mesh=get_subitem(hand_mesh, t, not_list=True),
            obj_mesh=get_subitem(obj_mesh, t, not_list=True),
            hand_mesh_ls=get_subitem(hand_mesh_ls, t),
            obj_mesh_ls=get_subitem(object_mesh_ls, t),
            hand_name_ls=hand_name_ls,
            return_data=True,
            show_line=show_line,  
        )
        frames.append(go.Frame(data=data, name=f"Frame {t}"))

    slider_steps = []
    for t in range(T):
        step = {
            "method": "animate",
            "label": f"{t}",
            "args": [
                [f"Frame {t}"],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
        }
        slider_steps.append(step)

    layout = go.Layout(
        scene=dict(
            aspectmode="data",
            xaxis_visible=show_axis,
            yaxis_visible=show_axis,
            zaxis_visible=show_axis,
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": slider_steps,
            }
        ],
    )

    fig = go.Figure(data=initial_data, layout=layout, frames=frames)
    if filename is not None:
        fig.write_html(f"{filename}.html")
    else:
        # fig.show()
        fig.write_html("temp_vis.html")
        webbrowser.open("temp_vis.html")
        os.remove("temp_vis.html")



def vis_pc_coor_plotly(
    pc_ls: Optional[List[np.ndarray]] = None,
    hand_pts_ls: Optional[List[np.ndarray]] = None,
    transformation_ls: Optional[List[np.ndarray]] = None,
    gt_transformation_ls: Optional[List[np.ndarray]] = None,
    gt_posi_pts: Optional[np.ndarray] = None,
    posi_pts_ls: Optional[List[np.ndarray]] = None,
    hand_joints_ls: Optional[List[np.ndarray]] = None,
    gt_hand_joints: Optional[np.ndarray] = None,
    opt_points: Optional[np.ndarray] = None,
    gt_opt_points: Optional[np.ndarray] = None,
    voxel_dict: Optional[Dict[str, Any]] = None,
    hand_mesh: Any = None,
    hand_mesh_ls: Optional[List[Any]] = None,
    hand_name_ls: Optional[List[str]] = None,
    show_axis: bool = False,
    obj_mesh: Any = None,
    obj_mesh_ls: Optional[List[Any]] = None,
    obj_norm_ls: Optional[List[np.ndarray]] = None,
    return_data: bool = False,
    show_line: bool = False,
    filename: Optional[str] = None,
):
    """
    使用 Plotly 可视化点云、手部/物体网格、关节点和坐标系。
    此函数主要用于在静态图中显示多个元素或一个元素的时间序列。

    Args:
        pc_ls (List[np.ndarray], optional):
            - 输入格式: 一个 NumPy 数组的列表。列表的长度代表要显示的不同点云的数量。
                        列表中的每个元素是一个形状为 `(N, 3)` 的 NumPy 数组，其中 `N` 是点的数量，
                        `3` 代表 (x, y, z) 坐标。
            - 表现形式: 渲染为一系列3D散点图。每个点云会根据其在列表中的顺序，
                        被赋予 "Purples" 色彩图中的一个渐变紫色。点的大小为中等 (size=5)。

        hand_pts_ls (List[np.ndarray], optional):
            - 输入格式: 一个 NumPy 数组的列表。列表长度代表不同手部点云集合的数量。
                        每个元素是形状为 `(N, 3)` 的 NumPy 数组。
            - 表现形式: 渲染为一系列3D散点图。每个点云集合会被赋予 "Oranges" 色彩图中的
                        一个渐变橙色。点的大小较小 (size=3)。

        transformation_ls (List[np.ndarray], optional):
            - 输入格式: 一个 NumPy 数组的列表。列表长度代表要绘制的坐标系的数量。
                        每个元素是一个 `(4, 4)` 的齐次变换矩阵。
            - 表现形式: 渲染为一组小的、不透明的3D坐标系。每个坐标系的X轴为红色，
                        Y轴为绿色，Z轴为蓝色。

        gt_transformation_ls (List[np.ndarray], optional):
            - 输入格式: 与 `transformation_ls` 相同，一个 `(4, 4)` 变换矩阵的列表。
            - 表现形式: 与 `transformation_ls` 类似，但渲染为半透明（透明度0.4）的坐标系，
                        以表示它们是真值（Ground Truth）。

        gt_posi_pts (np.ndarray, optional):
            - 输入格式: 一个形状为 `(N, 3)` 的 NumPy 数组，代表一组真值位置点。
            - 表现形式: 渲染为一组大的（size=12）、绿色的散点。

        posi_pts_ls (List[np.ndarray], optional):
            - 输入格式: 一个 NumPy 数组的列表。每个元素是形状为 `(N, 3)` 的位置点数组。
            - 表现形式: 渲染为一组或多组大的（size=10）、紫色的散点。

        hand_joints_ls (List[np.ndarray], optional):
            - 输入格式: 一个 NumPy 数组的列表。列表长度代表要可视化的手的数量。
                        每个元素是形状为 `(21, 3)` 或 `(16, 3)` 的数组，代表手部关节点的坐标。
            - 表现形式: 渲染成完整的手部骨架（关节点和连接线）。使用 "Plasma" 色彩图进行着色。

        gt_hand_joints (np.ndarray, optional):
            - 输入格式: 一个形状为 `(21, 3)` 或 `(16, 3)` 的 NumPy 数组，代表真值手部关节点。
            - 表现形式: 渲染成一个完整的手部骨架。使用 "Bluered" (蓝红色) 色彩图进行着色。

        opt_points (np.ndarray, optional):
            - 输入格式: 一个形状为 `(N, 3)` 的 NumPy 数组，通常是一组待优化的特定点。
            - 表现形式: 渲染成一个部分的骨架，使用 "Viridis" (绿色系) 色彩图。

        gt_opt_points (np.ndarray, optional):
            - 输入格式: 一个形状为 `(N, 3)` 的 NumPy 数组，是 `opt_points` 的真值。
            - 表现形式: 渲染成一个部分的骨架，使用 "Bluered" (蓝红色) 色彩图。

        voxel_dict (Dict, optional):
            - 输入格式: 包含体素信息的字典，如 `{"grid_centers": (N, 3) np.ndarray, "selected_points": (M, 3) np.ndarray}`。
            - 表现形式: 渲染一个体素网格。"selected_points" 中的体素显示为半透明的蓝色实体方块，
                        其他体素显示为浅灰色的线框。

        hand_mesh (trimesh or open3d.geometry.TriangleMesh, optional):
            - 输入格式: 单个具有 `.vertices` 和 `.triangles` (或 `.faces`) 属性的网格对象。
            - 表现形式: 渲染为单个半透明（透明度0.5）的宝蓝色（royalblue）手部网格。

        hand_mesh_ls (List, optional):
            - 输入格式: 一个网格对象的列表。列表长度代表要显示的手部网格数量。
            - 表现形式: 渲染多个手部网格，每个网格根据其在列表中的顺序被赋予 "Reds" 
                        色彩图中的渐变红色，并带有半透明效果。图例名称来自 `hand_name_ls`。

        hand_name_ls (List[str], optional):
            - 输入格式: 字符串列表，长度应与 `hand_mesh_ls` 匹配。
            - 表现形式: 为 `hand_mesh_ls` 中的每个手部网格在图例中提供名称。

        show_axis (bool, optional):
            - 输入格式: 布尔值。
            - 表现形式: 如果为 `True`，则显示场景的X, Y, Z坐标轴；否则隐藏。

        obj_mesh (List, optional):
            - 输入格式: 一个网格对象的列表。列表长度是单帧内要显示的物体数量。
            - 表现形式: 渲染一个或多个物体网格。每个网格都是不透明的浅灰色（`#D3D3D3`）。

        obj_mesh_ls (List, optional):
            - 输入格式: 一个网格对象的列表。通常用于在静态图中表示单个物体随时间变化的轨迹。
            - 表现形式: 渲染多个物体网格，每个网格根据顺序被赋予 "Blues" 色彩图中的渐变蓝色。

        obj_norm_ls (List[np.ndarray], optional):
            - 输入格式: 一个 NumPy 数组的列表，长度需与 `pc_ls` 匹配。每个元素是形状为 `(N, 3)` 的数组，
                        包含与 `pc_ls` 中对应点云每个点相匹配的法向量。
            - 表现形式: 从每个点云的点上绘制出代表其法向量的短线。线的颜色与它们所属的点云颜色一致
                        （即 "Purples" 渐变色）。

        return_data (bool, optional):
            - 输入格式: 布尔值。
            - 表现形式: 控制参数。如果 `True`，函数不绘图，而是返回一个包含所有 Plotly 绘图对象的列表。

        show_line (bool, optional):
            - 如果为 True，将为 `gt_posi_pts` 和 `posi_pts_ls` 中的每对对应点绘制连接线。

        filename (str, optional):
            - 输入格式: 字符串，代表文件名（例如 "output"，没有后缀）。
            - 表现形式: 控制参数。如果提供，则不会在浏览器中显示图像，而是将其保存为HTML文件。
    """

    # --- 开始：用于管理绘图尺寸和字号的新代码结构 ---
    # 定义宏 IF_PC: 如果为 True, 则所有绘图相关的“尺寸”属性缩小至原值的 1/3。
    # 如果为 False，则使用原始尺寸。
    IF_PC = True # 您可以通过修改此宏来控制整体尺寸缩放

    # 根据 IF_PC 决定缩放因子
    scale_factor = 1/3 if IF_PC else 1

    # 定义所有绘图相关的基础尺寸属性（未缩放的原始值）
    drawing_sizes_base = {
        # 坐标系 (Coordinate Frame) 的尺寸和线宽
        "coord_frame_size": 0.02,
        "coord_frame_line_width": 12,

        # 点云 (Point Clouds) 的标记点尺寸
        "pc_marker_size": 5,

        # 手部点 (Hand Points) 的标记点尺寸
        "hand_pts_marker_size": 3,

        # 位置点 (Position Points) 的标记点尺寸
        "gt_posi_pts_marker_size": 12,
        "posi_pts_marker_size": 10,
        "correspondence_line_width": 2,

        # 对象法线 (Object Normals) 的长度和线宽
        "obj_normal_scale": 0.02,
        "obj_normal_line_width": 2,

        # 体素 (Voxel) 的相关视觉属性 (opacity 不是尺寸，不缩放；线宽是尺寸，缩放)
        "voxel_selected_opacity": 0.5, # 不缩放
        "voxel_empty_line_width": 1,

        # 关键点 (Keypoints) 的基础尺寸。这些值将作为参数传递给 get_vis_hand_keypoints_with_color_gradient_and_lines
        # get_vis_hand_keypoints_with_color_gradient_and_lines 会根据这些基础值应用其内部的相对比例。
        "keypoint_base_marker_size": 10, # 原始 get_vis_hand_keypoints_with_color_gradient_and_lines 中常规点的大小
        "keypoint_base_line_width": 10,  # 原始 get_vis_hand_keypoints_with_color_gradient_and_lines 中指节连接线的宽度

        # 字号相关属性
        "axis_title_font_size": 14,
        "legend_font_size": 12,
        "mesh_name_font_size": 12, # 暂未直接使用，但为了管理一致性而保留
    }

    # 应用 scale_factor 到所有需要缩放的数值
    drawing_sizes = {}
    for k, v in drawing_sizes_base.items():
        # 'voxel_selected_opacity' 是透明度，不应该被 scale_factor 影响
        if isinstance(v, (int, float)) and k != "voxel_selected_opacity":
            drawing_sizes[k] = v * scale_factor
        else:
            drawing_sizes[k] = v
    # --- 结束：用于管理绘图尺寸和字号的新代码结构 ---


    # Define colors
    red = "rgb(255, 0, 0)"
    green = "rgb(0, 255, 0)"
    blue = "rgb(0, 0, 255)"
    # posi_pts_colors = [blue, green] # 此变量在原始代码中定义但未使用

    # 为时间序列中的点设置插值颜色梯度
    color_gradient_points = plt.get_cmap("Purples")(np.linspace(0.3, 0.8, len(pc_ls))) if pc_ls is not None else []
    color_gradient_hands = plt.get_cmap("Oranges")(np.linspace(0.3, 0.8, len(hand_pts_ls))) if hand_pts_ls is not None else []
    color_gradient_hand_mesh = plt.get_cmap("Reds")(np.linspace(0.3, 0.8, len(hand_mesh_ls))) if hand_mesh_ls is not None else []
    color_gradient_obj_mesh = plt.get_cmap("Blues")(np.linspace(0.3, 0.8, len(obj_mesh_ls))) if obj_mesh_ls is not None else []

    data = []

    def add_coordinate_frame(size, opacity=1, transformation=None, name="Coordinate Frame"):
        origin = np.array([[0, 0, 0]])
        axis = np.array([[size, 0, 0], [0, size, 0], [0, 0, size]])
        if transformation is not None:
            origin = pt_transform(origin, transformation)
            axis = pt_transform(axis, transformation)
        lines = []
        colors = ["red", "green", "blue"]
        for i in range(3):
            lines.append(
                go.Scatter3d(
                    x=[origin[0, 0], axis[i, 0]],
                    y=[origin[0, 1], axis[i, 1]],
                    z=[origin[0, 2], axis[i, 2]],
                    mode="lines",
                    line=dict(color=colors[i], width=drawing_sizes["coord_frame_line_width"]), # 使用动态线宽
                    opacity=opacity,
                    name=name,
                    showlegend=False # 坐标轴不显示在图例中
                )
            )
        return lines

    if voxel_dict is not None:
        grid_centers = voxel_dict["grid_centers"]
        selected_points = voxel_dict.get("selected_points", np.empty((0, 3)))
        vis_empty = voxel_dict.get("vis_empty", True)
        auto_calc_size = voxel_dict.get("auto_calculate_size", True)
        voxel_raw_size = voxel_dict.get("voxel_size", (1.0, 1.0, 1.0)) # 原始体素尺寸

        # 将体素尺寸应用于显示，如果 IF_PC 为 True 则进行缩放
        voxel_display_size = (voxel_raw_size[0] * scale_factor,
                              voxel_raw_size[1] * scale_factor,
                              voxel_raw_size[2] * scale_factor)

        def unique_within_tolerance(values: np.ndarray, tol: float = 1e-3) -> np.ndarray:
            sorted_vals = np.sort(values)
            merged = [sorted_vals[0]]
            for x in sorted_vals[1:]:
                if abs(x - merged[-1]) >= tol:
                    merged.append(x)
            return np.array(merged)

        if auto_calc_size:
            tolerance = 1e-3
            unique_x = unique_within_tolerance(grid_centers[:, 0], tol=tolerance)
            unique_y = unique_within_tolerance(grid_centers[:, 1], tol=tolerance)
            unique_z = unique_within_tolerance(grid_centers[:, 2], tol=tolerance)
            dx = np.min(np.diff(unique_x)) if unique_x.size > 1 else 0
            dy = np.min(np.diff(unique_y)) if unique_y.size > 1 else 0
            dz = np.min(np.diff(unique_z)) if unique_z.size > 1 else 0

            # 如果自动计算，则按比例缩放这些维度
            voxel_display_size = (dx * scale_factor, dy * scale_factor, dz * scale_factor)

        for center in grid_centers:
            cx, cy, cz = center
            # 检查此中心点是否为选定的点之一
            is_selected = False
            if selected_points.size > 0:
                is_selected = np.any(np.all(np.isclose(center, selected_points, atol=1e-6), axis=1)) # 使用 isclose 进行浮点比较

            x_edges = [cx - voxel_display_size[0] / 2, cx + voxel_display_size[0] / 2]
            y_edges = [cy - voxel_display_size[1] / 2, cy + voxel_display_size[1] / 2]
            z_edges = [cz - voxel_display_size[2] / 2, cz + voxel_display_size[2] / 2]

            if is_selected:
                data.append(
                    go.Mesh3d(
                        x=[x_edges[0],x_edges[1],x_edges[1],x_edges[0],x_edges[0],x_edges[1],x_edges[1],x_edges[0]],
                        y=[y_edges[0],y_edges[0],y_edges[1],y_edges[1],y_edges[0],y_edges[0],y_edges[1],y_edges[1]],
                        z=[z_edges[0],z_edges[0],z_edges[0],z_edges[0],z_edges[1],z_edges[1],z_edges[1],z_edges[1]],
                        color="blue", opacity=drawing_sizes["voxel_selected_opacity"], alphahull=0,
                        name="Selected Voxel", showlegend=False
                    )
                )
            elif vis_empty:
                corners = np.array([[x, y, z] for z in z_edges for y in y_edges for x in x_edges])
                edges = [(0, 1),(1, 3),(3, 2),(2, 0),(4, 5),(5, 7),(7, 6),(6, 4),(0, 4),(1, 5),(2, 6),(3, 7)]
                x_lines, y_lines, z_lines = [], [], []
                for edge in edges:
                    x_lines.extend([corners[edge[0], 0], corners[edge[1], 0], None])
                    y_lines.extend([corners[edge[0], 1], corners[edge[1], 1], None])
                    z_lines.extend([corners[edge[0], 2], corners[edge[1], 2], None])
                data.append(
                    go.Scatter3d(
                        x=x_lines, y=y_lines, z=z_lines, mode="lines",
                        line=dict(color="lightgray", width=drawing_sizes["voxel_empty_line_width"]),
                        name="Empty Voxel", showlegend=False
                    )
                )

    if transformation_ls is not None:
        for i, transformation in enumerate(transformation_ls):
            data.extend(add_coordinate_frame(drawing_sizes["coord_frame_size"], 1, transformation, name=f"Transformation {i+1}"))

    if gt_transformation_ls is not None:
        for i, transformation in enumerate(gt_transformation_ls):
            data.extend(add_coordinate_frame(drawing_sizes["coord_frame_size"], 0.4, transformation, name=f"GT Transformation {i+1}"))

    if pc_ls is not None:
        for i, pc in enumerate(pc_ls):
            color_rgb = [int(c * 255) for c in color_gradient_points[i][:3]]
            color_str = f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"
            data.append(
                go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode="markers",
                             marker=dict(size=drawing_sizes["pc_marker_size"], color=color_str), name=f"Point Cloud {i+1}", showlegend=True)
            )

    if hand_pts_ls is not None:
        for i, hand_pts in enumerate(hand_pts_ls):
            color_rgb = [int(c * 255) for c in color_gradient_hands[i][:3]]
            color_str = f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"
            data.append(
                go.Scatter3d(x=hand_pts[:, 0], y=hand_pts[:, 1], z=hand_pts[:, 2], mode="markers",
                             marker=dict(size=drawing_sizes["hand_pts_marker_size"], color=color_str), name=f"Hand Points {i+1}", showlegend=True)
            )

    if gt_posi_pts is not None:
        data.append(
            go.Scatter3d(x=gt_posi_pts[:, 0], y=gt_posi_pts[:, 1], z=gt_posi_pts[:, 2],
                         mode="markers", marker=dict(size=drawing_sizes["gt_posi_pts_marker_size"], color="green"), name="GT Position Points", showlegend=True)
        )

    if posi_pts_ls is not None:
        for i, posi_pts in enumerate(posi_pts_ls):
            data.append(
                go.Scatter3d(x=posi_pts[:, 0], y=posi_pts[:, 1], z=posi_pts[:, 2], mode="markers",
                             marker=dict(size=drawing_sizes["posi_pts_marker_size"], color="purple"), name=f"Position Points {i+1}", showlegend=True)
            )

    if show_line and gt_posi_pts is not None and posi_pts_ls is not None:
        lines_x, lines_y, lines_z = [], [], []

        for target_points_set in posi_pts_ls:
            if target_points_set.shape[0] != gt_posi_pts.shape[0]:
                print(f"Warning: Mismatch in point counts for line drawing. "
                      f"gt_posi_pts has {gt_posi_pts.shape[0]} points, "
                      f"but one posi_pts_ls item has {target_points_set.shape[0]}. Skipping.")
                continue

            for n in range(gt_posi_pts.shape[0]):
                p_gt = gt_posi_pts[n]
                p_target = target_points_set[n]

                lines_x.extend([p_gt[0], p_target[0], None])
                lines_y.extend([p_gt[1], p_target[1], None])
                lines_z.extend([p_gt[2], p_target[2], None])

        if lines_x:
            data.append(go.Scatter3d(
                x=lines_x,
                y=lines_y,
                z=lines_z,
                mode='lines',
                line=dict(color='black', width=drawing_sizes["correspondence_line_width"]), # 使用动态线宽
                name='Correspondence Lines',
                showlegend=False
            ))

    if opt_points is not None:
        finger_groups = {"thumb": [0, 1], "index": [2], "middle": [3], "ring": [4], "pinky": [5]}
        vis_data = get_vis_hand_keypoints_with_color_gradient_and_lines(
            opt_points[[5, 0, 1, 2, 3, 4]],
            color_scale="Viridis",
            finger_groups=finger_groups,
            marker_size=drawing_sizes["keypoint_base_marker_size"], # 传递缩放后的基础尺寸
            line_width=drawing_sizes["keypoint_base_line_width"]   # 传递缩放后的基础线宽
        )
        for trace in vis_data:
            data.append(trace)

    if gt_opt_points is not None:
        finger_groups = {"thumb": [0, 1], "index": [2], "middle": [3], "ring": [4], "pinky": [5]}
        vis_data = get_vis_hand_keypoints_with_color_gradient_and_lines(
            gt_opt_points[[5, 0, 1, 2, 3, 4]],
            color_scale="Bluered",
            finger_groups=finger_groups,
            marker_size=drawing_sizes["keypoint_base_marker_size"], # 传递缩放后的基础尺寸
            line_width=drawing_sizes["keypoint_base_line_width"]   # 传递缩放后的基础线宽
        )
        for trace in vis_data:
            data.append(trace)

    if gt_hand_joints is not None:
        vis_data = get_vis_hand_keypoints_with_color_gradient_and_lines(
            gt_hand_joints,
            color_scale="Bluered",
            marker_size=drawing_sizes["keypoint_base_marker_size"], # 传递缩放后的基础尺寸
            line_width=drawing_sizes["keypoint_base_line_width"]   # 传递缩放后的基础线宽
        )
        for trace in vis_data:
            data.append(trace)

    if hand_joints_ls is not None:
        for i, hand_joints in enumerate(hand_joints_ls):
            hand_joints_data = get_vis_hand_keypoints_with_color_gradient_and_lines(
                hand_joints,
                color_scale="Plasma",
                marker_size=drawing_sizes["keypoint_base_marker_size"], # 传递缩放后的基础尺寸
                line_width=drawing_sizes["keypoint_base_line_width"]   # 传递缩放后的基础线宽
            )
            for trace in hand_joints_data:
                data.append(trace)

    if hand_mesh is not None:
        verts = np.asarray(hand_mesh.vertices)
        faces = np.asarray(hand_mesh.triangles if hasattr(hand_mesh, "triangles") else hand_mesh.faces)
        data.append(
            go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                      i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                      color="royalblue", opacity=0.5, name="Hand Mesh", showlegend=True)
        )

    if obj_mesh is not None:
        # 如果 obj_mesh 是单个 mesh 对象，将其视为包含一个元素的列表
        if not isinstance(obj_mesh, list):
            obj_mesh = [obj_mesh]

        for i, mesh_item in enumerate(obj_mesh):
            if mesh_item is None or not hasattr(mesh_item, "vertices"): continue
            verts = np.asarray(mesh_item.vertices)
            faces = np.asarray(mesh_item.triangles if hasattr(mesh_item, "triangles") else mesh_item.faces)
            if verts.size == 0 or faces.size == 0: continue
            data.append(
                go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                          i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                          color="#D3D3D3", opacity=1, name=f"Object Mesh {i}", showlegend=True)
            )

    if obj_mesh_ls is not None:
        for i, obj_mesh_item in enumerate(obj_mesh_ls):
            verts = np.asarray(obj_mesh_item.vertices)
            faces = np.asarray(obj_mesh_item.triangles if hasattr(obj_mesh_item, "triangles") else obj_mesh_item.faces)
            color_rgb = [int(c * 255) for c in color_gradient_obj_mesh[i][:3]]
            color_str = f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"
            data.append(
                go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            color=color_str, name=f"Object Mesh {i+1}", showlegend=True)
            )

    if obj_norm_ls is not None:
        scale = drawing_sizes["obj_normal_scale"] # 使用缩放后的值
        if pc_ls is None:
            print("Warning: obj_norm_ls 提供，但 pc_ls 为 None。没有对应的点无法绘制法线。")
        else:
            for i, (pc, normals) in enumerate(zip(pc_ls, obj_norm_ls)):
                if pc is None or normals is None: continue # 跳过空数据
                color_rgb = [int(c * 255) for c in color_gradient_points[i][:3]]
                color_str = f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"
                x_lines, y_lines, z_lines = [], [], []

                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                normalized_normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms != 0)

                for point, normal in zip(pc, normalized_normals):
                    start, end = point, point + scale * normal
                    x_lines.extend([start[0], end[0], None])
                    y_lines.extend([start[1], end[1], None])
                    z_lines.extend([start[2], end[2], None])
                data.append(
                    go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode="lines",
                                 line=dict(color=color_str, width=drawing_sizes["obj_normal_line_width"]), # 使用缩放后的线宽
                                 name=f"Object Normals {i+1}", showlegend=True)
                )

    if hand_mesh_ls is not None:
        for i, hand_mesh_item in enumerate(hand_mesh_ls):
            verts = np.asarray(hand_mesh_item.vertices)
            faces = np.asarray(hand_mesh_item.triangles if hasattr(hand_mesh_item, "triangles") else hand_mesh_item.faces)
            color_rgb = [int(c * 255) for c in color_gradient_hand_mesh[i][:3]]
            color_str = f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"
            
            mesh_name = f"Hand Mesh {i+1}"
            if hand_name_ls is not None and i < len(hand_name_ls):
                mesh_name = hand_name_ls[i]

            data.append(
                go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                          i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                          color=color_str, opacity=0.5, name=mesh_name, showlegend=True)
            )

    if return_data:
        return data
    else:
        fig = go.Figure(data=data)
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis_visible=show_axis,
                yaxis_visible=show_axis,
                zaxis_visible=show_axis,
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    title=dict(text="X", font=dict(size=drawing_sizes["axis_title_font_size"])), # 应用轴标题字号
                    showgrid=show_axis,
                    zeroline=show_axis
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    title=dict(text="Y", font=dict(size=drawing_sizes["axis_title_font_size"])), # 应用轴标题字号
                    showgrid=show_axis,
                    zeroline=show_axis
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    title=dict(text="Z", font=dict(size=drawing_sizes["axis_title_font_size"])), # 应用轴标题字号
                    showgrid=show_axis,
                    zeroline=show_axis
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                x=1, y=1,
                xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="Black",
                borderwidth=1,
                font=dict(size=drawing_sizes["legend_font_size"]) # 应用图例字号
            )
        )

        if filename is not None:
            fig.write_html(f"{filename}.html")
        else:
            fig.show()



def visualize_dex_hand_sequence_together(seq_data_ls, name_list, filename: Optional[str] = None):

    seq_data = seq_data_ls[0]
    pc = get_point_clouds_from_human_data(seq_data)
    pc_ls = apply_transformation_human_data(pc, seq_data["obj_poses"]) # TxNx3
    object_mesh = get_object_meshes_from_human_data(seq_data)   
    object_ls = apply_transformation_on_object_mesh(object_mesh, seq_data["obj_poses"]) # a list of (B) a list of (T) meshes
    mano_hand_joints, hand_verts = extract_hand_points_and_mesh(seq_data["hand_tsls"], seq_data["hand_coeffs"], seq_data["side"])

    hand_mesh_ls = []
    hand_type = 'left' if seq_data["side"] == 0 else 'right'
    robot_name_str = seq_data["which_hand"]
    robot = load_robot(robot_name_str, hand_type)

    for seq_data in seq_data_ls:
        hand_meshes = []
        for i in tqdm(range(seq_data["hand_poses"].shape[0])):
            robot.set_qpos(seq_data["hand_poses"][i])
            hand_mesh = robot.get_hand_mesh()
            hand_meshes.append(hand_mesh)
        hand_mesh_ls.append(hand_meshes)

    vis_frames_plotly(
        pc_ls=[pc_ls],
        # object_mesh_ls=object_ls,
        hand_mesh_ls=hand_mesh_ls,
        hand_name_ls=name_list,
        gt_hand_joints=mano_hand_joints,
        filename=filename
    )



def _vis_dexhand_optimize_frame(
    pc_ls, gt_joints, hand_mesh, gt_posi, posi, obj_mesh, 
    contact_pt, corr_contact_pt, penetration_keypoints, inner_pen_pts, outer_pen_pts
):
    """
    Helper function to generate all Plotly traces for a SINGLE frame of the optimization visualization.
    """
    data = []
    
    # 1. Plot Point Cloud
    if pc_ls is not None:
        for pc in pc_ls:
            data.append(go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode="markers",
                                    marker=dict(size=3, color="lightpink", opacity=0.7), name="Point Cloud"))

    # 2. Plot GT Hand Joints (e.g., MANO)
    if gt_joints is not None:
        data.extend(get_vis_hand_keypoints_with_color_gradient_and_lines(gt_joints, color_scale="Bluered"))

    # 3. Plot Robot Hand Mesh
    if hand_mesh is not None:
        mesh_item = hand_mesh[0] if isinstance(hand_mesh, list) else hand_mesh
        verts = np.asarray(mesh_item.vertices)
        faces = np.asarray(mesh_item.triangles if hasattr(mesh_item, "triangles") else mesh_item.faces)
        data.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                              i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                              color="#E1B685", opacity=0.9, name="Robot Hand Mesh"))

    # 4. Plot Object Mesh
    if obj_mesh is not None:
        for i, mesh_item in enumerate(obj_mesh):
            verts = np.asarray(mesh_item.vertices)
            faces = np.asarray(mesh_item.triangles if hasattr(mesh_item, "triangles") else mesh_item.faces)
            data.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                  i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                  color="#D3D3D3", opacity=1, name=f"Object Mesh {i}"))

    # 5. Plot GT/Predicted Position Points + Lines
    if gt_posi is not None:
        data.append(go.Scatter3d(x=gt_posi[:, 0], y=gt_posi[:, 1], z=gt_posi[:, 2],
                                 mode="markers", marker=dict(size=12, color="green"), name="GT Position Points"))
    if posi is not None:
        data.append(go.Scatter3d(x=posi[:, 0], y=posi[:, 1], z=posi[:, 2],
                                mode="markers", marker=dict(size=10, color="purple"), name=f"Robot Position Points"))
    
    if gt_posi is not None and posi is not None:
        lines_x, lines_y, lines_z = [], [], []
        if posi.shape[0] == gt_posi.shape[0]:
            for n in range(gt_posi.shape[0]):
                p_gt, p_target = gt_posi[n], posi[n]
                lines_x.extend([p_gt[0], p_target[0], None])
                lines_y.extend([p_gt[1], p_target[1], None])
                lines_z.extend([p_gt[2], p_target[2], None])
        if lines_x:
            data.append(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines',
                                     line=dict(color='black', width=2), name='Correspondence Lines', showlegend=False))

    # 6. Plot Contact Points + Lines
    if contact_pt is not None:
        data.append(go.Scatter3d(x=contact_pt[:, 0], y=contact_pt[:, 1], z=contact_pt[:, 2],
                                 mode="markers", marker=dict(size=6, color="orange"), name="Contact Points (Robot)"))
    if corr_contact_pt is not None:
        data.append(go.Scatter3d(x=corr_contact_pt[:, 0], y=corr_contact_pt[:, 1], z=corr_contact_pt[:, 2],
                                 mode="markers", marker=dict(size=6, color="darkblue"), name="Contact Points (Object)"))

    if contact_pt is not None and corr_contact_pt is not None:
        lines_x, lines_y, lines_z = [], [], []
        if contact_pt.shape[0] == corr_contact_pt.shape[0]:
            for n in range(contact_pt.shape[0]):
                p1, p2 = contact_pt[n], corr_contact_pt[n]
                lines_x.extend([p1[0], p2[0], None])
                lines_y.extend([p1[1], p2[1], None])
                lines_z.extend([p1[2], p2[2], None])
        if lines_x:
            data.append(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines',
                                     line=dict(color='grey', width=1), name='Contact Correspondence', showlegend=False))
            
    # 7. Plot Self-Penetration Keypoints (lime green)
    if penetration_keypoints is not None and penetration_keypoints.shape[0] > 0:
        data.append(go.Scatter3d(
            x=penetration_keypoints[:, 0], 
            y=penetration_keypoints[:, 1], 
            z=penetration_keypoints[:, 2],
            mode="markers",
            marker=dict(size=10, color="#ACC313"), 
            name="Penetration Keypoints"
        ))
        
    # --- 8. NEW: Plot Object Penetrating Points (brown) ---
    if inner_pen_pts is not None and inner_pen_pts.shape[0] > 0:
        data.append(go.Scatter3d(
            x=inner_pen_pts[:, 0], 
            y=inner_pen_pts[:, 1], 
            z=inner_pen_pts[:, 2],
            mode="markers",
            marker=dict(size=6, color="#502305"), 
            name="Object Inner Points"
        ))
    if outer_pen_pts is not None and outer_pen_pts.shape[0] > 0:
        data.append(go.Scatter3d(
            x=outer_pen_pts[:, 0], 
            y=outer_pen_pts[:, 1], 
            z=outer_pen_pts[:, 2],
            mode="markers",
            marker=dict(size=6, color="#B5652C"), 
            name="Object Outer Points"
        ))

    return data


def _vis_dexhand_optimize_frame(
    pc_ls, gt_joints, hand_mesh, gt_posi, posi, obj_mesh,
    contact_pt, corr_contact_pt, penetration_keypoints, inner_pen_pts, outer_pen_pts
):
    """
    Helper function to generate all Plotly traces for a SINGLE frame of the optimization visualization.
    """
    data = []

    # --- 开始：用于管理绘图尺寸和字号的新代码结构 ---
    # 定义宏 IF_PC: 如果为 True, 则所有绘图相关的“尺寸”属性缩小至原值的 1/3。
    # 如果为 False，则使用原始尺寸。
    IF_PC = True # 您可以通过修改此宏来控制整体尺寸缩放

    # 根据 IF_PC 决定缩放因子
    scale_factor = 1/3 if IF_PC else 1

    # 定义所有绘图相关的基础尺寸属性（未缩放的原始值）
    drawing_sizes_base = {
        # 点云 (Point Clouds) 的标记点尺寸
        "dexhand_pc_marker_size": 3,

        # 关节 (Joints) 和关键点 (Keypoints) 的基础尺寸。
        # 这些值将作为参数传递给 get_vis_hand_keypoints_with_color_gradient_and_lines
        "dexhand_keypoint_base_marker_size": 5, # Default marker size in get_vis_hand_keypoints...
        "dexhand_keypoint_base_line_width": 2,  # Default line width in get_vis_hand_keypoints...

        # 位置点 (Position Points) 的标记点尺寸
        "dexhand_gt_posi_marker_size": 12,
        "dexhand_posi_marker_size": 10,
        "dexhand_correspondence_line_width": 2,

        # 接触点 (Contact Points) 的标记点尺寸和连接线宽
        "dexhand_contact_marker_size": 6,
        "dexhand_contact_line_width": 1,

        # 穿透点 (Penetration Points) 的标记点尺寸
        "dexhand_penetration_keypoints_marker_size": 10,
        "dexhand_inner_outer_pen_pts_marker_size": 6,
    }

    # 应用 scale_factor 到所有需要缩放的数值
    drawing_sizes = {}
    for k, v in drawing_sizes_base.items():
        if isinstance(v, (int, float)):
            drawing_sizes[k] = v * scale_factor
        else:
            drawing_sizes[k] = v
    # --- 结束：用于管理绘图尺寸和字号的新代码结构 ---


    # 1. Plot Point Cloud
    if pc_ls is not None:
        for pc in pc_ls:
            data.append(go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode="markers",
                                    marker=dict(size=drawing_sizes["dexhand_pc_marker_size"], color="lightpink", opacity=0.7), name="Point Cloud"))

    # 2. Plot GT Hand Joints (e.g., MANO)
    if gt_joints is not None:
        data.extend(get_vis_hand_keypoints_with_color_gradient_and_lines(
            gt_joints,
            color_scale="Bluered",
            marker_size=drawing_sizes["dexhand_keypoint_base_marker_size"],
            line_width=drawing_sizes["dexhand_keypoint_base_line_width"]
        ))

    # 3. Plot Robot Hand Mesh
    if hand_mesh is not None:
        mesh_item = hand_mesh[0] if isinstance(hand_mesh, list) else hand_mesh
        verts = np.asarray(mesh_item.vertices)
        faces = np.asarray(mesh_item.triangles if hasattr(mesh_item, "triangles") else mesh_item.faces)
        data.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                              i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                              color="#E1B685", opacity=0.9, name="Robot Hand Mesh"))

    # 4. Plot Object Mesh
    if obj_mesh is not None:
        if not isinstance(obj_mesh, list):
            obj_mesh = [obj_mesh]
        for i, mesh_item in enumerate(obj_mesh):
            verts = np.asarray(mesh_item.vertices)
            faces = np.asarray(mesh_item.triangles if hasattr(mesh_item, "triangles") else mesh_item.faces)
            data.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                  i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                  color="#D3D3D3", opacity=1, name=f"Object Mesh {i}"))

    # 5. Plot GT/Predicted Position Points + Lines
    if gt_posi is not None:
        data.append(go.Scatter3d(x=gt_posi[:, 0], y=gt_posi[:, 1], z=gt_posi[:, 2],
                                 mode="markers", marker=dict(size=drawing_sizes["dexhand_gt_posi_marker_size"], color="green"), name="GT Position Points"))
    if posi is not None:
        data.append(go.Scatter3d(x=posi[:, 0], y=posi[:, 1], z=posi[:, 2],
                                mode="markers", marker=dict(size=drawing_sizes["dexhand_posi_marker_size"], color="purple"), name=f"Robot Position Points"))

    if gt_posi is not None and posi is not None:
        lines_x, lines_y, lines_z = [], [], []
        if posi.shape[0] == gt_posi.shape[0]:
            for n in range(gt_posi.shape[0]):
                p_gt, p_target = gt_posi[n], posi[n]
                lines_x.extend([p_gt[0], p_target[0], None])
                lines_y.extend([p_gt[1], p_target[1], None])
                lines_z.extend([p_gt[2], p_target[2], None])
        if lines_x:
            data.append(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines',
                                     line=dict(color='black', width=drawing_sizes["dexhand_correspondence_line_width"]), name='Correspondence Lines', showlegend=False))

    # 6. Plot Contact Points + Lines
    if contact_pt is not None:
        data.append(go.Scatter3d(x=contact_pt[:, 0], y=contact_pt[:, 1], z=contact_pt[:, 2],
                                 mode="markers", marker=dict(size=drawing_sizes["dexhand_contact_marker_size"], color="orange"), name="Contact Points (Robot)"))
    if corr_contact_pt is not None:
        data.append(go.Scatter3d(x=corr_contact_pt[:, 0], y=corr_contact_pt[:, 1], z=corr_contact_pt[:, 2],
                                 mode="markers", marker=dict(size=drawing_sizes["dexhand_contact_marker_size"], color="darkblue"), name="Contact Points (Object)"))

    if contact_pt is not None and corr_contact_pt is not None:
        lines_x, lines_y, lines_z = [], [], []
        if contact_pt.shape[0] == corr_contact_pt.shape[0]:
            for n in range(contact_pt.shape[0]):
                p1, p2 = contact_pt[n], corr_contact_pt[n]
                lines_x.extend([p1[0], p2[0], None])
                lines_y.extend([p1[1], p2[1], None])
                lines_z.extend([p1[2], p2[2], None])
        if lines_x:
            data.append(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines',
                                     line=dict(color='grey', width=drawing_sizes["dexhand_contact_line_width"]), name='Contact Correspondence', showlegend=False))

    # 7. Plot Self-Penetration Keypoints (lime green)
    if penetration_keypoints is not None and penetration_keypoints.shape[0] > 0:
        data.append(go.Scatter3d(
            x=penetration_keypoints[:, 0],
            y=penetration_keypoints[:, 1],
            z=penetration_keypoints[:, 2],
            mode="markers",
            marker=dict(size=drawing_sizes["dexhand_penetration_keypoints_marker_size"], color="#ACC313"),
            name="Penetration Keypoints"
        ))

    # --- 8. NEW: Plot Object Penetrating Points (brown) ---
    if inner_pen_pts is not None and inner_pen_pts.shape[0] > 0:
        data.append(go.Scatter3d(
            x=inner_pen_pts[:, 0],
            y=inner_pen_pts[:, 1],
            z=inner_pen_pts[:, 2],
            mode="markers",
            marker=dict(size=drawing_sizes["dexhand_inner_outer_pen_pts_marker_size"], color="#502305"),
            name="Object Inner Points"
        ))
    if outer_pen_pts is not None and outer_pen_pts.shape[0] > 0:
        data.append(go.Scatter3d(
            x=outer_pen_pts[:, 0],
            y=outer_pen_pts[:, 1],
            z=outer_pen_pts[:, 2],
            mode="markers",
            marker=dict(size=drawing_sizes["dexhand_inner_outer_pen_pts_marker_size"], color="#B5652C"),
            name="Object Outer Points"
        ))

    return data


def vis_dexhand_optimize_stage2(
    filename: str,
    pc_ls: List[np.ndarray] = None,
    object_mesh_ls: List[List[any]] = None,
    gt_hand_joints: np.ndarray = None,
    hand_mesh_ls: List[List[any]] = None,
    gt_posi_pts: np.ndarray = None,
    posi_pts_ls: np.ndarray = None,
    contact_pt_ls: np.ndarray = None,
    corr_contact_pt_ls: np.ndarray = None,
    penetration_keypoints: np.ndarray = None,
    inner_pen_pts: List[np.ndarray] = None,
    outer_pen_pts: List[np.ndarray] = None,
):
    """
    Creates a comprehensive, multi-frame animation to visualize the results of a DexHand optimization process.

    This function is a specialized tool that aggregates multiple streams of time-series data
    (point clouds, meshes, joints, correspondence points) into a single, interactive HTML
    animation. Each frame in the animation corresponds to a time step in the input data,
    allowing for a clear visual analysis of the optimization's behavior over time.

    The final output is an HTML file with a slider and play/pause buttons to navigate through the frames.

    Args:
        pc_ls (List[np.ndarray]):
            - **Description**: The background point cloud of the scene or object.
            - **Shape**: A list containing one NumPy array of shape `(T, N, 3)`, where `T` is the
              number of frames, and `N` is the number of points.
            - **Visualization**: Rendered as small, semi-transparent, light pink markers.

        gt_hand_joints (np.ndarray):
            - **Description**: The ground truth hand joint locations, typically from a MANO model,
              used as a reference.
            - **Shape**: A NumPy array of shape `(T, K, 3)`, where `K` is the number of joints (e.g., 21).
            - **Visualization**: Rendered as a detailed, multi-colored skeleton using a "Bluered"
              color scale, with gradient opacity along the fingers.

        hand_mesh_ls (List[List[any]]):
            - **Description**: The animated mesh of the robot hand as it changes pose during the optimization.
            - **Shape**: A list containing one inner list of `T` mesh objects (e.g., Open3D TriangleMesh).
              Format: `[[mesh_t0, mesh_t1, ...]]`.
            - **Visualization**: A semi-transparent, skin-colored (`#E1B685`) 3D mesh.

        gt_posi_pts (np.ndarray):
            - **Description**: The ground truth target points for correspondence-based objectives.
              These are the "goal" locations.
            - **Shape**: A NumPy array of shape `(T, P, 3)`, where `P` is the number of correspondence points.
            - **Visualization**: Large green markers. A thin black line connects each of these
              points to its corresponding predicted point in `posi_pts_ls`.

        posi_pts_ls (np.ndarray):
            - **Description**: The corresponding predicted or optimized points on the robot hand's surface.
            - **Shape**: A NumPy array of shape `(T, P, 3)`.
            - **Visualization**: Medium-sized purple markers.

        filename (str):
            - **Description**: The output filename for the HTML animation (e.g., "optimization_result").
              The `.html` suffix will be added automatically.

        contact_pt_ls (np.ndarray, optional):
            - **Description**: The set of contact points located on the robot hand's surface.
            - **Shape**: A NumPy array of shape `(T, C, 3)`, where `C` is the number of contact points.
            - **Visualization**: Medium-sized orange markers. A thin grey line may connect these
              to their corresponding points in `corr_contact_pt_ls`.

        corr_contact_pt_ls (np.ndarray, optional):
            - **Description**: The corresponding contact points on the object's surface, paired one-to-one
              with the points in `contact_pt_ls`.
            - **Shape**: A NumPy array of shape `(T, C, 3)`.
            - **Visualization**: Medium-sized dark blue markers.

        penetration_keypoints (np.ndarray, optional):
            - **Description**: Keypoints on the robot hand used for the self-penetration energy term.
            - **Shape**: A NumPy array of shape `(T, M, 3)`, where `M` is the number of self-penetration points.
            - **Visualization**: Large lime-green (`#ACC313`) markers.

        inner_pen_pts (List[np.ndarray], optional):
            - **Description**: Points from the object that are detected to be in the hand mesh.
              This is useful for visualizing the object-penetration energy term.
            - **Shape**: A List of T NumPy array of shape `(Q, 3)`, where `Q` is the number of inner penetrating points.
            - **Visualization**: Medium-sized dark brown markers.

        outer_pen_pts (List[np.ndarray], optional):
            - **Description**: Points from the object that are near the hand mesh surface.
              This is useful for visualizing the object-penetration energy term.
            - **Shape**: A List of T NumPy array of shape `(R, 3)`, where `R` is the number of outer penetrating points.
            - **Visualization**: Medium-sized light brown markers.
    """
    T = 0
    potential_time_series_args = [
        gt_hand_joints, 
        hand_mesh_ls[0] if hand_mesh_ls is not None else None, 
        gt_posi_pts, 
        posi_pts_ls, 
        contact_pt_ls, 
        corr_contact_pt_ls, 
        penetration_keypoints, 
        inner_pen_pts[0] if inner_pen_pts is not None else None,
        outer_pen_pts[0] if outer_pen_pts is not None else None,
        pc_ls[0] if pc_ls is not None else None,
        object_mesh_ls[0] if object_mesh_ls is not None else None
    ]

    for arg in potential_time_series_args:
        if arg is not None:
            try:
                T = len(arg)
                if T > 0: break
            except TypeError: continue
    
    if T == 0:
        print("Warning: Could not determine the number of frames (T) from the provided arguments, or T is 0. No animation will be generated.")
        return

    # --- Create Initial Frame (t=0) ---
    initial_data = _vis_dexhand_optimize_frame(
        pc_ls=get_subitem(pc_ls, 0) if pc_ls is not None else None,
        gt_joints=gt_hand_joints[0] if gt_hand_joints is not None else None,
        hand_mesh=get_subitem(hand_mesh_ls, 0) if hand_mesh_ls is not None else None,
        gt_posi=gt_posi_pts[0] if gt_posi_pts is not None else None,
        posi=posi_pts_ls[0] if posi_pts_ls is not None else None,
        obj_mesh=get_subitem(object_mesh_ls, 0) if object_mesh_ls is not None else None,
        contact_pt=contact_pt_ls[0] if contact_pt_ls is not None else None,
        corr_contact_pt=corr_contact_pt_ls[0] if corr_contact_pt_ls is not None else None,
        penetration_keypoints=penetration_keypoints[0] if penetration_keypoints is not None else None,
        inner_pen_pts=inner_pen_pts[0] if inner_pen_pts is not None else None,
        outer_pen_pts=outer_pen_pts[0] if outer_pen_pts is not None else None
    )

    # --- Create All Animation Frames ---
    frames = []
    for t in tqdm(range(T), desc="Creating optimization frames"):
        frame_data = _vis_dexhand_optimize_frame(
            pc_ls=get_subitem(pc_ls, t) if pc_ls is not None else None,
            gt_joints=gt_hand_joints[t] if gt_hand_joints is not None else None,
            hand_mesh=get_subitem(hand_mesh_ls, t) if hand_mesh_ls is not None else None,
            gt_posi=gt_posi_pts[t] if gt_posi_pts is not None else None,
            posi=posi_pts_ls[t] if posi_pts_ls is not None else None,
            obj_mesh=get_subitem(object_mesh_ls, t) if object_mesh_ls is not None else None,
            contact_pt=contact_pt_ls[t] if contact_pt_ls is not None else None,
            corr_contact_pt=corr_contact_pt_ls[t] if corr_contact_pt_ls is not None else None,
            penetration_keypoints=penetration_keypoints[t] if penetration_keypoints is not None else None,
            inner_pen_pts=inner_pen_pts[t] if inner_pen_pts is not None else None,
            outer_pen_pts=outer_pen_pts[t] if outer_pen_pts is not None else None
        )
        frames.append(go.Frame(data=frame_data, name=f"Frame {t}"))

    # --- Setup Layout, Slider, and Buttons ---
    slider_steps = [{"method": "animate", "label": f"{t}", "args": [[f"Frame {t}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]} for t in range(T)]
    layout = go.Layout(
        title="DexHand Optimization Process",
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        updatemenus=[{"type": "buttons", "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
        ]}],
        sliders=[{"active": 0, "yanchor": "top", "xanchor": "left", "currentvalue": {"font": {"size": 20}, "prefix": "Frame:", "visible": True, "xanchor": "right"}, "transition": {"duration": 0}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0, "steps": slider_steps}]
    )

    # --- Create and Save Figure ---
    fig = go.Figure(data=initial_data, layout=layout, frames=frames)
    fig.write_html(f"{filename}.html")
    print(f"Optimization visualization saved to {filename}.html")

