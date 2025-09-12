import numpy as np
import torch
import os
from typing import List, Dict, Optional, Tuple, Union
import warnings
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pytorch3d.ops import sample_farthest_points, ball_query
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (
    quaternion_to_matrix, 
    matrix_to_quaternion, 
    axis_angle_to_quaternion, 
    quaternion_to_axis_angle
)
from pytransform3d import transformations as pt
from plotly.colors import get_colorscale
import plotly.graph_objects as go
import plotly.offline as pyo
import open3d as o3d
from manotorch.manolayer import ManoLayer
# from manopth.manolayer import ManoLayer

from .tools import (cosine_similarity, 
                    farthest_point_sampling, 
                    extract_hand_points_and_mesh,
                    get_point_clouds_from_human_data,
                    get_object_meshes_from_human_data,
                    apply_transformation_human_data,
                    apply_transformation_on_object_mesh)
from .hand_model import load_robot

def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(
            pixel_scaled.shape[1], size=(subsample_to,), replace=False
        )
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def plot_points(pts: np.ndarray, colors: np.ndarray) -> go.Scatter3d:
    """Create a 3D scatter plot for point cloud visualization."""
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1], 
        z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=2.5,
            color=colors,
            opacity=1
        )
    )



def visualize_pointclouds_and_mesh(
    pointcloud_list: List[np.ndarray],
    color_list: List[np.ndarray],
    mesh: Optional[Union[Meshes, Dict]] = None,
    view_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    point_size: float = 2.0,
    mesh_opacity: float = 0.3,
    mesh_color: str = 'lightgray',
    max_points_per_cloud: int = 5000,
    auto_open: bool = False,
    title: str = "Multi-Viewpoint Point Cloud and Mesh Visualization"
) -> Optional[str]:
    """
    Unified visualization function for multiple point clouds and an optional mesh.
    
    This function is designed to test multi-camera calibration by visualizing point clouds
    from different camera viewpoints along with the original mesh to check alignment.
    
    Args:
        pointcloud_list: List of point cloud arrays, each shape (N, 3)
        color_list: List of color arrays for each point cloud:
                   - RGB colors: shape (N, 3) with values in [0, 1] or [0, 255]
                   - Feature colors: shape (N, 3) from PCA-reduced features, normalized
                   - Any other 3D tensor representation
        mesh: Optional mesh to overlay:
              - PyTorch3D Meshes object, or
              - Dict with 'vertices' and 'faces' keys
        view_names: Optional names for each viewpoint (e.g., ['View 0', 'View 1', ...])
        save_path: Path to save visualization (HTML for Plotly, screenshot for Open3D)
        point_size: Size of points in visualization
        mesh_opacity: Transparency of mesh (0.0 to 1.0)
        mesh_color: Color of mesh ('lightgray', 'white', etc.)
        max_points_per_cloud: Maximum points per cloud for performance
        use_plotly: If True, use Plotly (SSH-compatible). If False, use Open3D
        auto_open: Whether to automatically open the visualization
        title: Title for the visualization
        
    Returns:
        Path to saved visualization file if save_path is provided, None otherwise
        
    Example:
        # Test 4-camera calibration
        pointclouds = [pc1, pc2, pc3, pc4]  # Point clouds from 4 viewpoints
        colors = [rgb1, rgb2, rgb3, rgb4]   # RGB or feature colors
        
        visualize_pointclouds_and_mesh(
            pointcloud_list=pointclouds,
            color_list=colors,
            mesh=original_mesh,
            view_names=['Camera 1', 'Camera 2', 'Camera 3', 'Camera 4'],
            save_path='calibration_test.html',
            title='Four-Camera Calibration Test'
        )
    """
    
    # Input validation
    if not pointcloud_list:
        raise ValueError("pointcloud_list cannot be empty")
    
    if len(pointcloud_list) != len(color_list):
        raise ValueError("pointcloud_list and color_list must have same length")
    
    # Generate view names if not provided
    if view_names is None:
        view_names = [f'View {i}' for i in range(len(pointcloud_list))]
    elif len(view_names) != len(pointcloud_list):
        raise ValueError("view_names must have same length as pointcloud_list")
    
    print(f"Creating visualization with {len(pointcloud_list)} point clouds...")
    
    return _create_plotly_visualization(
        pointcloud_list, color_list, mesh, view_names, save_path,
        point_size, mesh_opacity, mesh_color, max_points_per_cloud,
        auto_open, title
    )


def _create_plotly_visualization(
    pointcloud_list: List[np.ndarray],
    color_list: List[np.ndarray],
    mesh: Optional[Union[Meshes, Dict]],
    view_names: List[str],
    save_path: Optional[str],
    point_size: float,
    mesh_opacity: float,
    mesh_color: str,
    max_points_per_cloud: int,
    auto_open: bool,
    title: str
) -> Optional[str]:
    """Create Plotly-based visualization (SSH-compatible)"""
    
    traces = []
    
    # Color palette for different views
    view_colors = [
        'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'
    ]
    
    # Add point cloud traces
    for i, (points, colors, view_name) in enumerate(zip(pointcloud_list, color_list, view_names)):
        if len(points) == 0:
            print(f"Warning: {view_name} has no points, skipping")
            continue
            
        # Subsample points for performance
        if len(points) > max_points_per_cloud:
            indices = np.random.choice(len(points), max_points_per_cloud, replace=False)
            points_sub = points[indices]
            colors_sub = colors[indices]
        else:
            points_sub = points
            colors_sub = colors
        
        # Process colors
        if colors_sub.ndim == 2 and colors_sub.shape[1] == 3:
            # RGB or feature colors
            if colors_sub.max() <= 1.0:
                # Assume [0, 1] range, convert to [0, 255]
                colors_rgb = (colors_sub * 255).astype(int)
            else:
                # Assume [0, 255] range
                colors_rgb = colors_sub.astype(int)
            
            # Create RGB strings
            rgb_strings = [f'rgb({r},{g},{b})' for r, g, b in colors_rgb]
        else:
            # Fallback to view-specific color
            rgb_strings = view_colors[i % len(view_colors)]
        
        # Create scatter trace
        trace = go.Scatter3d(
            x=points_sub[:, 0],
            y=points_sub[:, 1],
            z=points_sub[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=rgb_strings,
                opacity=0.8
            ),
            name=view_name,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<extra></extra>'
        )
        traces.append(trace)
        print(f"Added {len(points_sub)} points for {view_name}")
    
    # Add mesh trace if provided
    if mesh is not None:
        try:
            vertices, faces = _extract_mesh_data(mesh)
            
            mesh_trace = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=mesh_opacity,
                color=mesh_color,
                name='Mesh',
                showscale=False
            )
            traces.append(mesh_trace)
            print("Added mesh to visualization")
        except Exception as e:
            print(f"Warning: Could not add mesh to visualization: {e}")
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            bgcolor="rgb(240, 240, 240)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'  # Maintain equal aspect ratio
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    # Save or display
    if save_path:
        if not save_path.endswith('.html'):
            save_path = save_path.replace('.png', '.html').replace('.jpg', '.html')
            if not save_path.endswith('.html'):
                save_path += '.html'
        
        pyo.plot(fig, filename=save_path, auto_open=auto_open)
        print(f"✓ Interactive visualization saved to: {save_path}")
        return save_path
    else:
        pyo.plot(fig, auto_open=auto_open)
        return None


def _extract_mesh_data(mesh: Union[Meshes, Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract vertices and faces from mesh object"""
    
    if isinstance(mesh, dict):
        # Dictionary format
        vertices = mesh['vertices']
        faces = mesh['faces']
        
        if torch.is_tensor(vertices):
            vertices = vertices.cpu().numpy()
        if torch.is_tensor(faces):
            faces = faces.cpu().numpy()
            
        return vertices, faces
    
    elif isinstance(mesh, Meshes):
        # PyTorch3D Meshes format
        vertices = mesh.verts_list()[0].cpu().numpy()
        faces = mesh.faces_list()[0].cpu().numpy()
        return vertices, faces
    
    else:
        raise ValueError("Unsupported mesh format. Use PyTorch3D Meshes or dict with 'vertices' and 'faces' keys")



def get_vis_hand_keypoints_with_color_gradient_and_lines(gt_posi_pts: np.ndarray, color_scale='Viridis', finger_groups=None, emphasize_idx=None):
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
                    marker=dict(size=15, color='green', opacity=1), 
                    name=f"{finger_name} {j+1}",
                    showlegend=False
                ))
            else:
                data.append(go.Scatter3d(
                    x=[gt_posi_pts[idx, 0]], 
                    y=[gt_posi_pts[idx, 1]], 
                    z=[gt_posi_pts[idx, 2]], 
                    mode='markers', 
                    marker=dict(size=10, color=finger_color[1], opacity=opacity), 
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
                    line=dict(color=finger_color[1], width=10),
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
                line=dict(color=finger_color[1], width=12,), 
                name=f"Root to {finger_name}",
                showlegend=False
            ))
    return data



def pt_transform(points, transformation):
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation @ points_homogeneous.T).T
    return transformed_points[:, :3]



def get_subitem(ls:List[np.ndarray], idx:int, not_list=False):
    if ls is None:
        return None
    if type(ls) is not list or not_list:
        return ls[idx]
    return [item[idx] for item in ls]



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
    filename: str = None,
):
    """
    visulize everything as frames in plotly
    """
    print(len(gt_hand_joints), gt_hand_joints[0].shape)

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
    )

    frames = []
    for t in range(T):
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
    pc_ls: List[np.ndarray] = None,
    hand_pts_ls: List[np.ndarray] = None,
    transformation_ls: List[np.ndarray] = None,
    gt_transformation_ls: List[np.ndarray] = None,
    gt_posi_pts: np.ndarray = None,
    posi_pts_ls: List[np.ndarray] = None,
    hand_joints_ls: List[np.ndarray] = None,
    gt_hand_joints: np.ndarray = None,
    opt_points=None,
    gt_opt_points=None,
    voxel_dict=None,
    hand_mesh=None,
    hand_mesh_ls=None,
    hand_name_ls: List[str] = None,
    show_axis: bool = False,
    obj_mesh=None,
    obj_mesh_ls=None,
    obj_norm_ls=None,
    return_data: bool = False,
    filename: str = None,
) -> None:
    """
    visulize point clouds and hand mesh in plotly.
    input is a list with time axis.

    Args:
        pc_ls (List[np.ndarray]): list of point clouds (pink and purple medium points)
        hand_pts_ls (List[np.ndarray]): list of hand points (orange small points)
        transformation_ls (List[np.ndarray]): list of transformations (small coordinate frames)
        gt_transformation_ls (List[np.ndarray]): list of ground truth transformations (big coordinate frames)
        gt_posi_pts (np.ndarray): ground truth position points (red big points)
        posi_pts_ls (List[np.ndarray]): list of position points (blue and green big points)
        hand_mesh (trimesh): hand mesh (royalblue)
    """
    # Define colors
    red = "rgb(255, 0, 0)"
    green = "rgb(0, 255, 0)"
    blue = "rgb(0, 0, 255)"
    posi_pts_colors = [blue, green]

    # Set interpolated color gradient for points in time series
    if pc_ls is not None:
        color_gradient_points = plt.get_cmap("Purples")(np.linspace(0.3, 0.8, len(pc_ls)))
    if hand_pts_ls is not None:
        color_gradient_hands = plt.get_cmap("Oranges")(np.linspace(0.3, 0.8, len(hand_pts_ls)))
    if hand_mesh_ls is not None:
        color_gradient_hand_mesh = plt.get_cmap("Reds")(np.linspace(0.3, 0.8, len(hand_mesh_ls)))
    if obj_mesh_ls is not None:
        color_gradient_obj_mesh = plt.get_cmap("Blues")(np.linspace(0.3, 0.8, len(obj_mesh_ls)))

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
                    line=dict(color=colors[i], width=12),
                    opacity=opacity,
                    name=name,
                )
            )
        return lines

    if voxel_dict is not None:
        grid_centers = voxel_dict["grid_centers"]
        selected_points = voxel_dict.get("selected_points", np.empty((0, 3)))
        vis_empty = voxel_dict.get("vis_empty", True)
        auto_calc_size = voxel_dict.get("auto_calculate_size", True)
        voxel_size = voxel_dict.get("voxel_size", (1.0, 1.0, 1.0))

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
            voxel_size = (dx, dy, dz)

        for center in grid_centers:
            cx, cy, cz = center
            is_selected = any(np.all(center == sp) for sp in selected_points)
            x_edges = [cx - voxel_size[0] / 2, cx + voxel_size[0] / 2]
            y_edges = [cy - voxel_size[1] / 2, cy + voxel_size[1] / 2]
            z_edges = [cz - voxel_size[2] / 2, cz + voxel_size[2] / 2]

            if is_selected:
                data.append(
                    go.Mesh3d(
                        x=[x_edges[0],x_edges[1],x_edges[1],x_edges[0],x_edges[0],x_edges[1],x_edges[1],x_edges[0]],
                        y=[y_edges[0],y_edges[0],y_edges[1],y_edges[1],y_edges[0],y_edges[0],y_edges[1],y_edges[1]],
                        z=[z_edges[0],z_edges[0],z_edges[0],z_edges[0],z_edges[1],z_edges[1],z_edges[1],z_edges[1]],
                        color="blue", opacity=0.5, alphahull=0
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
                        line=dict(color="lightgray", width=1), showlegend=False
                    )
                )

    if transformation_ls is not None:
        for i, transformation in enumerate(transformation_ls):
            data.extend(add_coordinate_frame(0.02, 1, transformation, name=f"Transformation {i+1}"))

    if gt_transformation_ls is not None:
        for i, transformation in enumerate(gt_transformation_ls):
            data.extend(add_coordinate_frame(0.02, 0.4, transformation, name=f"GT Transformation {i+1}"))

    if pc_ls is not None:
        for i, pc in enumerate(pc_ls):
            color = f"rgb({int(color_gradient_points[i][0]*255)}, {int(color_gradient_points[i][1]*255)}, {int(color_gradient_points[i][2]*255)})"
            data.append(
                go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode="markers",
                             marker=dict(size=5, color=color), name=f"Point Cloud {i+1}")
            )

    if hand_pts_ls is not None:
        for i, hand_pts in enumerate(hand_pts_ls):
            color = f"rgb({int(color_gradient_hands[i][0]*255)}, {int(color_gradient_hands[i][1]*255)}, {int(color_gradient_hands[i][2]*255)})"
            data.append(
                go.Scatter3d(x=hand_pts[:, 0], y=hand_pts[:, 1], z=hand_pts[:, 2], mode="markers",
                             marker=dict(size=3, color=color), name=f"Hand Points {i+1}")
            )

    if gt_posi_pts is not None:
        data.append(
            go.Scatter3d(x=gt_posi_pts[:, 0], y=gt_posi_pts[:, 1], z=gt_posi_pts[:, 2],
                         mode="markers", marker=dict(size=10, color=red), name="GT Position Points")
        )

    if posi_pts_ls is not None:
        for i, posi_pts in enumerate(posi_pts_ls):
            data.append(
                go.Scatter3d(x=posi_pts[:, 0], y=posi_pts[:, 1], z=posi_pts[:, 2], mode="markers",
                             marker=dict(size=10, color="orange"), name=f"Position Points {i+1}")
            )

    if opt_points is not None:
        finger_groups = {"thumb": [0, 1], "index": [2], "middle": [3], "ring": [4], "pinky": [5]}
        vis_data = get_vis_hand_keypoints_with_color_gradient_and_lines(
            opt_points[[5, 0, 1, 2, 3, 4]], color_scale="Viridis", finger_groups=finger_groups
        )
        data.extend(vis_data)

    if gt_opt_points is not None:
        finger_groups = {"thumb": [0, 1], "index": [2], "middle": [3], "ring": [4], "pinky": [5]}
        vis_data = get_vis_hand_keypoints_with_color_gradient_and_lines(
            gt_opt_points[[5, 0, 1, 2, 3, 4]], color_scale="Bluered", finger_groups=finger_groups
        )
        data.extend(vis_data)

    if gt_hand_joints is not None:
        data.extend(get_vis_hand_keypoints_with_color_gradient_and_lines(gt_hand_joints, color_scale="Bluered"))

    if hand_joints_ls is not None:
        for i, hand_joints in enumerate(hand_joints_ls):
            hand_joints_data = get_vis_hand_keypoints_with_color_gradient_and_lines(hand_joints, color_scale="Plasma")
            data.extend(hand_joints_data)

    if hand_mesh is not None:
        verts = np.asarray(hand_mesh.vertices)
        faces = np.asarray(hand_mesh.triangles if hasattr(hand_mesh, "triangles") else hand_mesh.faces)
        data.append(
            go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                      i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                      color="royalblue", opacity=0.5, name="Hand Mesh")
        )

    if obj_mesh is not None:
        for i, mesh_item in enumerate(obj_mesh):
            if mesh_item is None or not hasattr(mesh_item, "vertices"): continue
            verts = np.asarray(mesh_item.vertices)
            faces = np.asarray(mesh_item.triangles if hasattr(mesh_item, "triangles") else mesh_item.faces)
            if verts.size == 0 or faces.size == 0: continue
            data.append(
                go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                          i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                          color="#D3D3D3", opacity=1, name=f"Object Mesh {i}")
            )

    if obj_mesh_ls is not None:
        for i, obj_mesh_item in enumerate(obj_mesh_ls):
            verts = np.asarray(obj_mesh_item.vertices)
            faces = np.asarray(obj_mesh_item.triangles if hasattr(obj_mesh_item, "triangles") else obj_mesh_item.faces)
            color = f"rgb({int(color_gradient_obj_mesh[i][0]*255)}, {int(color_gradient_obj_mesh[i][1]*255)}, {int(color_gradient_obj_mesh[i][2]*255)})"
            data.append(
                go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            color=color)
            )

    if obj_norm_ls is not None:
        scale = 0.02
        for i, (pc, normals) in enumerate(zip(pc_ls, obj_norm_ls)):
            color = f"rgb({int(color_gradient_points[i][0]*255)}, {int(color_gradient_points[i][1]*255)}, {int(color_gradient_points[i][2]*255)})"
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
                             line=dict(color=color, width=2), name=f"Object Normals {i+1}")
            )

    if hand_mesh_ls is not None:
        for i, hand_mesh_item in enumerate(hand_mesh_ls):
            verts = np.asarray(hand_mesh_item.vertices)
            faces = np.asarray(hand_mesh_item.triangles if hasattr(hand_mesh_item, "triangles") else hand_mesh_item.faces)
            color = f"rgb({int(color_gradient_hand_mesh[i][0]*255)}, {int(color_gradient_hand_mesh[i][1]*255)}, {int(color_gradient_hand_mesh[i][2]*255)})"
            
            # Determine the name for the hand mesh from the provided list
            mesh_name = f"Hand Mesh {i}"  # Default/fallback name
            if hand_name_ls is not None and i < len(hand_name_ls):
                mesh_name = hand_name_ls[i]

            data.append(
                go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                          i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                          color=color, opacity=0.5, name=mesh_name, showlegend=True)
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
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", title="X"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", title="Y"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", title="Z"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        if filename is not None:
            fig.write_html(f"{filename}.html")
        else:
            fig.show()



def visualize_human_sequence(seq_data, filename: Optional[str] = None):
    
    ##  Visualize Object Point Clouds ##
    pc = get_point_clouds_from_human_data(seq_data)
    pc_ls = apply_transformation_human_data(pc, seq_data["obj_poses"])
    pc_ls = [np.asarray(pc_ls)] # due to some dim issue

    ## Extract Hand Points and Mesh ##
    mano_hand_joints, hand_verts = extract_hand_points_and_mesh(seq_data["hand_tsls"], seq_data["hand_coeffs"], seq_data["side"])

    # Visualize using vis_frames_plotly
    vis_frames_plotly(
        pc_ls=pc_ls, # should be a list, len 1, (T, N, 3)
        gt_hand_joints=mano_hand_joints, # should be a tensor, (T, 21, 3)
        show_axis=True,
        filename=filename if filename else None
    )


def visualize_human_sequence2(seq_data, filename: Optional[str] = None):
    
    ##  Visualize Object Point Clouds ##
    pc = get_point_clouds_from_human_data(seq_data)
    # pc = seq_data['object_points_ls']
    poses = seq_data["obj_poses"] @ seq_data['mesh_norm_trans']

    pc_ls = apply_transformation_human_data(pc, poses)
    pc_ls = [np.asarray(pc_ls)] # due to some dim issue

    ## Extract Hand Points and Mesh ##
    mano_hand_joints, hand_verts = extract_hand_points_and_mesh(seq_data["hand_tsls"], seq_data["hand_coeffs"], seq_data["side"])
    # mano_hand_joints = seq_data['hand_joints']
    print("Hi")

    # Visualize using vis_frames_plotly
    vis_frames_plotly(
        pc_ls=pc_ls, # should be a list, len 1, (T, N, 3)
        gt_hand_joints=mano_hand_joints, # should be a tensor, (T, 21, 3)
        show_axis=True,
        filename=filename if filename else None
    )



def visualize_dex_hand_sequence(seq_data, filename: Optional[str] = None):
    """
    Visualize a sequence of dexterous hand movements.
    
    Args:
        seq_data (DexHandSequenceData): The sequence data containing hand poses and meshes.
        filename (Optional[str]): If provided, save the visualization to this file.
    """
    data_id = seq_data["which_dataset"] + "_" + seq_data["which_sequence"]
    hand_type = 'left' if seq_data["side"] == 0 else 'right'
    robot_name_str = seq_data["which_hand"]
    fps = 10

    robot = load_robot(robot_name_str, hand_type)

    ### robot hand meshes ###
    hand_meshes = []
    for i in tqdm(range(seq_data["hand_poses"].shape[0])):
        robot.set_qpos(seq_data["hand_poses"][i])
        hand_mesh = robot.get_hand_mesh()
        hand_meshes.append(hand_mesh)

    ### hand joints ###
    mano_hand_joints, hand_verts = extract_hand_points_and_mesh(seq_data["hand_tsls"], seq_data["hand_coeffs"], seq_data["side"])
    # mano_hand_joints = seq_data["hand_joints"] # Tx21x3

    ### point clouds ###
    pc = get_point_clouds_from_human_data(seq_data)
    pc_ls = apply_transformation_human_data(pc, seq_data["obj_poses"]) # TxNx3
    pc_ls = [np.asarray(pc_ls)] # due to some dim issue
    object_mesh = get_object_meshes_from_human_data(seq_data)   
    object_ls = apply_transformation_on_object_mesh(object_mesh, seq_data["obj_poses"]) # a list of (B) a list of (T) meshes

    vis_frames_plotly(
        pc_ls=pc_ls,
        gt_hand_joints=mano_hand_joints,
        hand_mesh=hand_meshes,
        object_mesh_ls=object_ls,
        show_axis=True,
        filename=filename
    )    


def vis_single_frame_with_offset(pc: np.ndarray, gt_hand_joints: torch.Tensor, 
                       filename: str = None, 
                       row: int = 0, column: int = 0, padding: float = 0.0) -> None:
    """
    在Plotly中可视化点云和手部关节点（单帧）。

    Args:
        pc (np.ndarray): 点云数组 (N, 3)。
        gt_hand_joints (torch.Tensor): 真实手部关节点 (21, 3)。
        filename (str, optional): 如果提供，则将图表保存为HTML文件。默认为None，即直接显示图表。
        row (int, optional): 网格布局的行索引，用于计算Y轴偏移。默认为0。
        column (int, optional): 网格布局的列索引，用于计算X轴偏移。默认为0。
        padding (float, optional): 网格布局中每个单元的间距/大小，用于计算偏移量。默认为0.0。
    """
    data = []

    # (1) 根据row, column, padding计算整体偏移量
    # 假设偏移主要在X-Y平面上
    offset = np.array([column * padding, row * padding, 0])

    # (2) 处理点云 (pc_ls -> pc)
    if pc is not None:
        # 应用偏移量
        pc_offset = pc + offset
        data.append(go.Scatter3d(
            x=pc_offset[:, 0], 
            y=pc_offset[:, 1], 
            z=pc_offset[:, 2], 
            mode='markers', 
            marker=dict(size=5, color='lightpink'), 
        ))

    # (3) 处理手部关节点 (gt_hand_joints)
    if gt_hand_joints is not None:
        # 将torch.tensor转换为numpy.array
        hand_joints_np = gt_hand_joints.cpu().numpy()
        # 应用偏移量
        hand_joints_offset = hand_joints_np + offset
        
        # 使用辅助函数生成关节点和骨架的可视化轨迹
        hand_joints_data = get_vis_hand_keypoints_with_color_gradient_and_lines(hand_joints_offset, color_scale='Bluered')
        data.extend(hand_joints_data)

    return data


def vis_grid(seq_data_ls, filename):
    """
    高效地在网格中可视化多个点云和手部姿态。
    所有点云合并为一个trace，所有手部关节点合并为一个trace，
    所有手部骨架合并为一个trace，以实现高性能渲染。
    """
    import plotly
    # 1. 定义手部关节点分组，这部分逻辑现在移到 vis_grid 内部或作为全局常量
    FINGER_GROUPS_21 = {
        'wrist': [0], # 单独处理手腕
        'thumb': [0, 1, 2, 3, 4],
        'index': [0, 5, 6, 7, 8],
        'middle': [0, 9, 10, 11, 12],
        'ring': [0, 13, 14, 15, 16],
        'pinky': [0, 17, 18, 19, 20]
    }

    # 2. 定义颜色映射
    # 使用Plotly的内置色板为每个手指分配一个固定的颜色
    finger_color_map = {
        'wrist': 'grey',
        'thumb': plotly.colors.qualitative.Plotly[0],
        'index': plotly.colors.qualitative.Plotly[1],
        'middle': plotly.colors.qualitative.Plotly[2],
        'ring': plotly.colors.qualitative.Plotly[3],
        'pinky': plotly.colors.qualitative.Plotly[4],
    }

    def apply_transformation_to_vertices(vertices, transform_matrix):
        """将4x4变换矩阵应用于 N*3 的顶点云"""
        vertices_h = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        transformed_vertices_h = (transform_matrix @ vertices_h.T).T
        return transformed_vertices_h[:, :3]

    # === 1. 初始化用于聚合所有数据的容器 ===
    all_pc_points = []
    all_mesh_x, all_mesh_y, all_mesh_z = [], [], []
    all_mesh_i, all_mesh_j, all_mesh_k = [], [], []

    # 手部关节点 (Markers)
    all_joint_x, all_joint_y, all_joint_z = [], [], []
    all_joint_colors = []
    
    # 手部骨架 (Lines)
    all_line_x, all_line_y, all_line_z = [], [], []

    row_total = int(np.sqrt(len(seq_data_ls)))
    col_total = int(np.ceil(len(seq_data_ls) / row_total))
    padding = 0.25
    vertex_offset = 0

    print("Aggregating data for visualization...")
    for iid, seq_data in tqdm(enumerate(seq_data_ls)):
        column = iid % col_total
        row = iid // col_total
        offset = np.array([column * padding, row * padding, 0])
        
        # --- 模拟数据加载，请替换为您自己的函数 ---
        poses = seq_data["obj_poses"] @ seq_data['mesh_norm_trans']
        # pc = get_point_clouds_from_human_data(seq_data, ds_num=600)
        # pc_ls = apply_transformation_human_data(pc, poses)

        for obj_ii, mesh_path in enumerate(seq_data["object_mesh_path"]):
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            target_faces = 100  # 目标面数
            if len(faces) > target_faces:
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
            transformed_verts = apply_transformation_to_vertices(verts*0.01, poses[obj_ii, 60])
            transformed_verts += offset

            all_mesh_x.extend(transformed_verts[:, 0])
            all_mesh_y.extend(transformed_verts[:, 1])
            all_mesh_z.extend(transformed_verts[:, 2])

            all_mesh_i.extend(faces[:, 0] + vertex_offset)
            all_mesh_j.extend(faces[:, 1] + vertex_offset)
            all_mesh_k.extend(faces[:, 2] + vertex_offset)

            # 更新顶点偏移量，为下一个网格的合并做准备
            vertex_offset += len(verts)

        mano_hand_joints = seq_data['hand_joints']
        # --- 数据加载结束 ---

        # (2.1) 处理并聚合点云数据
        # pc_offset = pc_ls[60] + offset
        # all_pc_points.append(pc_offset)

        # (2.2) 处理并聚合手部姿态数据
        hand_joints_np = mano_hand_joints[60].cpu().numpy()
        hand_joints_offset = hand_joints_np + offset
        
        # --- 内联 get_vis_hand_keypoints_with_color_gradient_and_lines 的逻辑 ---
        
        # 添加关节点坐标和颜色到聚合列表
        for finger_name, indices in FINGER_GROUPS_21.items():
            if finger_name == 'wrist': # 单独处理手腕点
                idx = indices[0]
                all_joint_x.append(hand_joints_offset[idx, 0])
                all_joint_y.append(hand_joints_offset[idx, 1])
                all_joint_z.append(hand_joints_offset[idx, 2])
                all_joint_colors.append(finger_color_map['wrist'])
                continue

            # 为每个手指内的关节点（除手腕）添加数据
            for idx in indices[1:]: # 跳过手腕点，因为它已被处理
                all_joint_x.append(hand_joints_offset[idx, 0])
                all_joint_y.append(hand_joints_offset[idx, 1])
                all_joint_z.append(hand_joints_offset[idx, 2])
                all_joint_colors.append(finger_color_map[finger_name])

            # 添加连接手指内部关节点的骨架线条
            for i in range(len(indices) - 1):
                p1_idx, p2_idx = indices[i], indices[i+1]
                p1 = hand_joints_offset[p1_idx]
                p2 = hand_joints_offset[p2_idx]
                
                all_line_x.extend([p1[0], p2[0], None]) # 'None' 用于断开线条
                all_line_y.extend([p1[1], p2[1], None])
                all_line_z.extend([p1[2], p2[2], None])

    # === 3. 循环结束后，根据聚合数据创建高效的 Plotly Traces ===
    
    data = []
    
    # (3.1) 创建点云的 Trace
    # 首先将点云列表合并成一个大的Numpy数组
    # all_pcs_np = np.concatenate(all_pc_points, axis=0)
    # data.append(go.Scatter3d(
    #     x=all_pcs_np[:, 0], 
    #     y=all_pcs_np[:, 1], 
    #     z=all_pcs_np[:, 2], 
    #     mode='markers', 
    #     marker=dict(size=5, color='lightpink', opacity=0.7), 
    # ))

    # (3.2) 创建手部骨架线条的 Trace
    data.append(go.Scatter3d(
        x=all_line_x,
        y=all_line_y,
        z=all_line_z,
        mode='lines',
        line=dict(color='grey', width=10),
    ))

    # (3.3) 创建手部关节点的 Trace
    data.append(go.Scatter3d(
        x=all_joint_x,
        y=all_joint_y,
        z=all_joint_z,
        mode='markers',
        marker=dict(
            size=5,
            color=all_joint_colors, # 关键：传入颜色数组实现逐点着色
        ),
    ))

    # (3.4) 创建所有对象网格的合并 Trace
    data.append(go.Mesh3d(
        x=all_mesh_x, y=all_mesh_y, z=all_mesh_z,
        i=all_mesh_i, j=all_mesh_j, k=all_mesh_k,
        color='saddlebrown',
        opacity=0.7, # 可以稍微调高不透明度，使其更有实体感
        # 添加光照效果可以让网格看起来更立体
        lighting=dict(ambient=0.4, diffuse=1.0, specular=0.5),
        lightposition=dict(x=100, y=200, z=0)
    ))

    tri_points = np.array([all_mesh_x, all_mesh_y, all_mesh_z]).T
    tri_indices = np.array([all_mesh_i, all_mesh_j, all_mesh_k]).T
    
    edge_x, edge_y, edge_z = [], [], []
    for i, j, k in tri_indices:
        p0, p1, p2 = tri_points[i], tri_points[j], tri_points[k]
        edge_x.extend([p0[0], p1[0], p1[0], p2[0], p2[0], p0[0], None])
        edge_y.extend([p0[1], p1[1], p1[1], p2[1], p2[1], p0[1], None])
        edge_z.extend([p0[2], p1[2], p1[2], p2[2], p2[2], p0[2], None])

    data.append(go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='black', width=0.5), # 黑色或深灰色的细线效果最好
        showlegend=False # 通常我们不需要为边线单独显示图例
    ))

    # === 4. 创建并显示/保存图表 ===
    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"Grid Visualization of {len(seq_data_ls)} items",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data' # 保持XYZ轴的比例一致
        ),
        legend_orientation="h",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    if filename is not None:
        fig.write_html(f'{filename}.html')
        print(f"Figure saved to {filename}.html")
    else:
        fig.show()


def visualize_dex_hand_sequence_together(seq_data_ls, name_list, filename: Optional[str] = None):

    seq_data = seq_data_ls[0]
    pc = get_point_clouds_from_human_data(seq_data)
    # pc_ls = apply_transformation_human_data(pc, seq_data["obj_poses"] @ seq_data['mesh_norm_trans']) # TxNx3
    pc_ls = apply_transformation_human_data(pc, seq_data["obj_poses"]) # TxNx3
    pc_ls = [np.asarray(pc_ls)] # due to some dim issue
    object_mesh = get_object_meshes_from_human_data(seq_data)   
    # object_ls = apply_transformation_on_object_mesh(object_mesh, seq_data["obj_poses"] @ seq_data['mesh_norm_trans']) # a list of (B) a list of (T) meshes
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
        pc_ls=pc_ls,
        # object_mesh_ls=object_ls,
        hand_mesh_ls=hand_mesh_ls,
        hand_name_ls=name_list,
        gt_hand_joints=mano_hand_joints,
        filename=filename
    )