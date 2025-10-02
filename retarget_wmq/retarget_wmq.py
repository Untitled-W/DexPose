import torch
import numpy as np
import joblib
import os
from matplotlib import pyplot as plt

from .robot_wrapper import load_robot, HandRobotWrapper
# from utils.hand_model import load_robot

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pytorch3d.transforms.rotation_conversions")

print_once = (lambda: (lambda num=[0]: lambda *a, **k: (print(*a, **k), num.__setitem__(0, 1)) if num[0] == 0 else None)())

def visualize_time_series_with_fill_between(losses_all_dict: dict, title: str = "Loss Distribution Over Timesteps", filename: str = "Loss_Distribution.png"):
    """
    Visualizes multiple time series using fill_between to represent the spread
    of N items at each timestep, with opacity diminishing for values further from 1.0.

    Args:
        losses_all_dict (dict): A dictionary where keys are loss types (e.g., 'total_loss')
                                and values are lists of lists.
                                Outer list (T elements): Represents timesteps.
                                Inner list (N elements): Represents individual loss values at that timestep.
        title (str): The title for the plot.
    """

    fig, axes = plt.subplots(nrows=len(losses_all_dict), ncols=1, figsize=(12, 5 * len(losses_all_dict)), squeeze=False)
    fig.suptitle(title, fontsize=16)

    for i, (key, i_data_list) in enumerate(losses_all_dict.items()):
        ax = axes[i, 0]

        # Store statistics for each timestep
        means = []
        stds = []
        quantiles_25 = []
        quantiles_75 = []
        mins = []
        maxs = []

        current_values = np.array(i_data_list) # N x T
        means = np.mean(current_values, axis=0)
        stds = np.std(current_values, axis=0)
        quantiles_25 = np.percentile(current_values, 25, axis=0)
        quantiles_75 = np.percentile(current_values, 75, axis=0)
        mins = np.min(current_values, axis=0)
        maxs = np.max(current_values, axis=0)
        timesteps = np.arange(current_values.shape[1])

        # Define color for the fills
        base_color = 'skyblue' #'steelblue'

        # Layer 1: Fill between min and max (widest range, lowest opacity)
        ax.fill_between(timesteps, mins, maxs, color=base_color, alpha=0.1, label='Min/Max Range')

        # Layer 2: Fill between 25th and 75th percentile (interquartile range, medium opacity)
        ax.fill_between(timesteps, quantiles_25, quantiles_75, color=base_color, alpha=0.3, label='25-75 Percentile')

        # Layer 3: Fill between mean - std and mean + std (closer to mean, higher opacity)
        # Use an alpha value that visually blends to higher opacity
        ax.fill_between(timesteps, means - stds, means + stds, color=base_color, alpha=0.5, label='Mean +/- Std Dev')

        # Layer 4: Plot the mean line (most opaque)
        ax.plot(timesteps, means, color='darkblue', linewidth=2, label='Mean Value')

        ax.set_title(f"Loss Type: {key}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Loss Value")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6) # Light grid

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.savefig(filename)

def quat_to_aa_wmq_0(quat, side='right'):
    from pytransform3d import rotations
    operator2mano = {
        "right": np.array([
                    [0, 0, -1],
                    [-1, 0, 0],
                    [0, 1, 0]]),
        "left": np.array([
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, -1, 0]])}
    euler = np.tile([rotations.euler_from_matrix(
            rotations.matrix_from_quaternion(q) @ operator2mano[side], 0, 1, 2, extrinsic=False
        ) for q in quat], (quat.shape[0],16,3))
    return torch.from_numpy(euler)

def quat_to_aa_wmq(quat, side='right'):
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles
    operator2mano = {
        "right": np.array([
                    [0, 0, -1],
                    [-1, 0, 0],
                    [0, 1, 0]]),
        "left": np.array([
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, -1, 0]])
    }
    euler = matrix_to_euler_angles(quaternion_to_matrix(quat) @ operator2mano[side], 'XYZ')
    return euler

def retarget_sequence(seq_data, robot_hand: HandRobotWrapper):

    device = robot_hand.device

    side = seq_data['side']

    # Init robot hand transformation
    init_hand_tsl, init_hand_quat = seq_data['h_joints'][:, 0], seq_data['h_coeffs'][:, 0]
    # This is based on the design that
    # (1) the first 6 DoF are dummy joints for world coordinate; 
    # (2) they have the same representation convention;
    dex_pose = torch.zeros((seq_data['h_joints'].shape[0], robot_hand.n_dofs), device=device)  # T x d

    # Turn quat into axis-angle, this should use this certain code because different transformation varies.
    init_hand_aa = quat_to_aa_wmq(init_hand_quat)
    dex_pose[:, 3:6] = init_hand_aa.clone()
    robot_hand.compute_forward_kinematics(dex_pose)
    dex_pose[:, :3] = (init_hand_tsl - robot_hand.get_joint_world_coordinates_dict()["WRJ1"]).clone()
    dex_pose.requires_grad_(True)
    
    # Init robot hand pose (all joints align)
    total_step = 200
    lr = 0.01
    sp_coeffs = .5
    thres = 0.02
    logger_1 = {
        "is_plot_optimize": False,
        "is_plot_timestep": True,
        "is_vis": False,
        "optimize_frame_check_ls": [30, 60, 90],
        "vis_frame": 90,
        "vis_interval": 10,
        'plot_interval': 20,
        "history": {"hand_mesh":[], "robot_keypoints":[]},
        "losses": {"E_align":[], f"E_spen x{sp_coeffs}":[]}
    }
    init_optimizer = torch.optim.Adam([dex_pose], lr=lr, weight_decay=0)
    mano_fingertip = seq_data['h_joints'].to(device)
    for ii in tqdm(range(total_step), desc="Initial alignment"):
        robot_hand.compute_forward_kinematics(dex_pose)
        fingertip_keypoints = robot_hand.get_all_joints_in_mano_order()
        E_align = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip, reduction='none').mean((-2,-1))
        E_spen = robot_hand.self_penetration_part(thres)
        loss = E_align.sum() + E_spen.mean() * sp_coeffs # (B,) --> ()
        init_optimizer.zero_grad()
        loss.backward()
        init_optimizer.step()
        if logger_1['is_vis'] and ii % logger_1['vis_interval'] == 0:
            logger_1['history']['hand_mesh'].append(robot_hand.get_trimesh_data_single(logger_1['vis_frame']))
            logger_1['history']['robot_keypoints'].append(fingertip_keypoints[logger_1['vis_frame']].detach().cpu().numpy())
        logger_1['losses']['E_align'].append(E_align.detach().numpy().tolist())
        logger_1['losses'][f"E_spen x{sp_coeffs}"].append(E_spen.detach().numpy().tolist())

    folder_name = "/home/wangminqi/workspace/test/DexPose/retarget_wmq/Taco_vis"
    config_name = f"stage1_spen{sp_coeffs}_thres{thres}_step{total_step}_lr{lr}"
    filename = f"uid{seq_data['uid']}_{seq_data['which_sequence']}"
    if logger_1['is_vis']:
        os.makedirs(os.path.join(folder_name, config_name, f"vis_frame{logger_1['vis_frame']}"), exist_ok=True)
        from utils.wmq import vis_frames_plotly
        TT = len(logger_1["history"]["hand_mesh"])
        vis_frames_plotly(
            gt_hand_joints=seq_data['h_joints'][logger_1['vis_frame']].expand(TT, -1, -1).cpu().numpy(),
            gt_posi_pts=mano_fingertip[logger_1['vis_frame']].expand(TT, -1, -1).cpu().numpy(),
            posi_pts_ls=[np.stack(logger_1['history']['robot_keypoints'])],
            hand_mesh_ls=[logger_1["history"]["hand_mesh"]],
            show_line=True,
            filename=os.path.join(folder_name, config_name, f"vis_frame{logger_1['vis_frame']}", filename)
        )
    if logger_1['is_plot_optimize']:
        os.makedirs(os.path.join(folder_name, config_name, 'optim'+'_'.join(map(str, logger_1['optimize_frame_check_ls']))), exist_ok=True)
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(1, len(logger_1["optimize_frame_check_ls"]), figsize=(6 * len(logger_1["optimize_frame_check_ls"]), 5))
        if len(logger_1["optimize_frame_check_ls"]) == 1: axs = [axs]
        for idx, ff_n in enumerate(logger_1["optimize_frame_check_ls"]):
            ax = axs[idx]
            for lr in logger_1['losses']:
                loss_item = np.stack([ii[ff_n] for ii in logger_1['losses'][lr]])
                ax.plot(np.arange(len(loss_item)), loss_item, label=lr)
            ax.set_yscale('log')
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.set_title(f"Frame {ff_n}")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, config_name, 'optim'+'_'.join(map(str, logger_1['optimize_frame_check_ls'])), filename+'.png'))
        plt.close()
    if logger_1["is_plot_timestep"]:
        os.makedirs(os.path.join(folder_name, config_name, "sequence_losses"), exist_ok=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        steps   = total_step
        T       = seq_data['seq_len']
        plot_iv = logger_1.get('plot_interval', 1)
        idx_plt = np.arange(0, steps, plot_iv)

        n_loss  = len(logger_1['losses'])
        fig, axs = plt.subplots(n_loss, 1, figsize=(8, 2*n_loss), sharex=True)
        if n_loss == 1: axs = [axs]

        cmap = plt.get_cmap('Greys')
        norm = plt.Normalize(vmin=0, vmax=len(idx_plt)-1)

        for ax_idx, (name, loss_mat) in enumerate(logger_1['losses'].items()):
            ax = axs[ax_idx]
            for j, opt_i in enumerate(idx_plt):
                color = to_rgba(cmap(norm(j)))
                ax.plot(range(T), loss_mat[opt_i], color=color, label=f'step {opt_i}' if j%max(1, len(idx_plt)//5)==0 else "")
            ax.set_ylabel(name)
            ax.set_yscale('log')
            if ax_idx==0:
                ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=6)
        plt.xlabel('time step')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, config_name, "sequence_losses", f"{filename}.png"))
        plt.close()

    # Optimize qpos for better contact + less penetration + less self-penetration
    total_step = 200
    lr = 0.01
    sp_coeffs = 5
    thres = 0.02
    dis_coeffs = 1
    dis_thres = 0.01
    pen_coeffs = 0.005
    pen_thres = 0.005
    logger_2 = {
        "is_plot_optimize": False,
        "is_plot_timestep": True,
        "is_vis": False,
        "optimize_frame_check_ls": [30, 60, 90],
        "vis_frame": 90,
        "vis_interval": 10,
        "plot_interval": 20,
        "history": {"hand_mesh":[], "robot_keypoints":[], "ct_pts":[], "corr_ct_pts":[],"spen":[], "inner_pts":[], "outer_pts":[]},
        "losses": {"E_align":[], f"E_spen x{sp_coeffs}":[], f"E_dist x{dis_coeffs}":[], f"E_pen x{pen_coeffs}":[]}
    }
    check_frame = logger_2["vis_frame"]
    next_optimizer = torch.optim.Adam([dex_pose], lr=lr, weight_decay=0)
    mano_fingertip = seq_data['h_joints'].to(device)
    from utils.tools import get_point_clouds_from_human_data, apply_transformation_human_data, get_object_meshes_from_human_data, apply_transformation_on_object_mesh
    pc, pc_norm = get_point_clouds_from_human_data(seq_data, return_norm=True, ds_num=1000)
    pc_ls, pc_norm_ls = apply_transformation_human_data(pc, seq_data['o_transf'], norm=pc_norm)

    if True:

        # ### Debug SDF, visual mesh is wrong, approximate is right ###
        # import plotly.graph_objects as go
        # robot_hand.compute_forward_kinematics(torch.zeros((30)))
        # mode = "primitive"
        # # mode = "original"

        # def get_frame_data_by_link_name(link_name):
        #     data = []
        #     all_mesh = robot_hand.get_trimesh_data()[0]
        #     verts = np.asarray(all_mesh.vertices)
        #     faces = np.asarray(all_mesh.triangles if hasattr(all_mesh, "triangles") else all_mesh.faces)
        #     data.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        #                         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        #                         color="#E1B685", opacity=0.5, name="Robot Hand Mesh", showlegend=True))
        #     import open3d as o3d
        #     transform_for_t = robot_hand.current_status[link_name]
        #     if mode == "primitive":
        #         v_tensor = transform_for_t.transform_points(robot_hand.mesh[link_name]['c_vertices'])
        #         v_numpy = v_tensor.detach().cpu().numpy()
        #         f_numpy = robot_hand.mesh[link_name]['c_faces'].detach().cpu().numpy()
        #     elif mode == "original":
        #         v_tensor = transform_for_t.transform_points(robot_hand.mesh[link_name]['vertices'])
        #         v_numpy = v_tensor.detach().cpu().numpy()
        #         f_numpy = robot_hand.mesh[link_name]['faces'].detach().cpu().numpy()
        #     else:
        #         raise NotImplementedError("Only support primitive and original mode.")
            
        #     mesh_item = o3d.geometry.TriangleMesh(
        #             vertices=o3d.utility.Vector3dVector(v_numpy),
        #             triangles=o3d.utility.Vector3iVector(f_numpy)
        #         )
        #     verts = np.asarray(mesh_item.vertices)
        #     faces = np.asarray(mesh_item.triangles if hasattr(mesh_item, "triangles") else mesh_item.faces)
        #     data.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        #                         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        #                         color="#914E02", opacity=0.5, name=link_name, showlegend=True))
        #     is_edge_manifold = mesh_item.is_edge_manifold(allow_boundary_edges=True)
        #     is_vertex_manifold = mesh_item.is_vertex_manifold()
        #     is_self_intersecting = mesh_item.is_self_intersecting()
        #     watertight = mesh_item.is_watertight()
        #     print(f"{link_name:<20} │ {is_edge_manifold!s:<13} │ {is_vertex_manifold!s:<15} │ "
        #             f"{is_self_intersecting!s:<14} │ {watertight!s:<9} │ "
        #             f"{len(verts):>8} / {len(faces):>8}")

        #     if link_name == "forearm":
        #         rrange = 0.2
        #         interval = 0.02
        #     elif link_name == "palm":
        #         rrange = 0.12
        #         interval = 0.012
        #     else:
        #         rrange = 0.08
        #         interval = 0.008
        #     grid_range = np.arange(-rrange, rrange+interval, interval)
        #     grid_x, grid_y, grid_z = np.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
        #     grid_points_local_np = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        #     grid_points_local_torch = torch.from_numpy(grid_points_local_np).to(dtype=torch.float, device=robot_hand.device)
        #     grid_points_world_torch = transform_for_t.transform_points(grid_points_local_torch)
        #     grid_points_world_np = grid_points_world_torch.detach().cpu().numpy()

        #     # 3. Compute the SDF value for each point.
        #     if mode == "primitive":
        #         face_verts_local = robot_hand.mesh[link_name]['c_face_verts']
        #     elif mode == "original":
        #         face_verts_local = robot_hand.mesh[link_name]['face_verts']
        #     else:
        #         raise NotImplementedError("Only support primitive and original mode.")
        #     face_verts_world = transform_for_t.transform_points(face_verts_local)
        #     # from torchsdf import compute_sdf
        #     # dist_sq, signs, _, _ = compute_sdf(grid_points_world_torch.to('cuda'), face_verts_world.to('cuda'))
        #     # sdf_values = torch.sqrt(dist_sq.clamp(min=1e-8)) * (signs)
        #     # from .robot_wrapper import sdf_capsule_analytical_torch
        #     # sdf_values = sdf_capsule_analytical_torch(grid_points_local_torch, robot_hand.mesh[link_name]['capsule_params'])
        #     # sdf_values = sdf_values.detach().cpu().numpy()
        #     from .robot_wrapper import sdf_capsule_analytical_batch_torch
        #     sdf_values = sdf_capsule_analytical_batch_torch(grid_points_local_torch.expand(10,-1,-1), robot_hand.mesh[link_name]['capsule_params'])[0]
        #     sdf_values = sdf_values.detach().cpu().numpy()

        #     # 4. Normalize the SDF values to a [0, 1] opacity scale
        #     opacities = np.zeros_like(sdf_values)
        #     positive_mask = sdf_values > 0
        #     negative_mask = sdf_values < 0
        #     # Normalize positive values: The largest positive SDF value will have opacity 1.
        #     if np.any(positive_mask):
        #         positive_sdfs = sdf_values[positive_mask]
        #         max_pos_sdf = np.max(positive_sdfs)
        #         if max_pos_sdf > 1e-6: # Avoid division by zero
        #             opacities[positive_mask] = (positive_sdfs / max_pos_sdf) ** 3
        #     # Normalize negative values: The most negative SDF value will have opacity 1.
        #     if np.any(negative_mask):
        #         negative_sdfs = sdf_values[negative_mask]
        #         # The "largest" negative value is the minimum value. Its absolute is the max distance inside.
        #         max_abs_neg_sdf = np.abs(np.min(negative_sdfs))
        #         if max_abs_neg_sdf > 1e-6: # Avoid division by zero
        #             opacities[negative_mask] = (np.abs(negative_sdfs) / max_abs_neg_sdf) ** 3

        #     # 5. Plot the points with their calculated colors and opacities
        #     # Plot positive (outside) points as blue
        #         rgba_colors_positive = [
        #             f"rgba(0, 0, 255, {op:.8f})"  for op in opacities[positive_mask]
        #         ]
        #         data.append(go.Scatter3d(
        #             x=grid_points_world_np[positive_mask, 0],
        #             y=grid_points_world_np[positive_mask, 1],
        #             z=grid_points_world_np[positive_mask, 2],
        #             mode='markers',
        #             marker=dict(
        #                 # Pass the list of RGBA strings to the color property
        #                 color=rgba_colors_positive,
        #                 size=3
        #             ),
        #             name='SDF (Outside)'
        #         ))

        #     # Plot negative (inside) points as red with variable opacity
        #     if np.any(negative_mask):
        #         rgba_colors_negative = [
        #             f"rgba(255, 0, 0, {op:.8f})"  for op in opacities[negative_mask]
        #         ]
        #         data.append(go.Scatter3d(
        #             x=grid_points_world_np[negative_mask, 0],
        #             y=grid_points_world_np[negative_mask, 1],
        #             z=grid_points_world_np[negative_mask, 2],
        #             mode='markers',
        #             marker=dict(
        #                 # Pass the list of RGBA strings to the color property
        #                 color=rgba_colors_negative,
        #                 size=3
        #             ),
        #             name='SDF (Inside)'
        #         ))
            
        #     # --- NEW: Calculate and Visualize Face Normals ---
        #     # Your face_verts_world is shape (B, F, 3, 3). Since qpos was 1D, B=1.
        #     # We can squeeze it to (F, 3, 3) for easier handling.
        #     face_verts_world = transform_for_t.transform_points(face_verts_local).unsqueeze(0) # Shape: (1, F, 3, 3)
        #     faces = face_verts_world.squeeze(0) # Shape: (F, 3, 3)

        #     # Calculate face normals using the cross product of two edges
        #     v1 = faces[:, 1, :] - faces[:, 0, :] # Edge P0 -> P1
        #     v2 = faces[:, 2, :] - faces[:, 0, :] # Edge P0 -> P2
        #     face_normals = torch.cross(v1, v2, dim=1)
            
        #     # Normalize to get unit vectors
        #     norms = torch.linalg.norm(face_normals, dim=1, keepdim=True)
        #     unit_normals = face_normals / (norms + 1e-8)

        #     # Calculate the center of each face (centroid) to be the starting point of the normal vector
        #     face_centers = torch.mean(faces, dim=1)

        #     # Define the length of the visualized normal vectors
        #     normal_length_scale = 0.005
        #     end_points = face_centers + unit_normals * normal_length_scale

        #     # Convert to NumPy for plotting
        #     centers_np = face_centers.detach().cpu().numpy()
        #     ends_np = end_points.detach().cpu().numpy()

        #     # Prepare coordinate lists for a single efficient Scatter3d trace
        #     lines_x, lines_y, lines_z = [], [], []
        #     for i in range(len(centers_np)):
        #         lines_x.extend([centers_np[i, 0], ends_np[i, 0], None])
        #         lines_y.extend([centers_np[i, 1], ends_np[i, 1], None])
        #         lines_z.extend([centers_np[i, 2], ends_np[i, 2], None])

        #     data.append(go.Scatter3d(
        #         x=lines_x,
        #         y=lines_y,
        #         z=lines_z,
        #         mode='lines',
        #         line=dict(color='cyan', width=2),
        #         name='Face Normals',
        #     ))
            
        #     return go.Frame(data=data, name=link_name)
        
        # if mode == "primitive":
        #     names = [link_name for link_name in robot_hand.mesh if link_name.endswith("distal") or link_name.endswith("proximal") or link_name.endswith("middle")]
        # elif mode == "original":
        #     names = list(robot_hand.mesh.keys())
        # else:
        #     raise NotImplementedError("Only support primitive and original mode.")
        # frames = []
        # for link_name in names:
        #     frames.append(get_frame_data_by_link_name(link_name))
        # init_frame = frames[0]

        # # --- Setup Layout, Slider, and Buttons ---
        # slider_steps = [{"method": "animate", "label": link_name, "args": [[link_name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]} for link_name in names]
        # layout = go.Layout(
        #     title="DexHand Optimization Process",
        #     scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        #     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        #     updatemenus=[{"type": "buttons", "buttons": [
        #         {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]},
        #         {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
        #     ]}],
        #     sliders=[{"active": 0, "yanchor": "top", "xanchor": "left", "currentvalue": {"font": {"size": 20}, "prefix": "Frame:", "visible": True, "xanchor": "right"}, "transition": {"duration": 0}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0, "steps": slider_steps}]
        # )

        # # --- Create and Save Figure ---
        # fig = go.Figure(data=init_frame.data, layout=layout, frames=frames)
        # fig.write_html(f"retarget_wmq/0929vis/(4)0926/test_sdf_{mode}.html")
        # import sys;sys.exit()
        # ### Debug SDF, visual mesh is wrong, approximate is right ###


        pass

    for ii in tqdm(range(total_step), desc="Stage 2 optimization"):
        robot_hand.compute_forward_kinematics(dex_pose)
        fingertip_keypoints = robot_hand.get_align_points()
        fingertip_keypoints = robot_hand.get_all_joints_in_mano_order()
        E_align = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip, reduction='none').mean((-2,-1))
        E_dis, ct_pts, cc_ct_pts = robot_hand.cal_distance(pc_ls, pc_norm_ls, dis_thres, True)
        E_pen, inner_pts, outer_pts = robot_hand.cal_object_penetration(pc_ls, pen_thres, True)
        E_spen = robot_hand.self_penetration_part(thres)
        loss = E_spen.mean() * sp_coeffs + E_pen.mean() * pen_coeffs + E_dis.mean() * dis_coeffs + E_align.sum()
        next_optimizer.zero_grad()
        loss.backward()
        next_optimizer.step()
        if logger_2["is_vis"] and ii % logger_2['vis_interval'] == 0:
            logger_2["history"]["hand_mesh"].append(robot_hand.get_trimesh_data_single(check_frame))
            logger_2["history"]["robot_keypoints"].append(fingertip_keypoints[check_frame].detach().cpu().numpy())
            logger_2["history"]["ct_pts"].append(ct_pts[check_frame].detach().cpu().numpy())
            logger_2["history"]["corr_ct_pts"].append(cc_ct_pts[check_frame].detach().cpu().numpy())
            logger_2["history"]["outer_pts"].append(outer_pts[check_frame].detach().cpu().numpy())
        logger_2["history"]["inner_pts"].append([ii.detach().cpu().numpy() for ii in inner_pts])
        logger_2["losses"]["E_align"].append(E_align.detach().numpy().tolist())
        logger_2["losses"][f"E_spen x{sp_coeffs}"].append(E_spen.detach().numpy().tolist())
        logger_2["losses"][f"E_dist x{dis_coeffs}"].append(E_dis.detach().numpy().tolist())
        logger_2["losses"][f"E_pen x{pen_coeffs}"].append(E_pen.detach().numpy().tolist())

    folder_name = "/home/wangminqi/workspace/test/DexPose/retarget_wmq/Taco_vis"
    config_name = f"stage2_step{total_step}_all_lr{lr}_thres{thres}_spcoeffs{sp_coeffs}_pen{pen_coeffs}_{pen_thres}_dis{dis_coeffs}square_disthres{dis_thres}"
    filename = f"uid{seq_data['uid']}_{seq_data['which_sequence']}"
    if logger_2["is_vis"]:
        os.makedirs(os.path.join(folder_name, config_name, f"vis_frame{check_frame}"), exist_ok=True)
        from utils.wmq import vis_dexhand_optimize
        obj_mesh = get_object_meshes_from_human_data(seq_data)
        obj_mesh_ls = apply_transformation_on_object_mesh(obj_mesh, seq_data['o_transf'][:, check_frame:check_frame+1, :, :])
        TT = len(logger_2["history"]["hand_mesh"])
        vis_dexhand_optimize(
            pc_ls=[np.tile(pc_ls[check_frame], (TT, 1, 1))],
            object_mesh_ls=[i*(TT) for i in obj_mesh_ls],
            gt_hand_joints=seq_data['h_joints'][check_frame].expand(TT, -1, -1).cpu().numpy(),
            hand_mesh_ls=[logger_2["history"]["hand_mesh"]],
            gt_posi_pts=mano_fingertip[check_frame].expand(TT, -1, -1).cpu().numpy(),
            posi_pts_ls=np.stack(logger_2["history"]["robot_keypoints"]),
            contact_pt_ls=np.stack(logger_2["history"]["ct_pts"]),
            corr_contact_pt_ls=np.stack(logger_2["history"]["corr_ct_pts"]),
            inner_pen_pts=logger_2["history"]["inner_pts"][check_frame],
            outer_pen_pts=logger_2["history"]["outer_pts"],
            filename=os.path.join(folder_name, config_name, f"vis_frame{check_frame}", filename)
        )
    if logger_2['is_plot_optimize']:
        os.makedirs(os.path.join(folder_name, config_name, 'optim'+'_'.join(map(str, logger_2['optimize_frame_check_ls']))), exist_ok=True)
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(6 * len(logger_2["optimize_frame_check_ls"]), 5 * 2))
        gs = fig.add_gridspec(2, len(logger_2["optimize_frame_check_ls"]), hspace=0.3)

        # 上排：loss 曲线
        for col, ff_n in enumerate(logger_2["optimize_frame_check_ls"]):
            ax = fig.add_subplot(gs[0, col])
            for lr in logger_2['losses']:
                loss_item = np.stack([ii[ff_n] for ii in logger_2['losses'][lr]])
                ax.plot(np.arange(len(loss_item)), loss_item, label=lr)
            ax.set_yscale('log')
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.set_title(f"Frame {ff_n}")

        # 下排：inner points 数量
        for col, ff_n in enumerate(logger_2["optimize_frame_check_ls"]):
            ax = fig.add_subplot(gs[1, col])
            inner_cnt = [len(step_ls[ff_n]) for step_ls in logger_2["history"]["inner_pts"]]
            ax.plot(np.arange(len(inner_cnt)), inner_cnt, label="inner points num")
            ax.set_xlabel("Step")
            ax.set_ylabel("Num")
            ax.legend()
            ax.set_title(f"Frame {ff_n}")

        plt.savefig(os.path.join(folder_name, config_name, 'optim'+'_'.join(map(str, logger_2['optimize_frame_check_ls'])), filename+'.png'))
        plt.close()
    if logger_2["is_plot_timestep"]:
        os.makedirs(os.path.join(folder_name, config_name, "sequence_losses"), exist_ok=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        steps   = total_step
        T       = seq_data['seq_len']
        plot_iv = logger_2.get('plot_interval', 1)
        idx_plt = np.arange(0, steps, plot_iv)

        n_loss  = len(logger_2['losses'])
        fig, axs = plt.subplots(n_loss+1, 1, figsize=(8, 2+2*n_loss), sharex=True)

        cmap = plt.get_cmap('Greys')
        norm = plt.Normalize(vmin=0, vmax=len(idx_plt)-1)

        for ax_idx, (name, loss_mat) in enumerate(logger_2['losses'].items()):
            # 左列：loss 随 time-step 变化
            ax = axs[ax_idx]
            for j, opt_i in enumerate(idx_plt):
                color = to_rgba(cmap(norm(j)))
                ax.plot(range(T), loss_mat[opt_i], color=color,
                        label=f'step {opt_i}' if j % max(1, len(idx_plt)//5) == 0 else "")
            ax.set_ylabel(name)
            ax.set_yscale('log')
            ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=6)

        ax = axs[-1]
        for j, opt_i in enumerate(idx_plt):
            color = to_rgba(cmap(norm(j)))
            cnt_t = [len(time_ls) for time_ls in logger_2["history"]["inner_pts"][opt_i]]
            ax.plot(range(T), cnt_t, color=color,
                    label=f'step {opt_i}' if j % max(1, len(idx_plt)//5) == 0 else "")
        ax.set_ylabel('Inner points num')
        ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=6)

        plt.xlabel('time step')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, config_name, "sequence_losses", f"{filename}.png"))
        plt.close()

    retargeted_seq = dict(            
        which_hand=robot_hand.robot_name,
        dex_poses=dex_pose,
        )
    retargeted_seq.update(seq_data)

    return retargeted_seq, logger_1['losses'], logger_2['losses']

def main_retarget(seq_data_ls, robots):
    save_path = '/home/wangminqi/workspace/test/data/Taco/dex_save1002/shadow_1.p'
    if os.path.isfile(save_path): os.remove(save_path)
    processed_data = {}
    for robot_name in robots:
        robot_hand = load_robot(robot_name)
        print(f"Retargeting to {robot_hand.robot_name} ...")
        retargeted_data = []
        losses_1_all = []
        losses_2_all = []
        for i, seq_data in enumerate(seq_data_ls):
            print(f"Sequence {i}: Processing sequence {seq_data['uid']} with {robot_name} from {seq_data['which_dataset']} with {seq_data['which_sequence']}")
            if seq_data['if_flip']:
                print(f"SKIP! Sequence {seq_data['uid']}")
                continue
            retargeted_seq, losses_1, losses_2 = retarget_sequence(seq_data, robot_hand)
            retargeted_data.append(retargeted_seq)
            losses_1_all.append(losses_1)
            losses_2_all.append(losses_2)
            with open(save_path, 'wb') as ofs:
                joblib.dump(retargeted_data, ofs)
                from utils.vis_utils import visualize_dex_hand_sequence
            
            # os.makedirs('/home/wangminqi/workspace/test/DexPose/retarget_wmq/Taco_vis/seq_vis/',exist_ok=True)
            # visualize_dex_hand_sequence(retargeted_seq, f"/home/wangminqi/workspace/test/DexPose/retarget_wmq/Taco_vis/seq_vis/{retargeted_seq['uid']}")

        processed_data[robot_hand] = retargeted_data

        if False:
            # draw losses
            avg_losses_1 = {}
            avg_losses_2 = {}
            for key in losses_1_all[0]:
                avg_losses_1[key] = [losses_1_all[i][key] for i in range(len(losses_1_all))] # list (N) of list (T)
            for key in losses_2_all[0]:
                avg_losses_2[key] = [losses_2_all[i][key] for i in range(len(losses_2_all))] # list (N) of list (T)

            visualize_time_series_with_fill_between(
                avg_losses_1, 
                title=f"Stage 1 Losses for {robot_name}", 
                filename=f"retarget_wmq/0929vis/all_seq_stage1_losses_{robot_name}"
            )
            visualize_time_series_with_fill_between(
                avg_losses_2, 
                title=f"Stage 2 Losses for {robot_name}", 
                filename=f"retarget_wmq/0929vis/all_seq_stage2_losses_{robot_name}"
            )

    return processed_data


if __name__ == "__main__":
    
    import os
    import pickle
    from tqdm import tqdm

    robots = ['shadow_hand']
    file_path = '/home/wangminqi/workspace/test/data/Taco/human_save0929/seq_mirror_correct_1.p'
    import pickle
    with open(file_path, 'rb') as f:
        seq_data_ls = pickle.load(f)[0::2]
    processed_data = main_retarget(seq_data_ls, robots)

