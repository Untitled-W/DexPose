import torch
import numpy as np

from .robot_wrapper import load_robot, HandRobotWrapper
# from utils.hand_model import load_robot
from utils.wmq import visualize_dex_hand_sequence_together, vis_frames_plotly, vis_pc_coor_plotly

def quat_to_aa_wmq_0(quat, side='right'):
    """
    quat is not batched
    """
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
    euler = rotations.euler_from_matrix(
            rotations.matrix_from_quaternion(quat) @ operator2mano[side], 0, 1, 2, extrinsic=False
        )
    return euler

def quat_to_aa_wmq(quat, side='right'):
    """
    quat is not batched
    """
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles
    operator2mano = {
        "right": np.array([
                    [0, 0, -1],
                    [-1, 0, 0],
                    [0, 1, 0]]),
        "left": np.array([
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, -1, 0]])}
    euler = matrix_to_euler_angles(quaternion_to_matrix(quat) @ operator2mano[side], 'XYZ')
    return euler

def retarget_sequence(seq_data, robot_hand: HandRobotWrapper):

    device = robot_hand.device

    # Init robot hand transformation
    init_hand_tsl, init_hand_quat = seq_data['hand_joints'][:, 0], seq_data['hand_coeffs'][:, 0]
    # This is based on the design that
    # (1) the first 6 DoF are dummy joints for world coordinate; 
    # (2) they have the same representation convention;
    dex_pose = torch.zeros((seq_data['hand_joints'].shape[0], robot_hand.n_dofs), device=device)  # T x d

    # Turn quat into axis-angle, this should use this certain code because different transformation varies.
    init_hand_aa = quat_to_aa_wmq(init_hand_quat)
    dex_pose[:, 3:6] = init_hand_aa.clone()
    robot_hand.compute_forward_kinematics(dex_pose)
    dex_pose[:, :3] = (init_hand_tsl - robot_hand.get_joint_world_coordinates_dict()["WRJ1"]).clone()
    dex_pose.requires_grad_(True)
    
    if True:
        # ### Test hand orientation ###
        # seq_data['which_hand']='shadow_hand'
        # seq_data['hand_poses'] = dex_pose
        # visualize_dex_hand_sequence_together([seq_data], [robot_hand.robot_name], filename="retarget/0912/xx")
        # import sys;sys.exit()
        # ### Test hand orientation ###

        
        # ### Test finger keypoints ###
        # robot_hand.compute_forward_kinematics(dex_pose)
        # mano_fingertip = seq_data["hand_joints"][:, [5, 8, 9, 12, 13, 16, 17, 20, 4]].to(device)
        # fingertip_keypoints = robot_hand.get_tip_points()
        # vis_frames_plotly(
        #             # pc_ls=[np.tile(kwargs['point_cloud'], (T, 1, 1))], 
        #             gt_hand_joints=seq_data['hand_joints'].cpu().numpy(),
        #             gt_posi_pts=mano_fingertip.cpu().numpy(),
        #             posi_pts_ls=[fingertip_keypoints.detach().cpu().numpy()],
        #             hand_mesh_ls=[robot_hand.get_trimesh_data()], 
        #             show_line=True,
        #             filename=f"retarget/0912/test_finger_keypoints")
        # import sys;sys.exit()
        # ### Test finger keypoints ###


        # ### Test hand keypoints order and joints order ###
        # qq = torch.zeros(robot_hand.dof)
        # qq[3:6] = torch.from_numpy(quat_to_aa_wmq_0(torch.zeros(4)))
        # robot_hand.compute_forward_kinematics(qq)
        # robot_keypoints = robot_hand.get_joint_world_coordinates_dict()
        # from utils.tools import extract_hand_points_and_mesh
        # human_keypoints, _ = extract_hand_points_and_mesh((seq_data['hand_tsls'][0]), torch.zeros_like(seq_data['hand_coeffs'][0]), "right")
        # from utils.wmq import visualize_hand_and_joints
        # visualize_hand_and_joints(
        #     mano_joint=human_keypoints[0],
        #     robot_keypoints=robot_keypoints,
        #     robot_hand_mesh=robot_hand.get_trimesh_data()[0],
        #     human_keypoints=human_keypoints[0],
        #     filename="retarget/0912/test_hand_joints_order"
        # )
        # import sys;sys.exit()
        # ### Test hand keypoints order and joints order ###

        # ### Test hand links ###
        # data_traces = []
        # qq = torch.zeros(robot_hand.n_dofs)
        # qq[3:6] = torch.from_numpy(quat_to_aa_wmq_0(torch.zeros(4)))
        # robot_hand.compute_forward_kinematics(qq)
        # from utils.tools import extract_hand_points_and_mesh
        # human_keypoints, _ = extract_hand_points_and_mesh((seq_data['hand_tsls'][0]), torch.zeros_like(seq_data['hand_coeffs'][0]), "right")
        # import plotly.graph_objects as go
        # for link_name in robot_hand.mesh:
        #     import open3d as o3d
        #     transform_for_t = robot_hand.current_status[link_name]
        #     v_tensor = transform_for_t.transform_points(robot_hand.mesh[link_name]['vertices'])
        #     v_numpy = v_tensor.detach().cpu().numpy()
        #     f_numpy = robot_hand.mesh[link_name]['faces'].detach().cpu().numpy()
        #     mesh_item = o3d.geometry.TriangleMesh(
        #             vertices=o3d.utility.Vector3dVector(v_numpy),
        #             triangles=o3d.utility.Vector3iVector(f_numpy)
        #         )
        #     verts = np.asarray(mesh_item.vertices)
        #     faces = np.asarray(mesh_item.triangles if hasattr(mesh_item, "triangles") else mesh_item.faces)
        #     data_traces.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        #                         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        #                         color="#914E02", opacity=0.5, name=link_name, showlegend=True))
        # human_keypoints_np = human_keypoints[0]
        # from utils.wmq import get_vis_hand_keypoints_with_color_gradient_and_lines
        # skeleton_traces = get_vis_hand_keypoints_with_color_gradient_and_lines(
        #     human_keypoints_np, 
        #     color_scale='Reds'
        # )
        # data_traces.extend(skeleton_traces)
        # fig = go.Figure(data=data_traces)
        # fig.update_layout(
        #     title="MANO vs. Robot Joint and Mesh Comparison",
        #     scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        #     legend_title_text='Joints & Meshes',
        #     margin=dict(l=0, r=0, b=0, t=40)
        # )
        # filename = f"retarget/0912/(2)good_wrist/test_hand_links"
        # fig.write_html(f"{filename}.html")
        # print(f"Visualization saved to {filename}.html")
        # import sys;sys.exit()
        # ### Test hand links ###

        # ### Test penetration order ###
        # qq = torch.zeros(robot_hand.dof)
        # qq[3:6] = torch.from_numpy(quat_to_aa_wmq_0(torch.zeros(4)))
        # robot_hand.compute_forward_kinematics(qq)
        # robot_keypoints = {ii:v for ii, v in enumerate(robot_hand.get_penetration_keypoints())}
        # print(len(list(robot_keypoints.keys())))
        # from utils.tools import extract_hand_points_and_mesh
        # human_keypoints, _ = extract_hand_points_and_mesh((seq_data['hand_tsls'][0]), torch.zeros_like(seq_data['hand_coeffs'][0]), "right")
        # from utils.wmq import visualize_hand_and_joints
        # visualize_hand_and_joints(
        #     robot_keypoints=robot_keypoints,
        #     robot_hand_mesh=robot_hand.get_trimesh_data()[0],
        #     filename="retarget/0912/(2)pose_initialization/test_penetration_order"
        # )
        # import sys;sys.exit()
        # ### Test penetration order ###

        # # Init robot hand pose (only fingertips align)
        # history = {"mesh":[], "keypoints":[]}
        # check_frame = 60
        # total_step = 150
        # lr = 0.05
        # init_optimizer = torch.optim.Adam([dex_pose], lr=lr, weight_decay=0)
        # mano_fingertip = seq_data["hand_joints"][:, [5, 8, 9, 12, 13, 16, 17, 20, 4]].to(device)
        # for ii in tqdm(range(total_step), desc="Initial alignment"):
        #     robot_hand.compute_forward_kinematics(dex_pose[check_frame])
        #     fingertip_keypoints = robot_hand.get_tip_points()
        #     history["keypoints"].append(fingertip_keypoints.detach().cpu().numpy())
        #     history["mesh"].append(robot_hand.get_trimesh_data())
        #     loss = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip[check_frame], reduction='sum')
        #     init_optimizer.zero_grad()
        #     loss.backward()
        #     init_optimizer.step()


        # ### Test batched get_all_joints_in_mano_order ###
        # robot_hand.compute_forward_kinematics(dex_pose)
        # fingertip_keypoints = robot_hand.get_tip_points()
        # joints_dict = robot_hand.get_joint_world_coordinates_dict()
        # joints = robot_hand.get_all_joints_in_mano_order()
        # print(joints.shape) # T x 21 x 3
        # robot_hand.compute_forward_kinematics(dex_pose[0])
        # fingertip_keypoints = robot_hand.get_tip_points()
        # joints_dict = robot_hand.get_joint_world_coordinates_dict()
        # joints = robot_hand.get_all_joints_in_mano_order()
        # print(joints.shape) # 21 x 3
        # import sys;sys.exit()
        # ### Test batched get_all_joints_in_mano_order ###

        # ### Init robot hand pose (all joints align)
        # # history = {"mesh":[], "keypoints":[]}
        # # losses = {}
        # # check_frame = 90
        # total_step = 200
        # lr = 0.01
        # # for lr in [0.1, 0.05, 0.01]:
        #     # losses[lr] = []
        # init_optimizer = torch.optim.Adam([dex_pose], lr=lr, weight_decay=0)
        # # TODO Fix later. This should return corresponding hand keypoints for different hands.
        # mano_fingertip = seq_data["hand_joints"].to(device)#[check_frame]
        # for ii in tqdm(range(total_step), desc="Initial alignment"):
        #     robot_hand.compute_forward_kinematics(dex_pose)#[check_frame])
        #     fingertip_keypoints = robot_hand.get_all_joints_in_mano_order()
        #     loss = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip, reduction='sum')
        #     # history["keypoints"].append(fingertip_keypoints.detach().cpu().numpy())
        #     # history["mesh"].append(robot_hand.get_trimesh_data())
        #     # losses[lr].append(loss.item())
        #     init_optimizer.zero_grad()
        #     loss.backward()
        #     init_optimizer.step()
        # ###
        pass

    # Init robot hand pose (all joints align)
    total_step = 200
    lr = 0.01
    init_optimizer = torch.optim.Adam([dex_pose], lr=lr, weight_decay=0)
    mano_fingertip = seq_data["hand_joints"].to(device)
    for ii in tqdm(range(total_step), desc="Initial alignment"):
        robot_hand.compute_forward_kinematics(dex_pose)
        fingertip_keypoints = robot_hand.get_all_joints_in_mano_order()
        loss = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip, reduction='sum')
        init_optimizer.zero_grad()
        loss.backward()
        init_optimizer.step()

    if True:
        ### Test finger align ###
        # vis_frames_plotly(
        #             # pc_ls=[np.tile(kwargs['point_cloud'], (T, 1, 1))],
        #             gt_hand_joints=seq_data['hand_joints'][check_frame].expand(total_step, -1, -1).cpu().numpy(),
        #             gt_posi_pts=mano_fingertip.expand(total_step, -1, -1).cpu().numpy(),
        #             posi_pts_ls=[np.stack(history["keypoints"])],
        #             hand_mesh_ls=[[i[0] for i in history["mesh"]]],
        #             show_line=True,
        #             filename=f"retarget/0912/(2)good_wrist/all_joints_step{total_step}_lr{lr}_frame{check_frame}")
        # from matplotlib import pyplot as plt
        # plt.figure()
        # for lr in losses:
        #     plt.plot(np.arange(len(losses[lr])), losses[lr], label=f"lr={lr}")
        # plt.yscale('log')
        # plt.xlabel("Step")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig(f"retarget/0912/(2)good_wrist/all_joints_frame{check_frame}.png")
        # plt.close()
        # import sys;sys.exit()
        ### Test finger align ###
        pass

    # Optimize qpos for better contact + less penetration + less self-penetration
    total_step = 200
    lr = 0.01
    sp_coeffs = 1
    thres = 0.02
    dis_coeffs = 2
    pen_coeffs = 1
    next_optimizer = torch.optim.Adam([dex_pose], lr=lr, weight_decay=0)
    mano_fingertip = seq_data["hand_joints"].to(device)[:, [5, 8, 9, 12, 13, 16, 17, 20, 4]]
    from utils.tools import get_point_clouds_from_human_data, apply_transformation_human_data
    pc, pc_norm = get_point_clouds_from_human_data(seq_data, return_norm=True, ds_num=1000)
    pc_ls, pc_norm_ls = apply_transformation_human_data(pc, seq_data["obj_poses"], norm=pc_norm)

    if True:
        # ### Debug spen ###
        # for ii in tqdm(range(total_step), desc="Stage 2 optimization"):
        #     robot_hand.compute_forward_kinematics(dex_pose[check_frame])
        #     fingertip_keypoints = robot_hand.get_align_points()
        #     E_align = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip, reduction='sum')
        #     E_spen = robot_hand.self_penetration_part(thres)
        #     history["keypoints"].append(fingertip_keypoints.detach().cpu().numpy())
        #     history["mesh"].append(robot_hand.get_trimesh_data())
        #     history["spen"].append(robot_hand.get_penetration_keypoints().detach().cpu().numpy())
        #     loss = E_spen * sp_coeffs +  E_align
        #     losses["E_align"].append(E_align.item())
        #     losses["E_spen"].append(E_spen.item())
        #     losses["total"].append(loss.item())
        #     next_optimizer.zero_grad()
        #     loss.backward()
        #     next_optimizer.step()

        # from utils.wmq import vis_dexhand_optimize
        # vis_dexhand_optimize(
        #     pc_ls=[np.tile(pc_ls, (total_step, 1, 1))],
        #     # object_mesh_ls=[i*total_step for i in obj_mesh_ls],
        #     gt_hand_joints=seq_data['hand_joints'][check_frame].expand(total_step, -1, -1).cpu().numpy(),
        #     hand_mesh_ls=[[i[0] for i in history["mesh"]]],
        #     gt_posi_pts=mano_fingertip.expand(total_step, -1, -1).cpu().numpy(),
        #     posi_pts_ls=np.stack(history["keypoints"]),
        #     penetration_keypoints=np.stack(history["spen"]),
        #     # contact_pt_ls=np.concatenate(history["ct_pts"]),
        #     # corr_contact_pt_ls=np.concatenate(history["corr_ct_pts"]),
        #     filename=f"retarget/0912/(2)good_wrist/spen_v2_step{total_step}_lr{lr}_thres{thres}_spcoeffs{sp_coeffs}_frame{check_frame}"
        # )
        # from matplotlib import pyplot as plt
        # plt.figure()
        # for key in losses:
        #     plt.plot(np.arange(len(losses[key])), losses[key], label=key)
        # plt.yscale('log')
        # plt.xlabel("Step")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig(f"retarget/0912/(2)good_wrist/spen_v2_step{total_step}_lr{lr}__thres{thres}_spcoeffs{sp_coeffs}_frame{check_frame}.png")
        # plt.close()
        # import sys;sys.exit()
        # ### Debug spen ###

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
        #     from torchsdf import compute_sdf
        #     dist_sq, signs, _, _ = compute_sdf(grid_points_world_torch.to('cuda'), face_verts_world.to('cuda'))
        #     sdf_values = torch.sqrt(dist_sq.clamp(min=1e-8)) * (signs)
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
        # fig.write_html(f"retarget/0912/(3)contact_penetration/test_sdf_{mode}.html")
        # import sys;sys.exit()
        # ### Debug SDF, visual mesh is wrong, approximate is right ###

        # ### Debug SDF using pytorch-volumetric ###
        # rrange = 0.2
        # interval = 0.05
        # grid_range = np.arange(-rrange, rrange+interval, interval)
        # grid_x, grid_y, grid_z = np.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
        # grid_points_np = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        # grid_points_torch = torch.from_numpy(grid_points_np).to(dtype=torch.float, device=robot_hand.device) + torch.tensor([-0.2, -0.2, 0.0], device=robot_hand.device) # Move to the hand center
        # robot_hand.sdf.set_joint_configuration(robot_hand.qpos[check_frame])
        # sdf_val, sdf_grad = robot_hand.sdf(grid_points_torch)
        # print(grid_points_torch.shape, sdf_val.shape, sdf_grad.shape)
        
        # import plotly.graph_objects as go
        # def draw(grid_points_world_np, sdf_values):
        #     data = []
        #     all_mesh = robot_hand.get_trimesh_data()[0]
        #     verts = np.asarray(all_mesh.vertices)
        #     faces = np.asarray(all_mesh.triangles if hasattr(all_mesh, "triangles") else all_mesh.faces)
        #     data.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        #                         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        #                         color="#E1B685", opacity=0.5, name="Robot Hand Mesh", showlegend=True))

        #     # 4. Normalize the SDF values to a [0, 1] opacity scale
        #     opacities = np.zeros_like(sdf_values)
        #     positive_mask = sdf_values > 0
        #     negative_mask = sdf_values < 0
        #     # Normalize positive values: The largest positive SDF value will have opacity 1.
        #     if np.any(positive_mask):
        #         positive_sdfs = sdf_values[positive_mask]
        #         max_pos_sdf = np.max(positive_sdfs)
        #         if max_pos_sdf > 1e-6: # Avoid division by zero
        #             opacities[positive_mask] = (positive_sdfs / max_pos_sdf) ** 2
        #     # Normalize negative values: The most negative SDF value will have opacity 1.
        #     if np.any(negative_mask):
        #         negative_sdfs = sdf_values[negative_mask]
        #         # The "largest" negative value is the minimum value. Its absolute is the max distance inside.
        #         max_abs_neg_sdf = np.abs(np.min(negative_sdfs))
        #         if max_abs_neg_sdf > 1e-6: # Avoid division by zero
        #             opacities[negative_mask] = (np.abs(negative_sdfs) / max_abs_neg_sdf) ** 2

        #     # 5. Plot the points with their calculated colors and opacities
        #     # Plot positive (outside) points as red
        #         rgba_colors_positive = [
        #             f"rgba(255, 0, 0, {op:.8f})"  for op in opacities[positive_mask]
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

        #     # Plot negative (inside) points as blue with variable opacity
        #     if np.any(negative_mask):
        #         rgba_colors_negative = [
        #             f"rgba(0, 0, 255, {op:.8f})"  for op in opacities[negative_mask]
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
            
        #     return data
        
        # fig = go.Figure(data=draw(grid_points_world_np=grid_points_np, sdf_values=sdf_val.detach().cpu().numpy()))
        # fig.write_html(f"retarget/0912/(3)contact_penetration/test_sdf_{rrange}_{interval}.html")
        # import sys;sys.exit()
        # ### Debug SDF using pytorch-volumetric ###
    
        # ### Debug penetration ###
        # for ii in tqdm(range(total_step), desc="Stage 2 optimization"):
        #     robot_hand.compute_forward_kinematics(dex_pose[check_frame])
        #     fingertip_keypoints = robot_hand.get_align_points()
        #     history["keypoints"].append(fingertip_keypoints.detach().cpu().numpy())
        #     history["mesh"].append(robot_hand.get_trimesh_data())
        #     E_pen, inner_pts, outer_pts = robot_hand.cal_object_penetration(pc_ls, True)
        #     history['ct_pts'].append(ct_pts.detach().cpu().numpy())
        #     history['corr_ct_pts'].append(cc_ct_pts.detach().cpu().numpy())
        #     history['inner_pts'].append(inner_pts.detach().cpu().numpy())
        #     history['outer_pts'].append(outer_pts.detach().cpu().numpy())
        #     loss = E_pen * pen_coeffs
        #     losses["E_pen"].append(E_pen.item())
        #     next_optimizer.zero_grad()
        #     loss.backward()
        #     next_optimizer.step()

        # from utils.wmq import vis_dexhand_optimize
        # filename = f"retarget/0912/(3)contact_penetration/contact_only_step{total_step}_lr{lr}_pen{pen_coeffs}_dis{dis_coeffs}_frame{check_frame}"
        # vis_dexhand_optimize(
        #     pc_ls=[np.tile(pc_ls, (total_step, 1, 1))],
        #     object_mesh_ls=[i*total_step for i in obj_mesh_ls],
        #     gt_hand_joints=seq_data['hand_joints'][check_frame].expand(total_step, -1, -1).cpu().numpy(),
        #     hand_mesh_ls=[[i[0] for i in history["mesh"]]],
        #     gt_posi_pts=mano_fingertip.expand(total_step, -1, -1).cpu().numpy(),
        #     posi_pts_ls=np.stack(history["keypoints"]),
        #     contact_pt_ls=np.concatenate(history["ct_pts"]),
        #     corr_contact_pt_ls=np.concatenate(history["corr_ct_pts"]),
        #     inner_pen_pts=history["inner_pts"],
        #     outer_pen_pts=history["outer_pts"],
        #     filename=filename
        # )
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # for key in losses:
        #     plt.plot(np.arange(len(losses[key])), losses[key], label=key)
        # plt.yscale('log')
        # plt.xlabel("Step")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.subplot(1, 2, 2)
        # inner_num = [len(i) for i in history["inner_pts"]]
        # plt.plot(np.arange(len(inner_num)), inner_num, label="inner points num")
        # plt.xlabel("Step")
        # plt.ylabel("Num")
        # plt.legend()
        # plt.savefig(f"{filename}.png")
        # plt.close()
        # import sys;sys.exit()
        # ### Debug penetration ###

        # ### Debug contact ###
        # for ii in tqdm(range(total_step), desc="Stage 2 optimization"):
        #     robot_hand.compute_forward_kinematics(dex_pose[check_frame])
        #     fingertip_keypoints = robot_hand.get_align_points()
        #     E_align = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip, reduction='sum')
        #     history["keypoints"].append(fingertip_keypoints.detach().cpu().numpy())
        #     history["mesh"].append(robot_hand.get_trimesh_data())
        #     E_dis, ct_pts, cc_ct_pts = robot_hand.cal_distance(pc_ls, pc_norm_ls, True)
        #     history['ct_pts'].append(ct_pts.detach().cpu().numpy())
        #     history['corr_ct_pts'].append(cc_ct_pts.detach().cpu().numpy())
        #     loss =  E_align + E_dis * dis_coeffs
        #     losses["E_align"].append(E_align.item())
        #     losses["E_dis"].append(E_dis.item())
        #     losses["total"].append(loss.item())
        #     next_optimizer.zero_grad()
        #     loss.backward()
        #     next_optimizer.step()

        # from utils.wmq import vis_dexhand_optimize
        # filename = f"retarget/0912/(3)contact_penetration/contact_only_step{total_step}_lr{lr}_pen{pen_coeffs}_dis{dis_coeffs}_frame{check_frame}"
        # vis_dexhand_optimize(
        #     pc_ls=[np.tile(pc_ls, (total_step, 1, 1))],
        #     object_mesh_ls=[i*total_step for i in obj_mesh_ls],
        #     gt_hand_joints=seq_data['hand_joints'][check_frame].expand(total_step, -1, -1).cpu().numpy(),
        #     hand_mesh_ls=[[i[0] for i in history["mesh"]]],
        #     gt_posi_pts=mano_fingertip.expand(total_step, -1, -1).cpu().numpy(),
        #     posi_pts_ls=np.stack(history["keypoints"]),
        #     contact_pt_ls=np.concatenate(history["ct_pts"]),
        #     corr_contact_pt_ls=np.concatenate(history["corr_ct_pts"]),
        #     filename=filename
        # )
        # from matplotlib import pyplot as plt
        # plt.figure()
        # for key in losses:
        #     plt.plot(np.arange(len(losses[key])), losses[key], label=key)
        # plt.yscale('log')
        # plt.xlabel("Step")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig(f"{filename}.png")
        # plt.close()
        # import sys;sys.exit()
        # ### Debug contact ###
        pass

    if True:
        # ### Debug params ###
        # for ii in tqdm(range(total_step), desc="Stage 2 optimization"):
        #     robot_hand.compute_forward_kinematics(dex_pose[check_frame])
        #     fingertip_keypoints = robot_hand.get_align_points()
        #     E_align = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip, reduction='sum')
        #     history["keypoints"].append(fingertip_keypoints.detach().cpu().numpy())
        #     history["mesh"].append(robot_hand.get_trimesh_data())
        #     E_dis, ct_pts, cc_ct_pts = robot_hand.cal_distance(pc_ls, pc_norm_ls, True)
        #     E_pen, inner_pts, outer_pts = robot_hand.cal_object_penetration(pc_ls, True)
        #     history['ct_pts'].append(ct_pts.detach().cpu().numpy())
        #     history['corr_ct_pts'].append(cc_ct_pts.detach().cpu().numpy())
        #     history['inner_pts'].append(inner_pts.detach().cpu().numpy())
        #     history['outer_pts'].append(outer_pts.detach().cpu().numpy())
        #     E_spen = robot_hand.self_penetration_part(thres)
        #     loss = E_spen * sp_coeffs + E_pen * pen_coeffs + E_dis * dis_coeffs + E_align
        #     losses["E_align"].append(E_align.item())
        #     losses["E_dis"].append(E_dis.item())
        #     losses["E_pen"].append(E_pen.item())
        #     losses["E_spen"].append(E_spen.item())
        #     losses["total"].append(loss.item())
        #     next_optimizer.zero_grad()
        #     loss.backward()
        #     next_optimizer.step()

        # from utils.wmq import vis_dexhand_optimize
        # filename = f"retarget/0912/(3)contact_penetration/params_step{total_step}_lr{lr}_thres{thres}_spcoeffs{sp_coeffs}_pen{pen_coeffs}_dis{dis_coeffs}_frame{check_frame}"
        # vis_dexhand_optimize(
        #     pc_ls=[np.tile(pc_ls, (total_step, 1, 1))],
        #     object_mesh_ls=[i*total_step for i in obj_mesh_ls],
        #     gt_hand_joints=seq_data['hand_joints'][check_frame].expand(total_step, -1, -1).cpu().numpy(),
        #     hand_mesh_ls=[[i[0] for i in history["mesh"]]],
        #     gt_posi_pts=mano_fingertip.expand(total_step, -1, -1).cpu().numpy(),
        #     posi_pts_ls=np.stack(history["keypoints"]),
        #     contact_pt_ls=np.concatenate(history["ct_pts"]),
        #     corr_contact_pt_ls=np.concatenate(history["corr_ct_pts"]),
        #     inner_pen_pts=history["inner_pts"],
        #     outer_pen_pts=history["outer_pts"],
        #     filename=filename
        # )
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # for key in losses:
        #     plt.plot(np.arange(len(losses[key])), losses[key], label=key)
        # plt.yscale('log')
        # plt.xlabel("Step")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.subplot(1, 2, 2)
        # inner_num = [len(i) for i in history["inner_pts"]]
        # plt.plot(np.arange(len(inner_num)), inner_num, label="inner points num")
        # plt.xlabel("Step")
        # plt.ylabel("Num")
        # plt.legend()
        # plt.savefig(f"{filename}.png")
        # plt.close()
        # import sys;sys.exit()
        # ### Debug params ###
        pass

    for ii in tqdm(range(total_step), desc="Stage 2 optimization"):
        robot_hand.compute_forward_kinematics(dex_pose)
        fingertip_keypoints = robot_hand.get_align_points()
        E_align = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip, reduction='sum')
        E_dis = robot_hand.cal_distance(pc_ls, pc_norm_ls)
        E_pen = robot_hand.cal_object_penetration(pc_ls)
        E_spen = robot_hand.self_penetration_part(thres)
        loss = E_spen * sp_coeffs + E_pen * pen_coeffs + E_dis * dis_coeffs + E_align
        next_optimizer.zero_grad()
        loss.backward()
        next_optimizer.step()

    import sys;sys.exit()

    retargeted_seq = dict(            
        which_hand=robot_hand.robot_name,
        hand_poses=dex_pose,
        side=seq_data["side"],

        hand_tsls=seq_data["hand_tsls"],
        hand_coeffs=seq_data["hand_coeffs"],
        hand_joints=seq_data["hand_joints"],

        obj_poses=seq_data["obj_poses"],
        obj_point_clouds=seq_data["obj_point_clouds"],
        obj_norms=seq_data["obj_norms"],
        # obj_feature=seq_data["obj_feature"],
        object_names=seq_data["object_names"],
        object_mesh_path=seq_data["object_mesh_path"],

        frame_indices=seq_data["frame_indices"],
        task_description=seq_data["task_description"],
        which_dataset=seq_data["which_dataset"],
        which_sequence=seq_data["which_sequence"],
        extra_info=seq_data["extra_info"]
        )

    return retargeted_seq


def main_retarget(seq_data_ls, robots):
    processed_data = {}
    for robot_name in robots:
        robot_hand = load_robot(robot_name)
        print(f"Retargeting to {robot_hand.robot_name} ...")
        retargeted_data = []
        for i, seq_data in enumerate(seq_data_ls):
            print(f"Processing sequence with {robot_name} from {seq_data['which_dataset']} with {seq_data['which_sequence']}")
            retargeted_seq = retarget_sequence(seq_data, robot_hand)
            retargeted_data.append(retargeted_seq)
        processed_data[robot_hand] = retargeted_data
    return processed_data

if __name__ == "__main__":
    import os
    import pickle
    from tqdm import tqdm

    robots = ['shadow_hand']
    file_path = "/home/qianxu/Desktop/Project/DexPose/data_dict_wqx_1.pth"
    seq_data_ls = torch.load(file_path)[16:17]
    processed_data = main_retarget(seq_data_ls, robots)