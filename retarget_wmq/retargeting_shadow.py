import os
import torch
import numpy as np
import pytorch3d.transforms as T
from manotorch.manolayer import ManoLayer, MANOOutput
import sys; sys.path.append('.')
# from shadowhand import ShallowHandModel
from data_process.dataset import PoseDataset, collate_fn, StreamDataset
from torch.utils.data import DataLoader
import tqdm
import random
from tools import vis_pc_coor_plotly, vis_frames_plotly, rotation_6d_to_matrix, matrix_to_rotation_6d
from hand_model import HandModelMJCF, cal_distance
from scipy.spatial.transform import Rotation as R
import time

def normalize_6Dpose(pose:torch.Tensor, transformations:torch.Tensor):
    """
    Normalize rotations given as rot6d.

    Args:
        rot6d: rotations as tensor of shape (..., 9).
    """
    rotation_matrix = rotation_6d_to_matrix(pose[:, 3:])
    base_transformations = torch.eye(4)[None, ...].repeat(pose.shape[0], 1, 1).to(pose.device)
    base_transformations[:, :3, :3] = rotation_matrix
    base_transformations[:, :3, 3] = pose[:, :3]
    norm_transformations = transformations @ base_transformations
    norm_rot6d = matrix_to_rotation_6d(norm_transformations[:, :3, :3])
    norm_position = norm_transformations[:, :3, 3]
    return torch.cat([norm_position, norm_rot6d], dim=-1)

def calculate_velocity(poses):
    """
    Calculate the velocity of a sequence of poses.

    Args:
        poses: poses as tensor of shape (..., T, 3).

    Returns:
        Velocities as tensor of shape (..., T - 1, 3).
    """
    return poses[..., 1:, :] - poses[..., :-1, :]

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (torch.sin(half_angles[~small_angles]) / angles[~small_angles])
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (0.5 - (angles[small_angles] * angles[small_angles]) / 48)
    return quaternions[..., 1:] / sin_half_angles_over_angles

def mano2dex_batch(manopose):
    # to eurler
    batch_size = manopose.shape[0]
    manoaxisangle = manopose.reshape(batch_size, -1, 3)
    manoeurler = T.matrix_to_euler_angles(T.axis_angle_to_matrix(manoaxisangle),"XYZ") # batch_size, 16, 3

    dexpose = torch.zeros(batch_size, 22).to(manopose.device)
    # forefinger, index finger
    dexpose[:, 0], dexpose[:, 1]  = manoeurler[:, 1, 1], manoeurler[:, 1, 2]
    dexpose[:, 2] = manoeurler[:, 2, 2]
    dexpose[:, 3] = manoeurler[:, 3, 2]
    # middle finger
    dexpose[:, 4], dexpose[:, 5]  = manoeurler[:, 4, 1]-0.2, manoeurler[:, 4, 2]
    dexpose[:, 6] = manoeurler[:, 5, 2]
    dexpose[:, 7] = manoeurler[:, 6, 2]
    # ring finger
    dexpose[:, 8], dexpose[:, 9]  = manoeurler[:, 10, 1]-0.4, manoeurler[:, 10, 2]
    dexpose[:, 10] = manoeurler[:, 11, 2]
    dexpose[:, 11] = manoeurler[:, 12, 2]
    # little finger
    dexpose[:, 12] =  0
    dexpose[:, 13], dexpose[:, 14]  = manoeurler[:, 7, 1]-0.6, manoeurler[:, 7, 2]
    dexpose[:, 15] = manoeurler[:, 8, 2]
    dexpose[:, 16] = manoeurler[:, 9, 2]
    # thumb
    dexpose[:, 17], dexpose[:, 18]  = manoeurler[:, 13, 0], manoeurler[:, 13, 2]
    dexpose[:, 19], dexpose[:, 20] = manoeurler[:, 14, 2], manoeurler[:, 14, 1]
    dexpose[:, 21] = manoeurler[:, 15, 1]

    return dexpose

def get_angle(v, transformation,):
    x_axis, y_axis, z_axis = transformation[:, :3, 0], transformation[:, :3, 1], transformation[:, :3, 2]
    v_z = torch.sum(v * z_axis, dim=-1, keepdim=True) * z_axis
    v_xoy = v - v_z
    z_angle = torch.acos(torch.sum(v_xoy * (-x_axis), dim=-1) / (torch.norm(v_xoy, dim=-1) * torch.norm(-x_axis, dim=-1))) * torch.sign(torch.sum(v_xoy * -y_axis, dim=-1))
    v_y = torch.sum(v * y_axis, dim=-1, keepdim=True) * y_axis
    v_zox = v - v_y
    y_angle = torch.acos(torch.sum(v_zox * (-x_axis), dim=-1) / (torch.norm(v_zox, dim=-1) * torch.norm(-x_axis, dim=-1))) * torch.sign(torch.sum(v_zox * x_axis, dim=-1))
    v_x = torch.sum(v * x_axis, dim=-1, keepdim=True) * x_axis
    v_yoz = v - v_x
    x_angle = torch.acos(torch.sum(v_yoz * y_axis, dim=-1) / (torch.norm(v_yoz, dim=-1) * torch.norm(y_axis, dim=-1))) * torch.sign(torch.sum(v_yoz * z_axis, dim=-1))
    return torch.stack([x_angle, y_angle, z_angle], dim=0)

def joint2dex_batch(joint:torch.Tensor, transformation:torch.Tensor):

    dexpose = torch.zeros(joint.shape[0], 22).to(joint.device)
    ### index finger
    index_0 = joint[:, 5] - joint[:, 0]
    index_1 = joint[:, 6] - joint[:, 5]
    index_2 = joint[:, 7] - joint[:, 6]
    index_3 = joint[:, 8] - joint[:, 7]
    # dexpose[:, 0] = get_angle(index_1, transformation)[1]
    # dexpose[:, 1] = abs(torch.acos(torch.sum(index_1 * index_0, dim=-1) / (torch.norm(index_0, dim=-1) * torch.norm(index_1, dim=-1))))
    dexpose[:, 2] = abs(torch.acos(torch.sum(index_2 * index_1, dim=-1) / (torch.norm(index_1, dim=-1) * torch.norm(index_2, dim=-1))))
    dexpose[:, 3] = abs(torch.acos(torch.sum(index_3 * index_2, dim=-1) / (torch.norm(index_2, dim=-1) * torch.norm(index_3, dim=-1))))
    dexpose[:, 0] = get_angle(index_1, transformation)[1]
    dexpose[:, 1] = get_angle(index_1, transformation)[2]
    # dexpose[:, 2] = get_angle(index_2, transformation)[2]
    # dexpose[:, 3] = get_angle(index_3, transformation)[2]
                        
    ### middle finger
    middle_0 = joint[:, 9] - joint[:, 0]
    middle_1 = joint[:, 10] - joint[:, 9]
    middle_2 = joint[:, 11] - joint[:, 10]
    middle_3 = joint[:, 12] - joint[:, 11]
    # dexpose[:, 4] = 0.01 -0.2 # temporary
    # dexpose[:, 5] = abs(torch.acos(torch.sum(middle_1 * middle_0, dim=-1) / (torch.norm(middle_0, dim=-1) * torch.norm(middle_1, dim=-1))))
    dexpose[:, 6] = abs(torch.acos(torch.sum(middle_2 * middle_1, dim=-1) / (torch.norm(middle_1, dim=-1) * torch.norm(middle_2, dim=-1))))
    dexpose[:, 7] = abs(torch.acos(torch.sum(middle_3 * middle_2, dim=-1) / (torch.norm(middle_2, dim=-1) * torch.norm(middle_3, dim=-1))))
    dexpose[:, 4] = -get_angle(middle_1, transformation)[1] - 0.2
    dexpose[:, 5] = get_angle(middle_1, transformation)[2]
    # dexpose[:, 6] = get_angle(middle_2, transformation)[2]
    # dexpose[:, 7] = get_angle(middle_3, transformation)[2]

    ### ring finger
    ring_0 = joint[:, 13] - joint[:, 0]
    ring_1 = joint[:, 14] - joint[:, 13]
    ring_2 = joint[:, 15] - joint[:, 14]
    ring_3 = joint[:, 16] - joint[:, 15]
    # dexpose[:, 8] = 0.01-0.4 # temporary
    # dexpose[:, 9] = abs(torch.acos(torch.sum(ring_1 * ring_0, dim=-1) / (torch.norm(ring_0, dim=-1) * torch.norm(ring_1, dim=-1))))
    dexpose[:, 10] = abs(torch.acos(torch.sum(ring_2 * ring_1, dim=-1) / (torch.norm(ring_1, dim=-1) * torch.norm(ring_2, dim=-1))))
    dexpose[:, 11] = abs(torch.acos(torch.sum(ring_3 * ring_2, dim=-1) / (torch.norm(ring_2, dim=-1) * torch.norm(ring_3, dim=-1))))
    dexpose[:, 8] = -get_angle(ring_1, transformation)[1] - 0.4
    dexpose[:, 9] = get_angle(ring_1, transformation)[2]
    # dexpose[:, 10] = get_angle(ring_2, transformation)[2]
    # dexpose[:, 11] = get_angle(ring_3, transformation)[2]

    ### little finger
    little_0 = joint[:, 17] - joint[:, 0]
    little_1 = joint[:, 18] - joint[:, 17]
    little_2 = joint[:, 19] - joint[:, 18]
    little_3 = joint[:, 20] - joint[:, 19]
    # dexpose[:, 12] =  0
    # dexpose[:, 13] = 0.01 -0.6# temporary
    # dexpose[:, 14] = abs(torch.acos(torch.sum(little_1 * little_0, dim=-1) / (torch.norm(little_0, dim=-1) * torch.norm(little_1, dim=-1))))
    dexpose[:, 15] = abs(torch.acos(torch.sum(little_2 * little_1, dim=-1) / (torch.norm(little_1, dim=-1) * torch.norm(little_2, dim=-1))))
    dexpose[:, 16] = abs(torch.acos(torch.sum(little_3 * little_2, dim=-1) / (torch.norm(little_2, dim=-1) * torch.norm(little_3, dim=-1))))
    dexpose[:, 12] = 0
    dexpose[:, 13] = -get_angle(little_1, transformation)[1] - 0.6
    dexpose[:, 14] = get_angle(little_1, transformation)[2]
    # dexpose[:, 15] = get_angle(little_2, transformation)[2]
    # dexpose[:, 16] = get_angle(little_3, transformation)[2]

    ### thumb
    thumb_0 = joint[:, 1] - joint[:, 0]
    thumb_1 = joint[:, 2] - joint[:, 1]
    thumb_2 = joint[:, 3] - joint[:, 2]
    thumb_3 = joint[:, 4] - joint[:, 3]
    # dexpose[:, 17] = 0.01 # temporary
    # dexpose[:, 18] = abs(torch.acos(torch.sum(thumb_1 * thumb_0, dim=-1) / (torch.norm(thumb_0, dim=-1) * torch.norm(thumb_1, dim=-1))))
    dexpose[:, 19] = -abs(torch.acos(torch.sum(thumb_2 * thumb_1, dim=-1) / (torch.norm(thumb_1, dim=-1) * torch.norm(thumb_2, dim=-1))))
    dexpose[:, 20] = 0
    dexpose[:, 21] = abs(torch.acos(torch.sum(thumb_3 * thumb_2, dim=-1) / (torch.norm(thumb_2, dim=-1) * torch.norm(thumb_3, dim=-1))))
    dexpose[:, 17] = -get_angle(thumb_1, transformation)[0]
    dexpose[:, 18] = -get_angle(thumb_1, transformation)[2]
    # dexpose[:, 20] = get_angle(thumb_2, transformation)[1]
    # dexpose[:, 21] = get_angle(thumb_3, transformation)[1]
    return dexpose

def optimize_batch(manograsp, shadow_hand:HandModelMJCF, optimized_step=50, 
                   contact_pts=None, contact_normals=None,
                   with_initialization=True, norm_transform=None, device="cuda"):

    global_translation = manograsp["hand_joints"][:, 0].to(device) 
    y_ratation = torch.from_numpy(R.from_euler('y', - np.pi / 16, degrees=False).as_matrix()).to(torch.float32).to(device)
    global_rotation = manograsp["transformation"][:, :3, :3].to(device) @ torch.tensor([[0.0,0,-1],[0,1,0],[1,0,0]]).to(device) @ y_ratation
    global_rotat6d = torch.cat([global_rotation[..., 0], global_rotation[..., 1]], dim=-1)


    # hand_pose_axis_angle = quaternion_to_axis_angle(manograsp["hand_pose"].to(device))
    # hand_pose_axis_angle  = manograsp["hand_pose"].to(device)
    # global_axis_angle = hand_pose_axis_angle[:, 0]
    # global_axis_angle = T.matrix_to_axis_angle(T.axis_angle_to_matrix(global_axis_angle) 
    #                                             @ torch.tensor([[0.0,0,-1],[0,1,0],[1,0,0]]).to(device))
    # global_m_np = R.from_rotvec(global_axis_angle.cpu().detach().numpy()).as_matrix()
    # if norm_transform is not None:
    #     global_m_np = np.matmul(norm_transform.cpu().numpy()[:, :3, :3], global_m_np)
    # global_rotat6d_np = np.concatenate([global_m_np[..., 0], global_m_np[..., 1]], axis=-1)
    # global_rotat6d = torch.tensor(global_rotat6d_np).to(device)
    
    # joint_eurler = mano2dex_batch(hand_pose_axis_angle)

    if with_initialization:
        dex_pose = torch.cat([joint2dex_batch(manograsp["hand_joints"], manograsp["transformation"]).cpu()],dim=-1).to(device).to(torch.float32)
    else:
        dex_pose = torch.cat([global_translation.cpu(), global_rotat6d.cpu(), torch.randn(global_translation.shape[0], 22) * 0.1 + 0.1],dim=-1).to(device).to(torch.float32)
    dex_pose.requires_grad_()

    optimizer = torch.optim.Adam([dex_pose], lr=0.01, weight_decay=0)

    mano_fingertip = manograsp["hand_joints"][:, [ 8, 12, 16, 20,  4]].to(device)
    contact_pts = contact_pts.to(device)
    contact_normals = contact_normals.to(device)
    n_contact = 100
    switch_possibility = 0.5
    contact_point_indices = torch.randint(hand.n_contact_candidates, size=[contact_pts.shape[0], n_contact], device=device)
    # return dex_pose
    for ii in range(optimized_step):
        shadow_hand.set_parameters(torch.cat([global_translation, global_rotat6d, dex_pose], dim=-1), retarget=True, robust=True, contact_point_indices=contact_point_indices)
        fingertip_keypoints = shadow_hand.get_tip_points()

        distances = hand.cal_distance(contact_pts)
        distances[distances <= 0] = 0
        E_pen = distances.sum(-1)
        E_spen = hand.self_penetration()
        E_joint = hand.get_E_joints()
        contact_distance = cal_distance(contact_pts, contact_normals, hand.contact_points)
        E_dis = torch.sum(contact_distance, dim=-1, dtype=torch.float).to(device)
        
        loss = torch.nn.functional.huber_loss(
                            fingertip_keypoints, mano_fingertip, reduction='sum')
        if ii > 80:
            loss += (E_pen + E_spen ).mean()
        # if ii > 120:
        #     loss += E_dis.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = dex_pose.shape[0]
        switch_mask = torch.rand(batch_size, n_contact, dtype=torch.float32, device=device) < switch_possibility
        contact_point_indices = hand.contact_point_indices.clone()
        contact_point_indices[switch_mask] = torch.randint(hand.n_contact_candidates, size=[switch_mask.sum()], device=device)
    
    return torch.cat([global_translation, global_rotat6d, dex_pose], dim=-1)

def random_select_index(valid_indices_sorted, padding_indices, n_contact=140, n_select=30):
    BS = valid_indices_sorted.shape[0]
    for i in range(BS):
        if torch.count_nonzero(valid_indices_sorted[i] == n_contact) == 0:
            first_index = n_contact
        else:
            first_index = torch.nonzero(valid_indices_sorted[i] == n_contact, as_tuple=True)[0][0].item()
        shuffle_indices = torch.cat([torch.randperm(first_index), torch.arange(first_index, n_contact)], dim=0)
        valid_indices_sorted[i] = valid_indices_sorted[i][shuffle_indices]
    valid_count_per_batch = (valid_indices_sorted < n_contact).sum(dim=1)  # Count of valid indices per batch
    # padding_indices = valid_indices_sorted[:, :1].expand(BS, 20)  # Shape (bs, 20), padding with the first valid index

    selected_indices = torch.cat([
        valid_indices_sorted[:, :valid_count_per_batch],  
        padding_indices
    ], dim=1)[:, :n_select] # Take the first 20 valid indices
    return selected_indices

def optimize_descent_batch(manograsp, shadow_hand:HandModelMJCF, optimized_step=50, 
                   contact_pts=None, contact_normals=None, transformations=None,
                   mano_layer=True, device="cuda"):
    
    norm_transformations = torch.inverse(transformations)
    mano_layer = mano_layer.to(device)
    global_translation = manograsp["hand_joints"][:, 0].to(device) 
    global_rotvec = torch.from_numpy(R.from_matrix(manograsp["transformation"][:, :3, :3].cpu().detach().numpy()).as_rotvec()).to(torch.float32).to(device)
    hand_coeff = (torch.zeros(global_translation.shape[0], 45).to(device)).requires_grad_()
    beta = torch.zeros(global_translation.shape[0], 10).to(device).requires_grad_()
    hand_optimizer = torch.optim.Adam([hand_coeff, beta], lr=0.05, weight_decay=0)
    goal_joints = manograsp["hand_joints"].to(device) - global_translation[:, None, :]
    for ii in range(150):
        hand_optimizer.zero_grad()
        hand_dict = mano_layer(torch.cat([global_rotvec, hand_coeff], dim=-1), beta)
        joints = hand_dict.joints
        loss = torch.nn.functional.huber_loss(joints, goal_joints, reduction='sum')
        loss.backward()
        hand_optimizer.step()
    # vis_frames_plotly(None, gt_hand_joints=goal_joints.cpu().detach().numpy(), hand_joints_ls=[joints.cpu().detach().numpy()])
    hand_pose_axis_angle = torch.cat([global_rotvec, hand_coeff.detach()],  dim=-1).reshape(-1, 16, 3)

    # hand_pose_axis_angle = quaternion_to_axis_angle(manograsp["hand_pose"].to(device))
    global_axis_angle = hand_pose_axis_angle[:, 0]
    global_axis_angle = T.matrix_to_axis_angle(T.axis_angle_to_matrix(global_axis_angle) 
                                                @ torch.tensor([[0.0,0,-1],[0,1,0],[1,0,0]]).to(device))
    global_m_np = R.from_rotvec(global_axis_angle.cpu().detach().numpy()).as_matrix()
    global_rotat6d_np = np.concatenate([global_m_np[..., 0], global_m_np[..., 1]], axis=-1)
    global_rotat6d = torch.tensor(global_rotat6d_np).to(device)
    
    joint_eurler = mano2dex_batch(hand_pose_axis_angle)

    dex_pose = torch.cat([global_translation, global_rotat6d, joint_eurler],dim=-1).to(device).to(torch.float32)
    dex_pose.requires_grad_()

    optimizer = torch.optim.Adam([dex_pose], lr=0.01, weight_decay=0)

    # mano_fingertip = manograsp["hand_joints"][:, [8, 12, 16, 20,  4]].to(device)
    mano_fingertip = manograsp["hand_joints"][:, [5, 8, 9, 12, 13, 16, 17, 20, 4]].to(device)
    contact_pts = contact_pts.to(device)
    contact_normals = contact_normals.to(device)
    n_contact = 50
    switch_possibility = 0.5
    contact_point_indices = torch.arange(n_contact, device=device).repeat(dex_pose.shape[0], 1)
    # return dex_pose
    for ii in range(optimized_step):
        shadow_hand.set_parameters(dex_pose, retarget=True, robust=True, contact_point_indices=contact_point_indices)
        # contact_points = shadow_hand.contact_points
        # hand_mesh = shadow_hand.get_trimesh_data(0)
        # if ii == 0:
        #     vis_pc_coor_plotly(None, [contact_pts[0].cpu().detach().numpy()], hand_mesh=hand_mesh)

        fingertip_keypoints = shadow_hand.get_tip_points()
        loss = torch.nn.functional.huber_loss(
                            fingertip_keypoints, mano_fingertip, reduction='sum')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ### contact indices: (B, N)
    all_contact_indices = torch.arange(n_contact, dtype=torch.long, device=device).unsqueeze(0).expand(dex_pose.shape[0], -1)
    shadow_hand.set_parameters(dex_pose, retarget=True, robust=True, contact_point_indices=all_contact_indices)
    contact_candidates = shadow_hand.contact_points
    candidate_object_distance = torch.norm(contact_candidates[:, :, None, :] - contact_pts[:, None, :, :], dim=-1, p=2).min(dim=-1)[0]
    candidate_object_distance = candidate_object_distance.mean(dim=0, keepdim=True)
    BS = candidate_object_distance.shape[0] # 1
    
    contact_threshold = 0.03
    mask = candidate_object_distance < contact_threshold  # Shape (bs, 140)
    indices = torch.arange(n_contact).unsqueeze(0).expand(BS, -1).to(contact_candidates.device)  # Shape (bs, 140)
    valid_indices = torch.where(mask, indices, torch.tensor(n_contact, device=contact_candidates.device))  # Shape (bs, 140)
    valid_indices_sorted, _ = valid_indices.sort(dim=1)
    th_indices = torch.nonzero(shadow_hand.thumb_contact_points_indices).squeeze(1).unsqueeze(0).expand(BS, -1)
    
    # return dex_pose
    for ii in range(optimized_step * 2):
        if ii % 50 == 0:
            selected_indices = random_select_index(valid_indices_sorted, th_indices, n_contact=n_contact, n_select=30).expand(dex_pose.shape[0], -1)
            if torch.count_nonzero(selected_indices == n_contact) > 0:
                print('No valid contact points')
                selected_indices = torch.zeros_like(selected_indices).to(device).to(selected_indices.dtype)

        shadow_hand.set_parameters(dex_pose, retarget=True, robust=True, contact_point_indices=selected_indices)
        # contact_points = shadow_hand.contact_points
        # hand_mesh = shadow_hand.get_trimesh_data(0)
        # if ii % 50 == 0:
        #     vis_pc_coor_plotly([contact_points[0].cpu().detach().numpy()], [contact_pts[0].cpu().detach().numpy()], hand_mesh=hand_mesh)

        fingertip_keypoints = shadow_hand.get_tip_points()

        distances = shadow_hand.cal_distance(contact_pts)
        distances[distances <= 0] = 0
        E_pen = distances.sum(-1)
        E_spen = shadow_hand.self_penetration()
        E_joint = shadow_hand.get_E_joints()
        contact_distance = cal_distance(contact_pts, contact_normals, shadow_hand.contact_points)
        E_dis = torch.sum(contact_distance, dim=-1, dtype=torch.float).to(device)

        loss = (E_pen + E_spen + E_joint + E_dis).mean() + torch.nn.functional.huber_loss(
                            fingertip_keypoints, mano_fingertip, reduction='sum') * 1 
                # + E_velocity
        # print((E_pen + E_spen + E_joint + E_dis).mean(), E_velocity, E_acceleration)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return dex_pose, shadow_hand.contact_points

def retarget(hand_joints, fixed_indices_ls, contact_pts, contact_normals, transformations, hand, mano_layer_right, device='cuda'):
    hand_key_joints = hand_joints[:, fixed_indices_ls]# A^T
    translation = hand_key_joints[:, 0]
    hand_key_joints = hand_key_joints - translation[:, None, :]
    h_dict = mano_layer_right(torch.zeros(1, 48).to(device), torch.zeros(1, 10).to(device))
    hand_kp_bases = h_dict.joints[:, fixed_indices_ls].to(device) # B^T

    ## orthogonal prorustes
    M = torch.matmul(hand_kp_bases.mT, hand_key_joints)
    U, _, V = torch.svd(M)
    Rotation = torch.matmul(U, V.mT)
    transformation = torch.eye(4)[None, ...].repeat(hand_key_joints.shape[0], 1, 1).to(hand_key_joints.device)
    transformation[:, :3, :3] = torch.inverse(Rotation)
    transformation[:, :3, 3] = translation
    # estimated_hand_joints = torch.matmul(h_dict.joints, torch.inverse(Rotation).mT) + translation[:, None, :]
    # vis_pc_coor_plotly(None, gt_posi_pts=item['hand_joints'][0][0][fixed_indices_ls].cpu().detach().numpy(), 
    #                    posi_pts_ls=[estimated_hand_joints[0][fixed_indices_ls].cpu().detach().numpy()], 
    #                    gt_transformation_ls=[transformation[0].cpu().detach().numpy()])

    manograsp = {"transformation": transformation,
                    "hand_joints": hand_joints}
    return optimize_descent_batch(manograsp=manograsp, shadow_hand=hand, optimized_step=50, 
                                mano_layer=mano_layer_right, device=device, transformations=transformations, 
                                contact_pts=contact_pts, contact_normals=contact_normals)

if __name__ == '__main__':
    manual_seed = 0
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    fixed_indices_ls = [0, 1, 5, 9, 13, 17]
    mano_layer_right = ManoLayer(center_idx=0, side='right', use_pca=False).to('cuda') # B^T


    # data_ls = os.listdir('/juno/u/qianxu/dataset/OakInk-v2/data')
    # set CUDA_LAUNCH_BLOCKING=1
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    data_ls = [
        # '/home/qianxu/project/interaction_pose/data/seq_data/seq_oakinkv2_1.p',
            #  '/home/qianxu/project/interaction_pose/data/seq_data/seq_taco_1.p',
            '/home/qianxu/Desktop/Project/interaction_pose/data/Taco/processed/seq_taco_2_train.p'
             ]

    # dataset = PoseDataset(data_ls=data_ls, train=True, retarget=True)
    dataset = StreamDataset(data_ls=data_ls, train=True,)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=0, collate_fn=collate_fn)
    # 25621
    hand_rot_ls = []
    hand_posi_ls =  []
    device = torch.device('cuda')
    hand = HandModelMJCF("./mjcf/shadow_hand_wrist_free.xml", "./mjcf/meshes", 
                         n_surface_points=500, device=torch.device('cuda:0'), tip_aug=None, ref_points=None)
    interval = 1
    while True:
        for item in tqdm.tqdm(dataloader):

            side = item['side'].reshape(-1).to(torch.bool).to(device)
            BS = torch.sum(side).item()
            for i in range(BS):
                hand_joints = item['hand_joints'].to(device)[side][i] # A^T
                object_pts = item['obj_pts_ori'].to(device)[side][i]
                object_normals = item['obj_normals_ori'].to(device)[side][i]
                transformations = item['obj_transformations'].to(device)[side][i]
                TT, N, _ = object_pts.shape
                dex_pose = retarget(hand_joints, fixed_indices_ls, object_pts, object_normals, transformations, hand, mano_layer_right, device)

                hand.set_parameters(dex_pose, retarget=True, robust=True)
                hand_mesh_ls = [hand.get_trimesh_data(ii) for ii in range(0, TT, interval)]
                # vis_pc_coor_plotly([object_pts[0].cpu().numpy()], 
                #                    gt_posi_pts=hand_joints[0].cpu().detach().numpy(),
                #                    hand_mesh=hand_mesh_ls[0],)
                vis_frames_plotly([object_pts[::interval].cpu().numpy()], gt_hand_joints=hand_joints[::interval].cpu().numpy(),
                        hand_mesh=hand_mesh_ls, filename='retargeting')
                vis_frames_plotly([object_pts[::interval].cpu().numpy()], gt_hand_joints=hand_joints[::interval].cpu().numpy(),
                        )
                break


            