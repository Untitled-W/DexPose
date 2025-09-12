import os
import pickle
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
import torch
from pytorch3d.transforms import matrix_to_quaternion
import time
from utils.hand_model import HandModelURDF

# setup_isaacgym_environment 函数保持不变...
def setup_isaacgym_environment():
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim = gym.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_params)
    if sim is None:
        raise ValueError("Failed to create sim")
    
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    cam_props = gymapi.CameraProperties()
    cam_props.width = 1920
    cam_props.height = 1080
    viewer = gym.create_viewer(sim, cam_props)
    if viewer is None:
        raise ValueError("Failed to create viewer")
        
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1, -1, 2), gymapi.Vec3(0.5, 0.5, 0))

    return gym, sim, viewer

def run_visualization_grid(all_dex_data):
    """
    加载所有数据序列，并将它们创建为并行的Isaac Gym环境网格进行可视化。
    """
    gym, sim, viewer = setup_isaacgym_environment()

    # --- 1. 资产加载 ---
    first_data = all_dex_data[0]
    hand_asset_root = os.path.join("/home/qianxu/Desktop/Project/DexPose/thirdparty/dex-retargeting/assets/robots/hands", first_data['which_hand'])
    side = 'left' if first_data['side'] == 0 else 'right'
    hand_asset_file = f"new_{side}_glb.urdf"

    hand_asset_options = gymapi.AssetOptions()
    hand_asset_options.fix_base_link = True
    hand_asset_options.disable_gravity = True
    hand_asset_options.use_physx_armature = True
    hand_asset_options.flip_visual_attachments = False
    hand_asset = gym.load_asset(sim, hand_asset_root, hand_asset_file, hand_asset_options)
    
    num_hand_dofs = gym.get_asset_dof_count(hand_asset)
    print(f"Hand asset loaded. Number of DOFs: {num_hand_dofs}")

    object_asset_cache = {}

    # --- 2. 环境创建 ---
    num_envs = len(all_dex_data)
    envs_per_row = int(np.sqrt(num_envs))
    env_spacing = 1.0
    
    envs, all_hand_actors, all_object_actors = [], [], []

    print(f"Creating {num_envs} environments in a grid...")
    for i, dex_seq_data in enumerate(all_dex_data):
        row = i // envs_per_row
        col = i % envs_per_row
        env_lower = gymapi.Vec3(col * env_spacing, row * env_spacing, 0.0)
        env_upper = gymapi.Vec3(col * env_spacing + env_spacing, row * env_spacing + env_spacing, env_spacing)
        
        env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        envs.append(env)

        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        hand_pose.r = gymapi.Quat(0, 0, 0, 1)
        hand_actor = gym.create_actor(env, hand_asset, hand_pose, f"hand_{i}", i, 0)
        all_hand_actors.append(hand_actor)

        object_actors_in_env = []
        asset_root, asset_file = [], []
        for path in dex_seq_data["object_mesh_path"]:
            asset_root.append('/home/qianxu/Desktop/Project/DexPose/data/Taco/object_models')
            asset_file.append(path.split('/')[-1][:-3] + 'urdf')

        for root, file in zip(asset_root, asset_file):
            asset_path = os.path.join(root, file)
            if asset_path not in object_asset_cache:
                obj_asset_options = gymapi.AssetOptions()
                obj_asset_options.vhacd_enabled = True
                obj_asset_options.vhacd_params = gymapi.VhacdParams()
                obj_asset_options.vhacd_params.resolution = 1000000
                obj_asset_options.density = 100
                object_asset_cache[asset_path] = gym.load_asset(sim, root, file, obj_asset_options)
            
            obj_handle = object_asset_cache[asset_path]
            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            obj_pose.r = gymapi.Quat(0, 0, 0, 1)
            obj_actor = gym.create_actor(env, obj_handle, obj_pose, f"object_{i}_{len(object_actors_in_env)}", i, 0)
            object_actors_in_env.append(obj_actor)

        all_object_actors.append(object_actors_in_env)

    # --- 3. 数据准备 ---
    T = all_dex_data[0]['obj_poses'].shape[1]
    all_hand_qpos = np.zeros((num_envs, T, num_hand_dofs), dtype=np.float32)
    all_obj_poses = []

    robot = HandModelURDF(first_data['which_hand'], os.path.join(hand_asset_root, f"new_{side}_glb.urdf"), os.path.join(hand_asset_root, 'meshes'))
    isaac_name_to_idx = {gym.get_asset_dof_name(hand_asset, i): i for i in range(num_hand_dofs)}
    pk_name_to_idx = {name: idx for idx, name in enumerate(robot.joints_names)}
    
    print("Preparing batched data...")
    for i, dex_seq_data in enumerate(all_dex_data):
        pk_hand_qpos = dex_seq_data['hand_poses']
        for pk_name, pk_idx in pk_name_to_idx.items():
            if pk_name in isaac_name_to_idx:
                isaac_idx = isaac_name_to_idx[pk_name]
                all_hand_qpos[i, :, isaac_idx] = pk_hand_qpos[:, pk_idx]
        
        # 将numpy数组转换为torch张量再append
        all_obj_poses.append(dex_seq_data['obj_poses'])

    all_obj_poses = torch.stack(all_obj_poses)
    all_hand_qpos = torch.from_numpy(all_hand_qpos)
    
    # --- 4. 仿真循环 ---
    gym.prepare_sim(sim) # 准备仿真资源，这对于获取正确的张量和索引很重要

    _root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_tensor = gymtorch.wrap_tensor(_root_tensor)

    _dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state_tensor = gymtorch.wrap_tensor(_dof_state_tensor)
    dof_pos_tensor = dof_state_tensor.view(num_envs, num_hand_dofs, 2)[..., 0]
    dof_vel_tensor = dof_state_tensor.view(num_envs, num_hand_dofs, 2)[..., 1]
    
    # === 新增：预计算所有演员的精确全局索引 ===
    actor_indices = {}
    print("Pre-calculating actor indices...")
    for i, env in enumerate(envs):
        hand_idx = gym.get_actor_index(env, all_hand_actors[i], gymapi.DOMAIN_SIM)
        obj_indices = [gym.get_actor_index(env, obj_actor, gymapi.DOMAIN_SIM) for obj_actor in all_object_actors[i]]
        actor_indices[i] = {'hand': hand_idx, 'objects': obj_indices}
    # ===============================================

    K = all_obj_poses.shape[1]
    SIM = True

    start_idx, end_idx = 50, 100
    # start_idx, end_idx = 0, T
    t = start_idx
    
    height_offset = torch.tensor([0.0, 0.0, 1.0])

    print("Starting simulation loop...")
    while not gym.query_viewer_has_closed(viewer):
        
        if t == start_idx or not SIM:
            ts = all_obj_poses[:, :, t, :3, 3].float()
            Rs = all_obj_poses[:, :, t, :3, :3].float()
            
            quats = matrix_to_quaternion(Rs.view(-1, 3, 3)).view(num_envs, K, 4)
            quats = quats[..., [1, 2, 3, 0]]

            # === 修改：使用精确索引更新 root_tensor ===
            for i in range(num_envs):
                # 更新手的位置（虽然是固定的，但重置是好习惯）
                hand_global_idx = actor_indices[i]['hand']
                root_tensor[hand_global_idx, 0:3] = height_offset
                root_tensor[hand_global_idx, 3:7] = torch.tensor([0, 0, 0, 1]) # 默认姿态
                root_tensor[hand_global_idx, 7:13] = 0.0

                # 更新物体位置
                object_global_indices = actor_indices[i]['objects']
                for k in range(K):
                    if k < len(object_global_indices):
                        actor_flat_index = object_global_indices[k]
                        root_tensor[actor_flat_index, 0:3] = ts[i, k] + height_offset
                        root_tensor[actor_flat_index, 3:7] = quats[i, k]
                        root_tensor[actor_flat_index, 7:13] = 0
            # ===============================================

            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))

        dof_pos_tensor[:, :] = all_hand_qpos[:, t, :]
        dof_vel_tensor[:, :] = 0.0
        
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state_tensor))

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        time.sleep(0.01)

        t = (t + 1 - start_idx) % (end_idx - start_idx) + start_idx

    print("Closing. Cleaning up...")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    hand = 'shadow'
    pickle_path = f"/home/qianxu/Desktop/Project/DexPose/retarget/0906/seq_taco_position_{hand}_hand.p"
    
    print(f"Loading data from {pickle_path}")
    with open(pickle_path, "rb") as f:
        load_data = pickle.load(f)

    # 运行多环境可视化
    run_visualization_grid(load_data[:16])