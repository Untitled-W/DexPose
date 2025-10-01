import os
import pickle
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
import torch
from pytorch3d.transforms import matrix_to_quaternion
import time
from utils.hand_model import HandModelURDF

def setup_isaacgym_environment():
    # Create a gym instance
    gym = gymapi.acquire_gym()

    # Create a simulation
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity.x = 0
    sim_params.gravity.y = 0
    sim_params.gravity.z = -9.81

    # Create the simulation
    sim = gym.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_params)

    if sim is None:
        print("Failed to create sim")
        return
    
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # Set up the viewer
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1440
    cam_props.height = 960
    cam_props.horizontal_fov = 90.0
    cam_pos = gymapi.Vec3(0.5, -0.0, 1.2)
    cam_target = gymapi.Vec3(-0.5, -0.0, 0.2)
    viewer = gym.create_viewer(sim, cam_props)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    if viewer is None:
        print("Failed to create viewer")
        return

    return gym, sim, viewer

def parse_object_path(dex_seq_data: dict):
    dataset_name = dex_seq_data["which_dataset"]
    
    asset_root = []
    asset_file = []
    if dataset_name == "TACO":
        for ii in dex_seq_data["object_mesh_path"]:
            asset_root.append('/home/qianxu/Desktop/Project/DexPose/data/Taco/object_models')
            name = ii.split('/')[-1]
            asset_file.append(name[:-3]+'urdf')
    elif dataset_name == "Oakinkv2":
        for ii in dex_seq_data["object_mesh_path"]:
            asset_root.append(os.path.dirname(ii))
            name = ii.split('/')[-1]
            asset_file.append(name[:-3]+'urdf')
    elif dataset_name == "DexYCB":
        for ii in dex_seq_data["object_mesh_path"]:
            asset_root.append(os.path.dirname(ii))
            name = ii.split('/')[-1]
            asset_file.append(f'{name}.xml')

    return asset_root, asset_file

def create_env(dex_seq_data: dict):
    spacing = 0.5
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    hand_asset_root = os.path.join("/home/wangminqi/workspace/test/packages/hands", dex_seq_data['which_hand'])
    side = 'left' if dex_seq_data['which_hand'] == 0 else 'right'
    hand_asset_file = os.path.join(f"new_{side}_glb.urdf")

    object_asset_roots, object_asset_files = parse_object_path(dex_seq_data)

    # Initialize the Isaac Gym environment
    gym, sim, viewer = setup_isaacgym_environment()

    # Load the robot
    hand_asset_options = gymapi.AssetOptions()
    hand_asset_options.flip_visual_attachments = False
    hand_asset_options.fix_base_link = True
    hand_asset_options.collapse_fixed_joints = True
    hand_asset_options.disable_gravity = True
    hand_asset_options.thickness = 0.001
    hand_asset_options.angular_damping = 0.01
    hand_asset_options.use_physx_armature = True
    hand_handle = gym.load_asset(sim, hand_asset_root, hand_asset_file, hand_asset_options)

    hand_pose = gymapi.Transform()
    hand_pose.p = gymapi.Vec3(0, 0, 0)
    hand_pose.r = gymapi.Quat(0, 0, 0, 1)

    print("self.num_shadow_hand_bodies: ", gym.get_asset_rigid_body_count(hand_handle))
    print("self.num_shadow_hand_dofs: ", gym.get_asset_dof_count(hand_handle))
    print("self.num_shadow_hand_actuators: ", gym.get_asset_actuator_count(hand_handle))

    # Load the objects
    object_handles = []
    object_asset_options = gymapi.AssetOptions()
    object_asset_options.density = 1
    object_asset_options.flip_visual_attachments = False
    object_asset_options.fix_base_link = False    
    object_asset_options.collapse_fixed_joints = True
    object_asset_options.disable_gravity = False
    object_asset_options.vhacd_enabled = True
    object_asset_options.vhacd_params = gymapi.VhacdParams()
    object_asset_options.vhacd_params.resolution = 1000000
    object_asset_options.vhacd_params.concavity = 0.002  # Adjust concavity for better collision mesh
    object_asset_options.vhacd_params.max_num_vertices_per_ch = 64  # Limit vertices for simpler collision mesh
    object_asset_options.vhacd_params.max_convex_hulls = 10  # Limit convex hulls for performance
    object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    object_asset_options.override_com = True
    object_asset_options.override_inertia = True
    object_asset_options.vhacd_enabled = True
    object_asset_options.vhacd_params = gymapi.VhacdParams()
    object_asset_options.vhacd_params.resolution = 1000000
    object_asset_options.use_mesh_materials = False
    for root, file in zip(object_asset_roots, object_asset_files):
        object_handle = gym.load_asset(sim, root, file, object_asset_options)
        object_handles.append(object_handle)

    # Create the environment
    env = gym.create_env(sim, lower, upper, 1)
    
    # Add the robot to the environment
    hand_actor = gym.create_actor(env, hand_handle, hand_pose, "hand", 0, 0)
    shadow_hand_properties = gym.get_actor_rigid_shape_properties(env, hand_actor)
    for sh_prop in shadow_hand_properties:
        sh_prop.friction = 20
    gym.set_actor_rigid_shape_properties(env, hand_actor, shadow_hand_properties)

    # print("Hand DOF names:")
    # for i in range(gym.get_asset_dof_count(hand_handle)):
    #     dof_name = gym.get_asset_dof_name(hand_handle, i)
    #     print(f"DOF {i}: {dof_name}")

    # set shadow_hand dof properties
    shadow_hand_dof_props = gym.get_asset_dof_properties(hand_handle)

    shadow_hand_dof_lower_limits = []
    shadow_hand_dof_upper_limits = []
    shadow_hand_dof_default_pos = []
    shadow_hand_dof_default_vel = []

    for i in range(shadow_hand_dof_props['lower'].shape[0]):
        shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
        shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
        shadow_hand_dof_default_pos.append(0.0)
        shadow_hand_dof_default_vel.append(0.0)

        shadow_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
        shadow_hand_dof_props['stiffness'][i] = 400
        shadow_hand_dof_props['effort'][i] = 400
        shadow_hand_dof_props['damping'][i] = 20
        shadow_hand_dof_props['velocity'][i] = 1

    # print("shadow_hand_dof_lower_limits: ", shadow_hand_dof_lower_limits)
    # print("shadow_hand_dof_upper_limits: ", shadow_hand_dof_upper_limits)

    # Add objects to the environment
    object_actors = []
    for obj_handle in object_handles:
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0, 0, 0)
        obj_pose.r = gymapi.Quat(0, 0, 0, 1)
        object_actor = gym.create_actor(env, obj_handle, obj_pose, "object", 0, 0)
        object_properties = gym.get_actor_rigid_shape_properties(env, object_actor)
        for obj_prop in object_properties:
            obj_prop.friction = 20
        gym.set_actor_rigid_shape_properties(env, object_actor, object_properties)
        object_body_properties = gym.get_actor_rigid_body_properties(env, object_actor)
        for obj_body_prop in object_body_properties:
            obj_body_prop.mass = 0.1
            obj_body_prop.inertia.x.x = 0.01
            obj_body_prop.inertia.y.y = 0.01
            obj_body_prop.inertia.z.z = 0.01
        gym.set_actor_rigid_body_properties(env, object_actor, object_body_properties)
        object_actors.append(object_actor)


    return gym, sim, viewer, env, hand_actor, hand_handle, object_actors

if __name__ == "__main__":
    
    # Example dex_seq_data dictionary
    with open("/home/qianxu/Desktop/Project/DexPose/data/Taco/dex_save/seq_inspire_hand_debug_1.p", "rb") as f:
        load_data = pickle.load(f)

    dex_seq_data = load_data[0]

    print(dex_seq_data['which_hand'], dex_seq_data['which_dataset'], dex_seq_data['which_sequence'], dex_seq_data['side'])

    gym, sim, viewer, env, hand_actor, hand_handle, object_actors = create_env(dex_seq_data)

    pk_hand_qpos = dex_seq_data['hand_poses'] # T X n_dof
    obj_poses = dex_seq_data['obj_poses'] # K X T X 4 X 4
    K, T = obj_poses.shape[:2]

    side = 'left' if dex_seq_data['side'] == 0 else 'right'
    hand_asset_root = os.path.join("/home/wangminqi/workspace/test/packages/hands", dex_seq_data['which_hand'])
    robot = HandModelURDF(dex_seq_data['which_hand'],os.path.join(hand_asset_root, f"new_{side}_glb.urdf"),os.path.join(hand_asset_root, f'meshes'))

    ### Create a mapping from PK to Isaac
    isaac_name_to_idx = {}
    for i in range(gym.get_asset_dof_count(hand_handle)):
        dof_name = gym.get_asset_dof_name(hand_handle, i)
        isaac_name_to_idx[dof_name] = i
    print("Isaac DOF names:", isaac_name_to_idx)

    pk_name_to_idx = {}
    for idx, name in enumerate(robot.joints_names):
        pk_name_to_idx[name] = idx
    # print("PK DOF names:", pk_name_to_idx)

    # remap pk_hand_qpos to isaacgym
    hand_qpos = np.zeros((T, gym.get_actor_dof_count(env, hand_actor)), dtype=np.float32)
    for pk_name, pk_idx in pk_name_to_idx.items():
        isaac_idx = isaac_name_to_idx[pk_name]
        hand_qpos[:, isaac_idx] = pk_hand_qpos[:, pk_idx]

    # ### Make hand tighter
    # actions_extend = np.copy(hand_qpos)
    # positive_number = 0.0
    # ## FFJ 4 3 2 1
    # actions_extend[:, 8:12] += np.array([0, positive_number, positive_number, positive_number]).astype(np.float32)
    # ### LFJ 5 4 3 2 1
    # actions_extend[:, 20:25] += np.array([positive_number, 0, positive_number, positive_number, positive_number]).astype(np.float32)
    # ### MFJ 4 3 2 1
    # actions_extend[:, 12:16] += np.array([0, positive_number, positive_number, positive_number]).astype(np.float32)
    # ### RFJ 4 3 2 1
    # actions_extend[:, 16:20] += np.array([0, positive_number, positive_number, positive_number]).astype(np.float32)
    # ### THJ 5 4 3 2 1
    # actions_extend[:, 25:30] += np.array([positive_number, positive_number, positive_number, positive_number, positive_number]).astype(np.float32)

    # hand_qpos = np.copy(actions_extend)

    SIM = False

    _root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_tensor = gymtorch.wrap_tensor(_root_tensor)  

    n_dof = hand_qpos.shape[-1]
    dof_state_template = np.zeros(n_dof, dtype=gymapi.DofState.dtype)

    start_idx = 50
    end_idx = 100
    t = start_idx

    while True:
        # for t in range(start_idx, T):

        if t == start_idx or not SIM: 
            ts = obj_poses[:, t, :3, 3].float()          # [K, 3]
            Rs = obj_poses[:, t, :3, :3].float()         # [K, 3, 3]
            quats = torch.stack([matrix_to_quaternion(R) for R in Rs])     # [K, 4]  (w, x,y,z)
            quats = quats[:, [1, 2, 3, 0]]  # Reorder to (x, y, z, w)

            gym.refresh_actor_root_state_tensor(sim)
            root_tensor_cpu = root_tensor.clone()

            for k in range(K):
                actor_idx = 1 + k
                root_tensor_cpu[actor_idx, 0:3] = ts[k] # + torch.tensor([0.02, -0.01, 0.03], device=ts.device)  # Offset the object position slightly above the ground
                root_tensor_cpu[actor_idx, 3:7] = quats[k]
                root_tensor_cpu[actor_idx, 7:13] = 0 

            root_tensor[:] = root_tensor_cpu
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))

        # Set the hand qpos
        dof_state_template['pos'] = hand_qpos[t]
        gym.set_actor_dof_states(env, hand_actor, dof_state_template, gymapi.STATE_POS)

        if gym.query_viewer_has_closed(viewer):
            break
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        time.sleep(0.05)  # sleep for 50ms, adjust as needed

        t += 1
        if t == end_idx: t = start_idx

    # Cleanup
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


def generate_hand_trajectory(gym, env, hand_actor):
    # Get DOF properties and limits
    dof_props = gym.get_actor_dof_properties(env, hand_actor)
    dof_count = gym.get_actor_dof_count(env, hand_actor)
    lower_limits = dof_props['lower']
    upper_limits = dof_props['upper']

    # Only consider joints from the seventh onward
    start_joint = 6
    trajectory = []

    # For each joint from the seventh onward, move it through its range
    for joint_idx in range(start_joint, dof_count):
        joint_traj = []
        # Generate 10 steps between lower and upper limits
        for alpha in np.linspace(0, 1, 10):
            pos = np.zeros(dof_count)
            # Set all joints to default (midpoint)
            pos[:] = (lower_limits + upper_limits) / 2
            # Move only the current joint
            pos[joint_idx] = lower_limits[joint_idx] * (1 - alpha) + upper_limits[joint_idx] * alpha
            joint_traj.append(pos.copy())
        trajectory.append(joint_traj)
    return trajectory


# with open("/home/qianxu/Desktop/Project/DexPose/data/Taco/dex_save/seq_shadow_hand_debug_1.p", "rb") as f:
#     load_data = pickle.load(f)

# dex_seq_data = load_data[0]
# gym, sim, viewer, env, hand_actor, object_actors = create_env(dex_seq_data)

# traj = generate_hand_trajectory(gym, env, hand_actor)
# while True:
#     for joint_traj in traj:
#         for qpos in joint_traj:
#             dof_state = np.zeros(qpos.shape[0], dtype=[('pos', np.float32), ('vel', np.float32)])
#             dof_state['pos'] = qpos
#             dof_state['vel'] = 0.0
#             gym.set_actor_dof_states(env, hand_actor, dof_state, gymapi.STATE_POS)
#             gym.simulate(sim)
#             gym.fetch_results(sim, True)
#             gym.step_graphics(sim)
#             gym.draw_viewer(viewer, sim, True)
#             gym.sync_frame_time(sim)
#             time.sleep(0.05)