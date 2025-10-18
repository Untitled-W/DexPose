import argparse
import csv
import json
import torch
import numpy as np
import joblib
import os
import itertools
from datetime import datetime
from matplotlib import pyplot as plt

from .robot_wrapper import load_robot, HandRobotWrapper
# from utils.hand_model import load_robot

from tqdm import tqdm

print_once = (lambda: (lambda num=[0]: lambda *a, **k: (print(*a, **k), num.__setitem__(0, 1)) if num[0] == 0 else None)())

def quat_to_aa_wmq_old(quat, side='right'):
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


def retarget_sequence(seq_data, robot_hand: HandRobotWrapper, params: dict = None,
                      do_vis: bool = False, save_dir: str = None):

    device = robot_hand.device

    side = seq_data.get('side', 'right')

    # Init robot hand transformation
    init_hand_tsl, init_hand_quat = seq_data['h_joints'][:, 0], seq_data['h_coeffs'][:, 0]
    dex_pose = torch.zeros((seq_data['h_joints'].shape[0], robot_hand.n_dofs), device=device)
    init_hand_aa = quat_to_aa_wmq(init_hand_quat)
    dex_pose[:, 3:6] = init_hand_aa.clone()
    robot_hand.compute_forward_kinematics(dex_pose)
    dex_pose[:, :3] = (init_hand_tsl - robot_hand.get_wrist()).clone()
    dex_pose.requires_grad_(True)

    # Init robot hand pose (all joints align)
    p = params or {}
    total_step = int(p.get('stage1_total_step', 200))
    lr = float(p.get('stage1_lr', 0.01))
    sp_coeffs = float(p.get('stage1_sp_coeffs', 0.5))
    thres = float(p.get('stage1_thres', 0.02))
    logger_1 = {
        "is_plot_optimize": bool(p.get('stage1_is_plot_optimize', False)),
        "is_plot_timestep": bool(p.get('stage1_is_plot_timestep', True)),
        "is_vis": bool(p.get('stage1_is_vis', False)),
        "optimize_frame_check_ls": p.get('stage1_optimize_frame_check_ls', [30, 60, 90]),
        "vis_frame": int(p.get('stage1_vis_frame', 90)),
        "vis_interval": int(p.get('stage1_vis_interval', 10)),
        'plot_interval': int(p.get('stage1_plot_interval', 20)),
        "history": {"hand_mesh":[], "robot_keypoints":[]},
        "losses": {"E_align":[], f"E_spen x{sp_coeffs}":[]}    
    }
    init_optimizer = torch.optim.Adam([dex_pose], lr=lr, weight_decay=0)
    mano_fingertip = seq_data['h_joints'].to(device)[:, robot_hand.human_keypoints_order, :]
    for ii in tqdm(range(total_step), desc="Initial alignment"):
        robot_hand.compute_forward_kinematics(dex_pose)
        fingertip_keypoints = robot_hand.get_all_joints_in_mano_order()
        E_align = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip, reduction='none').mean((-2,-1))
        E_spen = robot_hand.self_penetration(thres)
        loss = E_align.sum() + E_spen.mean() * sp_coeffs
        init_optimizer.zero_grad()
        loss.backward()
        init_optimizer.step()
        if logger_1['is_vis'] and ii % logger_1['vis_interval'] == 0:
            logger_1['history']['hand_mesh'].append(robot_hand.get_trimesh_data_single(logger_1['vis_frame']))
            logger_1['history']['robot_keypoints'].append(fingertip_keypoints[logger_1['vis_frame']].detach().cpu().numpy())
        logger_1['losses']['E_align'].append(E_align.detach().numpy().tolist())
        logger_1['losses'][f"E_spen x{sp_coeffs}"].append(E_spen.detach().numpy().tolist())

    folder_name = save_dir
    filename = f"uid{seq_data['uid']}_{seq_data['which_sequence']}"
    if logger_1['is_vis']:
        from utils.wmq import vis_frames_plotly
        TT = len(logger_1["history"]["hand_mesh"])
        vis_frames_plotly(
            gt_hand_joints=seq_data['h_joints'][logger_1['vis_frame']].expand(TT, -1, -1).cpu().numpy(),
            gt_posi_pts=mano_fingertip[logger_1['vis_frame']].expand(TT, -1, -1).cpu().numpy(),
            posi_pts_ls=[np.stack(logger_1['history']['robot_keypoints'])],
            hand_mesh_ls=[logger_1["history"]["hand_mesh"]],
            show_line=True,
            filename=os.path.join(folder_name, "visualization", f"vis_frame{logger_1['vis_frame']}", filename)
        )
    if logger_1['is_plot_optimize']:
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(1, len(logger_1["optimize_frame_check_ls"]), figsize=(6 * len(logger_1["optimize_frame_check_ls"]), 5))
        if len(logger_1["optimize_frame_check_ls"]) == 1: axs = [axs]
        for idx, ff_n in enumerate(logger_1["optimize_frame_check_ls"]):
            ax = axs[idx]
            for lr_key in logger_1['losses']:
                loss_item = np.stack([ii[ff_n] for ii in logger_1['losses'][lr_key]])
                ax.plot(np.arange(len(loss_item)), loss_item, label=lr_key)
            ax.set_yscale('log')
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.set_title(f"Frame {ff_n}")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, "loss", "optim_stage1", filename+'.png'))
        plt.close()
    if logger_1["is_plot_timestep"]:
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
        plt.savefig(os.path.join(folder_name, "loss", "seq_stage1", f"{filename}.png"))
        plt.close()

    # Stage 2
    total_step = int(p.get('stage2_total_step', 200))
    lr = float(p.get('stage2_lr', 0.01))
    sp_coeffs = float(p.get('stage2_sp_coeffs', 5))
    thres = float(p.get('stage2_thres', 0.02))
    dis_coeffs = float(p.get('stage2_dis_coeffs', 1))
    dis_thres = float(p.get('stage2_dis_thres', 0.01))
    pen_coeffs = float(p.get('stage2_pen_coeffs', 0.005))
    pen_thres = float(p.get('stage2_pen_thres', 0.005))
    logger_2 = {
        "is_plot_optimize": bool(p.get('stage2_is_plot_optimize', False)),
        "is_plot_timestep": bool(p.get('stage2_is_plot_timestep', True)),
        "is_vis": bool(p.get('stage2_is_vis', do_vis)),
        "optimize_frame_check_ls": p.get('stage2_optimize_frame_check_ls', [30, 60, 90]),
        "vis_frame": int(p.get('stage2_vis_frame', 90)),
        "vis_interval": int(p.get('stage2_vis_interval', 10)),
        "plot_interval": int(p.get('stage2_plot_interval', 20)),
        "history": {"hand_mesh":[], "robot_keypoints":[], "ct_pts":[], "corr_ct_pts":[],"spen":[], "inner_pts":[], "outer_pts":[]},
        "losses": {"E_align":[], f"E_spen x{sp_coeffs}":[], f"E_dist x{dis_coeffs}":[], f"E_pen x{pen_coeffs}":[]}    
    }
    check_frame = logger_2["vis_frame"]
    next_optimizer = torch.optim.Adam([dex_pose], lr=lr, weight_decay=0)
    mano_fingertip = seq_data['h_joints'].to(device)[:, robot_hand.human_keypoints_order, :]
    from utils.tools import get_point_clouds_from_human_data, apply_transformation_human_data, get_object_meshes_from_human_data, apply_transformation_on_object_mesh
    pc, pc_norm = get_point_clouds_from_human_data(seq_data, return_norm=True, ds_num=1000)
    pc_ls, pc_norm_ls = apply_transformation_human_data(pc, seq_data['o_transf'], norm=pc_norm)

    for ii in tqdm(range(total_step), desc="Stage 2 optimization"):
        robot_hand.compute_forward_kinematics(dex_pose)
        fingertip_keypoints = robot_hand.get_all_joints_in_mano_order()
        E_align = torch.nn.functional.huber_loss(fingertip_keypoints, mano_fingertip, reduction='none').mean((-2,-1))
        E_dis, ct_pts, cc_ct_pts = robot_hand.cal_distance(pc_ls, pc_norm_ls, dis_thres, True)
        E_pen, inner_pts, outer_pts = robot_hand.cal_object_penetration(pc_ls, pen_thres, True)
        E_spen = robot_hand.self_penetration(thres)
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

    folder_name = save_dir
    filename = f"uid{seq_data['uid']}_{seq_data['which_sequence']}"
    if logger_2["is_vis"]:
        from utils.wmq import vis_dexhand_optimize_stage2
        obj_mesh = get_object_meshes_from_human_data(seq_data)
        obj_mesh_ls = apply_transformation_on_object_mesh(obj_mesh, seq_data['o_transf'][:, check_frame:check_frame+1, :, :])
        TT = len(logger_2["history"]["hand_mesh"])
        vis_dexhand_optimize_stage2(
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
            filename=os.path.join(folder_name, "visualization", f"vis_frame{check_frame}", filename)
        )
    if logger_2['is_plot_optimize']:
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(6 * len(logger_2["optimize_frame_check_ls"]), 5 * 2))
        gs = fig.add_gridspec(2, len(logger_2["optimize_frame_check_ls"]), hspace=0.3)

        # 上排：loss 曲线
        for col, ff_n in enumerate(logger_2["optimize_frame_check_ls"]):
            ax = fig.add_subplot(gs[0, col])
            for lr_key in logger_2['losses']:
                loss_item = np.stack([ii[ff_n] for ii in logger_2['losses'][lr_key]])
                ax.plot(np.arange(len(loss_item)), loss_item, label=lr_key)
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

        plt.savefig(os.path.join(folder_name, "loss", 'optim_stage2', filename+'.png'))
        plt.close()
    if logger_2["is_plot_timestep"]:
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
        plt.savefig(os.path.join(folder_name, 'loss', 'seq_stage2', f"{filename}.png"))
        plt.close()

    retargeted_seq = dict(
        which_hand=robot_hand.robot_name,
        dex_poses=dex_pose,
    )
    retargeted_seq.update(seq_data)

    return retargeted_seq, logger_1['losses'], logger_2['losses']

def main_retarget(seq_data_ls, robot_name, run_root, params: dict = None, vis: bool = False):
    
    if vis:
        vis_dir = os.path.join(run_root, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
    loss_dir = os.path.join(run_root, 'loss')
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(os.path.join(loss_dir, 'seq_stage1'), exist_ok=True)
    os.makedirs(os.path.join(loss_dir, 'seq_stage2'), exist_ok=True)

    robot_hand = load_robot(robot_name)
    print(f"Retargeting to {robot_hand.robot_name} ...")

    # save params to config.json at run root
    try:
        with open(os.path.join(run_root, 'config.json'), 'w') as cf:
            json.dump(params or {}, cf, indent=2)
    except Exception:
        pass

    retargeted_data_all = []
    losses_stage1_all = []
    losses_stage2_all = []

    save_path = os.path.join(run_root, f"{robot_name}_retargeted.p")
    for i, seq_data in enumerate(seq_data_ls):
        print(f"Sequence {i}: Processing sequence {seq_data['uid']} with {robot_name} from {seq_data.get('which_dataset','?')} with {seq_data.get('which_sequence','?')}")

        retargeted_seq, losses_1, losses_2 = retarget_sequence(seq_data, robot_hand, params=params,save_dir=run_root)
        retargeted_data_all.append(retargeted_seq)
        losses_stage1_all.append(losses_1)
        losses_stage2_all.append(losses_2)

        with open(save_path, 'wb') as ofs:
            joblib.dump(retargeted_data_all, ofs)
            from utils.vis_utils import visualize_dex_hand_sequence
        
        if vis:
            visualize_dex_hand_sequence(retargeted_seq, os.path.join(run_root, "visualization", f"{retargeted_seq['uid']}_{retargeted_seq['which_sequence']}"))

    # Save all loss time series to loss.json (single file)
    aggregated = {
        'stage1': losses_stage1_all,
        'stage2': losses_stage2_all
    }
    try:
        with open(os.path.join(run_root, "loss", 'loss.json'), 'w') as lf:
            json.dump(aggregated, lf)
    except Exception:
        pass

    return {
        'run_root': run_root,
        'save_p': save_path
    }


def generate_seq(seq_data_ls, robots, base_save_dir, params: dict = None, vis: bool = False):

    for robot_name in robots:
        robot_folder = os.path.join(base_save_dir, 'save_results', robot_name)
        os.makedirs(robot_folder, exist_ok=True)
        time_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
        root = os.path.join(robot_folder, time_folder)
        os.makedirs(root, exist_ok=True)
        main_retarget(seq_data_ls, robot_name, base_save_dir=root, params=params, vis=False)


def find_params(seq_data_ls, robot_name, base_save_dir, param_grid: dict):
    """
    param_grid: dict of param_name -> list of values
    For each combination, run main_retarget-like workflow and save results under
    base_save_dir/save_results/<robot_name>/<datetime>/<param_combination>/
    Save visualization for each sequence result under a visualization folder (per-combo).
    Compute evaluation metric as described and save param_search_results.csv
    """
    robot_folder = os.path.join(base_save_dir, 'save_results', robot_name)
    os.makedirs(robot_folder, exist_ok=True)
    time_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
    root = os.path.join(robot_folder, time_folder)
    os.makedirs(root, exist_ok=True)

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    results = []
    for combo in combos:
        combo_dict = dict(zip(keys, combo))
        combo_name = '_'.join([f"{k}{v}" for k,v in combo_dict.items()])
        combo_dir = os.path.join(root, combo_name)
        os.makedirs(combo_dir, exist_ok=True)

        # run retarget for all sequences with this combo
        main_retarget(seq_data_ls, robot_name, run_root=combo_dir, params=combo_dict, vis=True)
        
        # compute evaluation metric
        eval_metric = None
        loss_json_path = os.path.join(combo_dir, "loss", "loss.json")
        with open(loss_json_path, 'r') as lf:
            loss_data = json.load(lf)
        # loss_data expects list per sequence per loss
        # compute metric as described: find contact frames where E_dist != 0 (we assume E_dist entries exist)
        # For each sequence, for each timestep where E_dist != 0, compute mean of all loss terms, then average across timesteps, then sum across sequences
        seq_metrics = []
        for seq_losses in loss_data.get('stage2', []):
            # seq_losses is dict of loss_name -> list of time series per seq? original structure ambiguous; attempt heuristic
            # If seq_losses is dict with loss_name -> list(T)
            contact_mask = None
            loss_means = []
            # attempt to build contact mask from E_dist key
            for k, v in seq_losses.items():
                if 'E_dist' in k:
                    # v expected to be list of length T
                    arr = np.array(v)
                    contact_mask = arr != 0
                    break
            if contact_mask is None:
                # no contact info -> skip
                continue
            # for all loss keys, compute per-timestep mean across losses
            loss_names = list(seq_losses.keys())
            per_timestep_vals = []
            for t_idx in range(len(contact_mask)):
                if not contact_mask[t_idx]:
                    continue
                vals = []
                for lk in loss_names:
                    try:
                        vals.append(float(np.array(seq_losses[lk])[t_idx]))
                    except Exception:
                        pass
                if len(vals) > 0:
                    per_timestep_vals.append(np.mean(vals))
            if len(per_timestep_vals) > 0:
                seq_metrics.append(np.mean(per_timestep_vals))
        if len(seq_metrics) > 0:
            eval_metric = float(np.sum(seq_metrics))
        results.append((combo_name, combo_dict, eval_metric))

        # save combo metric to a small json
        try:
            with open(os.path.join(combo_dir, 'result_summary.json'), 'w') as rf:
                json.dump({'metric': eval_metric, 'params': combo_dict}, rf, indent=2)
        except Exception:
            pass

    # write csv
    csv_path = os.path.join(root, 'param_search_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['combo_name', 'params_json', 'metric'])
        for combo_name, combo_dict, metric in results:
            writer.writerow([combo_name, json.dumps(combo_dict), metric])

    return root

def get_piece_by_frame(
    seq_data, 
    frame_slice: slice = None
):
    
    if frame_slice is None:
        return seq_data

    sliced_data = seq_data.copy()
    # --- 对与时间维度相关的字段应用切片 ---
    sliced_data['seq_len'] = len(range(*frame_slice.indices(seq_data['seq_len'])))

    # 1. 手部数据 (时间维度在 dim=0)
    sliced_data['h_tsl'] = seq_data['h_tsl'][frame_slice]
    sliced_data['h_coeffs'] = seq_data['h_coeffs'][frame_slice]
    sliced_data['h_params'] = seq_data['h_params'][frame_slice]
    
    if "dex_pose" in seq_data:
        sliced_data['dex_pose'] = seq_data['dex_pose'][frame_slice]
    if "h_joints" in seq_data:
        sliced_data['h_joints'] = seq_data['h_joints'][frame_slice]

    # 2. 物体位姿数据 (时间维度在 dim=1)
    # 形状为 (K, T, 4, 4)，所以我们在第二个维度上切片
    sliced_data['o_transf'] = seq_data['o_transf'][:, frame_slice]

    return sliced_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['generate_seq', 'find_params'], default='generate_seq')
    parser.add_argument('--robots', nargs='+', default=['shadow_hand'])
    parser.add_argument('--file_path', type=str, help='input pickle file path with sequences')
    parser.add_argument('--save_base', type=str, default=os.path.dirname(__file__))
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--param_json', type=str, help='optional json file with hyperparameters')
    parser.add_argument('--param_grid_json', type=str, help='json file specifying grid for find_params')
    args = parser.parse_args()

    # load sequences
    import pickle, random
    seq_data_ls = []
    if args.file_path:
        with open(args.file_path, 'rb') as f:
            seq_data_ls = pickle.load(f)
    else:
        print('No file_path provided, exiting')
        exit(1)

    for i in range(len(seq_data_ls)):
        seq_data_ls[i]['object_mesh_path'] = [ii.replace('/home/wangminqi/workspace/test/', '/mnt/shared-storage-user/wangminqi/') for ii in seq_data_ls[i]['object_mesh_path']]

    seq_data_ls = [get_piece_by_frame(ii, frame_slice=slice(0,10)) for ii in random.sample(seq_data_ls, 1)]

    params = {}
    if args.param_json:
        with open(args.param_json, 'r') as pf:
            params = json.load(pf)

    os.makedirs(args.save_base, exist_ok=True)

    if args.mode == 'retarget':
        results = main_retarget(seq_data_ls, args.robots, args.save_base, params=params)
        print('Saved results to:', results)
    else:
        if not args.param_grid_json:
            # using default grid
            param_grid = {
                'stage1_total_step': [2],
                'stage2_total_step': [2],
                'stage2_sp_coeffs': [1],
                'stage2_pen_coeffs': [0.001],
                'stage2_pen_thres': [0.005],
                'stage2_dis_coeffs': [0.5],
                'stage2_dis_thres': [0.01, 0.02],
            }
        else:
            with open(args.param_grid_json, 'r') as pgf:
                param_grid = json.load(pgf)
        root = find_params(seq_data_ls, args.robots[0], args.save_base, param_grid)
        print('Param search results at', root)
