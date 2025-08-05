# DexPose

## Environment Setup

```
conda create -n anytelop python=3.9
<!-- pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 -->

pip install torch==2.0.1+cu118 \
            torchvision==0.15.2+cu118 \
            torchaudio==2.0.2+cu118 \
            --index-url https://download.pytorch.org/whl/cu118

pip install open3d pytorch-kinematics

cd thirdparty/dex-retargeting
pip install -e .
cd thirdparty/pytorch3d
pip install -e .
cd thirdparty/manopth
pip install -e .
cd thirdparty/manotorch
pip install -e .

# be aware of this, sometimes problematic
<!-- pip install torch_cluster --no-index -f https://data.pyg.org/whl/torch-1.13.0%2Bcu117.html -->
pip install torch_cluster --no-index -f https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html

pip install chumpy
```

##


Debug only need to inspect `python -m retarget_old.render_hand_object`
- input: dexycb_format
- output: html
- directly use dex_retargeting code
- visualize using `vis_frames_plotly`


DexYCB format:
- hand_pose: T x 0 x 51; 51=16*3+3; `[:,:3]` = wrist rot, `[:,-3:]` = wrist tsl. in camera frame.
- object_pose: T x 7; 7=4+3; 4(quat)=xyzw, 3(tsl)=xyz. in camera frame.
- extrinsics: 4 x 4. camera_extrincis


dex_retargeting logic:
- warm_start ---> retarget
- warm_start input: joints(in world frame), wrist_pose(directly from raw_data, in camera_frame)
    - first transform wrist_pose by **operator2mano**. (this is the same as yup2xup).
    - then transform by root2wrist (calculate by robot fk, but actually it always identity).
    - transform into euler angle (note: it uses pytransform3d, it is **different** from using pytorch3d.transforms)
    - then it save as last_qpos (serve as initialization)
- retarget input: joints(in world frame)
    - calculate link pose by fk of last_qpose.
    - calculate loss with joints (in world frame.)
    - output qpos


My assumption:
- after root2wrist, the reference frame comes from camera frame to world frame.
- however, the implementation here might be wrong, because this transformation.
- as a result, the current initialization is buggy, it works only because it is close enough to the initial correct pose.


visualize logic:
- visualize hand joints:
    - Option 1: (line 133-142 in render_hand_object) first `compute_hand_geometry`: input hand_pose (dexycb format, in camera frame), output hand joints (in camera frame). then transform into world frame.
    - Option 2: (line 872 in utils.vis_utils) `extract_hand_points_and_mesh`: input hand_tsl (world frame), hand coeffs (world frame), output hand joints (world frame).
        - Currently, it has two version: manotorch and manopth. They are the same now.(Verified)
    - Currently, option1 = option2 (fixed last time)
    - Option 1 use manopth (written by dex_retargeting), Option 2 use manotorch (written by us)
    - if use_pca, then the input 45dim should undergo an other transformation; if forward(p) + t and forward(p, tsl=t) differ at whether center_idx is given, if yes, then the former will translate to the center_idx's frame.
- visualize robot mesh:
    - (line 108-113 in render_hand_object) first load robot (by dex-suite), then seq_qpos (implement by pk.chain), the calculate mesh.
    - robot urdf **do not** direct come from dex-suite, 6 dummy joints are added (this is what dex_retarget do), meaning xyzrpy. qpos=3+3+joint_angle.

