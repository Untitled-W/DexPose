# DexPose

## Environment Setup

```bash
conda create -n dexpose_feature python==3.10
pip install -r feature_requirements.txt # open_clip_torch will install some version of torch, but don't care
pip install xformers # automatically install torch 2.8.0, but without gpu
pip install open3d pytorch-kinematics==0.7.5 chumpy matplotlib tqdm scikit-learn opencv-python hdbscan manifold3d
cd thirdparty/manotorch; pip install .; cd -

# This is the torch we will use, and packages below needs compilation
pip install ../packages/torch-2.8.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl
pip install ../packages/torch_cluster-1.6.3+pt28cu126-cp310-cp310-linux_x86_64.whl
cd thirdparty/pytorch3d; rm -rf build/ **\*.so; pip install .; cd -

```

## Usage

### (0) utils

`utils.vis_utils`
1. `visualize_dex_hand_sequence`(utils.vis_utils)
    1. args: dex_sequence, filename
    2. use `vis_frames_plotly`(utils.vis_utils)
2. `vis_frames_plotly`(utils.wmq)
    1. args: ...
    2. use `vis_pc_coor_plotly`(utils.wmq)
3. `vis_dexhand_optimize_stage2`(utils.wmq)
    1. args:...
    2. use `_vis_dexhand_optimize_frame`
4. `visualize_human_sequence` (to do)
5. `vis_as_frame`(utils.vis_utils)
    1. args: seq_data_ls, filename, check_frame_ls, if_render

`utils.tools`
1. `extract_hand_points_and_mesh`
    1. args: tsl, coeffs, side


### (1) dataset

`dataset.preprocessor`
1. set `dataset_names` (and DATASET_CONFIGS) ; save dexhand sequence (in `BaseDatasetProcessor.process_all_sequences`)
2. visualization options:
    1. data before/after mirror +/using hand_coeffs (in `process_sequence`)
        1.using `vis_frames_plotly`(utils.wmq)
    2. render 60th frame per sequence (in ``)
        1. using `vis_as_frame`(utils.vis_utils)


### (2) retarget

`retarget.optim_aughoi`
1. set `file_path` (the file: a list, each element a dict); save dexhand sequence (in `main_retarget`)
2. visualization options:
    1. 3D optimization process at one frame (in `retarget_sequence`)
        1. using: `vis_frames_plotly`(utils.wmq) and `vis_dexhand_optimize_stage2`(utils.wmq)
    2. loss-curve on optimization step at some frame (in `retarget_sequence`)
    3. loss-curve on time-step for the whole sequence (in `retarget_sequence`)
    4. 3D results for the whole sequence (in `main_retarget`)
        1. using: `visualize_dex_hand_sequence`(utils.vis_utils)

`retarget.anyteleop`
1. 

