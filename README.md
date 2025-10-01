# DexPose

## Environment Setup

```bash
conda create -n dexpose_feature python==3.10
pip install -r feature_requirements.txt # open_clip_torch will install some version of torch, but don't care
pip install xformers # automatically install torch 2.8.0, but without gpu
pip install open3d pytorch-kinematics==0.7.5 chumpy matplotlib tqdm scikit-learn opencv-python hdbscan manifold3d
cd thirdparty/dex-retargeting; pip install -e .; cd -
cd thirdparty/manotorch; pip install .; cd -

# This is the torch we will use, and packages below needs compilation
pip install ../packages/torch-2.8.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl
pip install ../packages/torch_cluster-1.6.3+pt28cu126-cp310-cp310-linux_x86_64.whl
cd thirdparty/pytorch3d; rm -rf build/ **\*.so; pip install .; cd -

```