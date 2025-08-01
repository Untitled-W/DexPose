# DexPose

## Environment Setup

```
conda create -n anytelop python=3.9
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
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
pip install torch_cluster --no-index -f https://data.pyg.org/whl/torch-1.13.0%2Bcu117.html

pip install chumpy
```

## Repo Structure

```
dataset/
├── __init__.py
├── base_structure.py
├── preprocessor.py
└── vis_results
```

`python -m dataset.preprocessor` 

1. process 'oakink','taco','dexycb' into HumanSequenceData format; (load path at `base_structure.py`)
2. save them into 'xxx.p'; (save path at `base_structure.py`)
3. visualize them.
4. (optional) show statistics.


```
retarget/
├── __init__.py
├── hand_robot_viewer.py
├── hand_viewer.py
├── render_hand_object.py
├── main.py
├── hand_qpos
├── urdf
└── vis_results
```

`python -m retarget.render_hand_object`

1. load dexycb data (using `utils.dexycbdataset.py`)
2. retarget to 6 type of robot hands (using `retarget.xxx_viwer.py`) and save qpos.
3. visualize robot hand from qpos.


`python -m retarget.main`
(under develop)

1. load humanSequenceData
2. retarget to 6 type of robot hands (using `utils.veiwer.py`) and save to DexSequenceData.


diffusers==0.15.0
transformers==4.29.2
xformers==0.0.20
ipympl==0.9.3
accelerate==0.28.0
urllib3==1.25.11
open_clip_torch==2.24.0
einops==0.7.0
openai

