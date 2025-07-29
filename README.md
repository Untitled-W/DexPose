# DexPose

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

