from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RobotName,
    RetargetingType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import os
import shutil


### set robot directory
robot_dir = (
    Path("/home/qianxu/Desktop/Project/DexPose/thirdparty/dex-retargeting/assets").absolute() / "robots" / "hands"
)
RetargetingConfig.set_default_urdf_dir(robot_dir)


### copy Inspire hand URDF files
inspire_path = robot_dir / "inspire_hand"
for side in ["left", "right"]:
    src = inspire_path / f"inspire_hand_{side}.urdf"
    dst = inspire_path / f"inspire_hand_{side}_glb.urdf"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)


robot_names = ["shadow_hand", "inspire_hand", "allegro_hand", "leap_hand", "schunk_svh_hand"]

robot_name_map =  {
    "allegro_hand": RobotName.allegro,
    "shadow_hand": RobotName.shadow,
    "schunk_svh_hand": RobotName.svh,
    "leap_hand": RobotName.leap,
    "ability_hand": RobotName.ability,
    "inspire_hand": RobotName.inspire,
}

side_map = {
    "left": HandType.left,
    "right": HandType.right,
}

for robot_name in robot_names:
    for side in ["left", "right"]:
        config_path = get_default_config_path(
            robot_name_map[robot_name], RetargetingType.position, side_map[side]
        )

        # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
        override = dict(add_dummy_free_joint=True)
        config = RetargetingConfig.load_from_file(config_path, override=override)
        retargeting = config.build()
        robot_file_name = Path(config.urdf_path).stem

        # Build robot
        urdf_path = Path(config.urdf_path)
        if "glb" not in urdf_path.stem:
            urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
        robot_urdf = urdf.URDF.load(
            str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
        )
        save_path = urdf_path.parent / f"new_{side}_glb.urdf"
        if not save_path.exists():
            print(f"Saved: {save_path}")
            robot_urdf.write_xml_file(save_path)
        else:
            print(f"File already exists: {save_path}, skipping save.")

### rename Schunk hand to Schunk SVH hand
old_path = robot_dir / "schunk_hand"
new_path = robot_dir / "schunk_svh_hand"
if old_path.exists() and not new_path.exists():
    shutil.copytree(old_path, new_path)