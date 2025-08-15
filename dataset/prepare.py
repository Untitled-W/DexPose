


### Taco

template = '''
<?xml version="1.0"?>
<robot name="design">
    <material name="obj_color">
            <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
    </material>
    <link name="base">
        <visual>
            <origin xyz="0.0 0.0 0.0"/>
            <geometry>
                <mesh filename="{}.obj" scale=".01 .01 .01"/>
            </geometry>
            <material name="obj_color"/>
        </visual>
        <collision>
            <origin xyz="0.0 0.0 0.0"/>
            <geometry>
                <mesh filename="{}.obj" scale=".01 .01 .01"/>
            </geometry>
        </collision>
    </link>
</robot>
'''

import os

obj_directory = '/home/qianxu/Desktop/Project/DexPose/data/Taco/object_models'
for obj_file in os.listdir(obj_directory):
    if obj_file.endswith('.obj'):
        urdf_content = template.format(obj_file[:-4], obj_file[:-4])
        urdf_filename = os.path.join(obj_directory, obj_file[:-4] + '.urdf')
        if not os.path.exists(urdf_filename):
            print(f"Creating: {urdf_filename}")
            with open(urdf_filename, 'w') as urdf_file:
                urdf_file.write(urdf_content)
        else:
            print(f"File already exists: {urdf_filename}, skipping creation.")


### Oakinkv2

template_oakinkv2 = '''

<?xml version="1.0"?>
<robot name="design">
  <material name="obj_color">
      <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <link name="base">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="model.obj" scale="1 1 1"/>
      </geometry>
      <material name="obj_color"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="model.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>

'''

import os
import shutil

align_ds_dir = '/home/qianxu/Desktop/Project/DexPose/data/Oakinkv2/coacd_object_preview/align_ds'

for folder in os.listdir(align_ds_dir):
    folder_path = os.path.join(align_ds_dir, folder)
    if os.path.isdir(folder_path):
        dst_file = os.path.join(folder_path, 'model.urdf')
        if not os.path.exists(dst_file):
            print(f"Creating: {dst_file}")
            with open(dst_file, 'w') as f:
                f.write(template_oakinkv2)
        else:
            print(f"File already exists: {dst_file}, skipping copy.")