import torch
import joblib
import os
import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OBJ_DIC = {
    "Shovel": ["026_cm", "033_cm", "034_cm", "046_cm", "053_cm", "060_cm", "062_cm", "090_cm", "091_cm", "128_cm", "129_cm", "130_cm", "131_cm", 
           "133_cm", "162_cm", "173_cm", "181_cm", "182_cm", "184_cm", "197_cm", "208_cm", "211_cm", "212_cm", '173cm'],
    "Soup Spoon": ["016_cm", "025_cm", "032_cm", "050_cm", "132_cm", "134_cm", "163_cm", "117_cm", "169_cm", "170_cm", "172_cm", 
           "175_cm", "183_cm", "198_cm", "200_cm", "203_cm", "204_cm", "206_cm", "207_cm", "209_cm", "210_cm", "217_cm", "218_cm"],
    "Knife": ["044_cm", "045_cm", "061_cm", "063_cm", "073_cm", "074_cm", "083_cm", "109_cm", "120_cm", "122_cm", 
          "153_cm", "202_cm", "205_cm"],
    "Cleaning Brush": ["001_cm", "004_cm", "006_cm", "035_cm", "036_cm", "041_cm", "042_cm", "068_cm", "069_cm", "071_cm", "072_cm", 
           "171_cm", "178_cm", "179_cm"],
    "Lint Roller": ["031_cm", "070_cm", "076_cm", "111_cm", "113_cm", "114_cm", "115_cm", "116_cm", "164_cm", "195_cm"],
    "Kettle": ["029_cm", "030_cm", "040_cm", "079_cm", "089_cm", "199_cm", "215_cm", "216_cm"],
    "Teapot": ["010_cm", "075_cm", "088_cm", "094_cm", "096_cm", "097_cm", "106_cm", "157_cm", "159_cm", "160_cm"],
    # "Cup": ["009_cm", "007_cm", "008_cm", "080_cm", "095_cm", "098_cm", "099_cm", "100_cm", "101_cm", "107_cm", "155_cm", 
    #         "167_cm", "168_cm", "187_cm"],
    "Bowl": ["022_cm", "023_cm", "024_cm", "118_cm", "146_cm", "174_cm", "180_cm", "189_cm", "190_cm", "191_cm", "193_cm", "194_cm"],
    # "Plate": ["005_cm", "019_cm", "020_cm", "021_cm", "049_cm", "052_cm", "125_cm", "135_cm", "158_cm", "161_cm", "165_cm", "166_cm"],
    # "Box": ["087_cm", "093_cm", "103_cm", "104_cm", "105_cm", "121_cm", "136_cm", "137_cm", "143_cm"],
    "Pot": ["015_cm", "027_cm", "028_cm", "057_cm", "084_cm", "085_cm", "192_cm", "201_cm", "213_cm", "214_cm"],
    # "Toy": ["047_cm", "081_cm", "082_cm"],
    # "Helmet": ["038_cm", "039_cm", "051_cm", "086_cm", "102_cm"],
    "Eraser": ["065_cm", "066_cm", "067_cm", "078_cm", "123_cm", "176_cm", "186_cm", "188_cm"],
    # "Soap": ["013_cm", "056_cm", "092_cm", "124_cm", "177_cm", "196_cm"],
    # "Gun": ["043_cm", "048_cm", "077_cm", "126_cm", "141_cm", "149_cm"],
    "Hammer": ["002_cm", "037_cm", "110_cm", "127_cm", "139_cm", "142_cm", "144_cm", "148_cm"],
    # "Ruler": ["054_cm", "055_cm", "064_cm", "138_cm", "140_cm", "145_cm", "185_cm"],
    # "Screwdriver": ["058_cm", "059_cm", "147_cm", "150_cm", "151_cm"]
}

VALID_LS = ['Shovel', 'Soup Spoon', 'Knife', 'Cleaning Brush', 'Lint Roller', 
            'Kettle', 'Teapot', 'Cup', 'Bowl', 'Plate', 'Box', 'Pot', 'Eraser', 
            'Hammer']

OBJ_TRAIN_DIC = {
    "Shovel": ["181_cm", "182_cm", '173_cm'],
    "Soup Spoon": ["032_cm", "200_cm", "198_cm"],
    "Knife": ["063_cm","120_cm", "202_cm"],
    "Cleaning Brush": ["006_cm", "178_cm", "179_cm"],
    "Lint Roller": ["164_cm", "195_cm", "116_cm"],
    "Kettle": ["030_cm", "040_cm", "029_cm"],
    "Teapot": ["094_cm", "096_cm", "097_cm"],
    "Bowl": ["022_cm", "023_cm", "024_cm"],
    "Pot": ["084_cm", "192_cm", "201_cm"],
    "Eraser": ["065_cm", "066_cm", "067_cm"],
    "Hammer": ["139_cm", "142_cm", "144_cm"],
    # "Gun": ["043_cm", "048_cm", "077_cm"],
    # "Screwdriver": ["058_cm", "147_cm", "059_cm"],
}

SELECT_LS = ['Shovel', 'Soup Spoon', 'Knife', 'Cleaning Brush', 'Lint Roller', 
            'Kettle', 'Teapot', 'Cup', 'Bowl', 'Plate', 'Box', 'Pot', 'Eraser', 
            'Hammer']

def find_key_for_value(obj_dict, value_to_find):
    """
    Search through obj_dict's values (lists) to find value_to_find.
    If found, return the corresponding key; otherwise, return None.
    """
    for key, values in obj_dict.items():
        if value_to_find in values:
            return key
    return None

def data_static(data_ls):
    length_dict = {}
    data_num_dict = {}
    for item in data_ls:
        name:str = os.path.splitext(os.path.split(item['mesh_path'])[-1])[0]
        name = find_key_for_value(OBJ_DIC, name)
        length:int = item['seq_len']
        if name in length_dict:
            length_dict[name].append(length)
        else: 
            length_dict[name] = [length]
        data_num = 1 if length <= 120 else (length -119) 
        if name in data_num_dict:
            data_num_dict[name] += data_num
        else:
            data_num_dict[name] = data_num

    box_data = []
    for name, lengths in length_dict.items():
        box_data.append(
            go.Box(
                y=lengths,  # distribution of seq_len
                name=name    # label on the x-axis
            )
        )

    fig_box = go.Figure(data=box_data)
    fig_box.update_layout(
        title="Distribution of seq_len per name (Box Plot)",
        xaxis_title="Name",
        yaxis_title="seq_len",
    )
    fig_box.show()

    # ----------------------------------------------------
    # 2) Bar Chart for data_num_dict
    # ----------------------------------------------------
    bar_data = go.Bar(
        x=list(data_num_dict.keys()),   # names
        y=list(data_num_dict.values()), # data_num for each name
    )

    fig_bar = go.Figure(data=[bar_data])
    fig_bar.update_layout(
        title="data_num per name (Bar Chart)",
        xaxis_title="Name",
        yaxis_title="data_num"
    )
    fig_bar.show()

def spilt_dataset_num(set_num=2, seq_data_name="data/seq_data/seq_taco_2_train.p", save=False):
    train_item_ls = []
    test_item_ls = []
    validate_item_ls = []
    for k, v in OBJ_DIC.items():
        if k in VALID_LS:
            train_item_ls.extend(v[:set_num])
            test_item_ls.extend(v[set_num:])
            validate_item_ls.append(v[set_num])
    
    seq_data = joblib.load(seq_data_name)
    train_ls = []
    test_ls = []
    validate_ls = []

    for item in seq_data:
        name = os.path.splitext(os.path.split(item['mesh_path'])[-1])[0]
        if name in train_item_ls:
            train_ls.append(item)
        if name in test_item_ls:
            test_ls.append(item)
        if name in validate_item_ls:
            validate_ls.append(item)
    
    print(f"Train set: {len(train_ls)}")
    print(f"Test set: {len(test_ls)}")
    print(f"Validate set: {len(validate_ls)}")

    # data_static(train_ls)
    
    if save:
        train_save_path = os.path.splitext(seq_data_name)[0] + f"_{set_num}_train.p"
        test_save_path = os.path.splitext(seq_data_name)[0] + f"_{set_num}_test.p"
        validate_save_path = os.path.splitext(seq_data_name)[0] + f"_{set_num}_validate.p"
        with open(train_save_path, 'wb') as ofs:
                joblib.dump(train_ls, ofs)
        with open(test_save_path, 'wb') as ofs:
                joblib.dump(test_ls, ofs)
        with open(validate_save_path, 'wb') as ofs:
                joblib.dump(validate_ls, ofs)

def get_dataset(set_num=1, seq_data_name="data/seq_data/seq_taco_2_test.p", save=True):
    item_ls = []
    for k, v in OBJ_DIC.items():
        if k in SELECT_LS:
            item_ls.extend(v)
    
    seq_data = joblib.load(seq_data_name)
    train_ls = []

    for item in seq_data:
        name = os.path.splitext(os.path.split(item['mesh_path'])[-1])[0]
        if name in item_ls:
            train_ls.append(item)
            item_ls.remove(name)

    print(f"Train set: {len(train_ls)}")

    data_static(train_ls)
    
    if save:
        train_save_path = os.path.join(os.path.split(seq_data_name)[0], f"example_{set_num}.p")
        with open(train_save_path, 'wb') as ofs:
                joblib.dump(train_ls, ofs)

def get_dataset_meta(seq_data_name="data/seq_data/seq_taco.p"):
    seq_data = joblib.load(seq_data_name)
    data_dict = {}

    for item in seq_data:
        name = os.path.splitext(os.path.split(item['mesh_path'])[-1])[0]
        category = find_key_for_value(OBJ_DIC, name)
        if category not in data_dict:
            data_dict[category] = {}
        if name not in data_dict[category]:
            data_dict[category][name] = {}
            data_dict[category][name]['item'] = []
            data_dict[category][name]['traj_num'] = 0
            data_dict[category][name]['fram_num'] = 0
        data_dict[category][name]['item'].append(item['seq_len'])
        data_dict[category][name]['traj_num'] += 1
        data_dict[category][name]['fram_num'] += item['seq_len']
    for category in data_dict:
        for item in data_dict[category]:
            data_dict[category][item]['item'] = sorted(data_dict[category][item]['item'])
    
    print(data_dict)

def spilt_dataset(set_num=2, obj_num=2, seq_data_name="data/seq_data/seq_taco_2_train.p", save=False):
    train_item_ls = []
    test_item_ls = []
    validate_item_ls = []
    for k, v in OBJ_TRAIN_DIC.items():
        v_original = OBJ_DIC[k]
        train_item_ls.extend(v[:set_num])
        for item in v:
            v_original.remove(item)
        test_item_ls.extend(v_original)
        validate_item_ls.extend(v[set_num:])
    
    seq_data = joblib.load(seq_data_name)
    train_ls = []
    test_ls = []
    validate_ls = []

    test_obj_num_dict = {}

    for item in seq_data:
        name = os.path.splitext(os.path.split(item['mesh_path'][0])[-1])[0]
        if name in train_item_ls:
            train_ls.append(item)
        if name in validate_item_ls:
            validate_ls.append(item)
        if name in test_item_ls:
            if name not in test_obj_num_dict:
                test_obj_num_dict[name] = 1
                test_ls.append(item)
            elif test_obj_num_dict[name] <= obj_num:
                test_obj_num_dict[name] +=1
                test_ls.append(item)
            else:
                continue
                 
    
    print(f"Train set: {len(train_ls)}")
    print(f"Test set: {len(test_ls)}")
    print(f"Validate set: {len(validate_ls)}")

    # data_static(train_ls)
    
    if save:
        train_save_path = os.path.splitext(seq_data_name)[0] + f"_{set_num}_train.p"
        test_save_path = os.path.splitext(seq_data_name)[0] + f"_{set_num}_test.p"
        validate_save_path = os.path.splitext(seq_data_name)[0] + f"_{set_num}_validate.p"
        with open(train_save_path, 'wb') as ofs:
                joblib.dump(train_ls, ofs)
        with open(test_save_path, 'wb') as ofs:
                joblib.dump(test_ls, ofs)
        with open(validate_save_path, 'wb') as ofs:
                joblib.dump(validate_ls, ofs)

def spilt_dataset_per_category(obj_num=2, seq_data_name="data/seq_data/seq_taco_2_train.p", save=False):
    test_object_dict = {}
    for k, v in OBJ_TRAIN_DIC.items():
        v_original = OBJ_DIC[k]
        for item in v:
            v_original.remove(item)
        test_object_dict[k] = v_original
    
    seq_data = joblib.load(seq_data_name)

    test_obj_num_dict = {}
    test_trajectory_dict = {}
    for k, v in test_object_dict.items():
        for obj_name in v:
            test_obj_num_dict[obj_name] = 0
        test_trajectory_dict[k] = []

    for item in seq_data:
        name = os.path.splitext(os.path.split(item['mesh_path'])[-1])[0]
        category_name = find_key_for_value(OBJ_DIC, name)
        if category_name:
            if name in test_object_dict[category_name]:
                if test_obj_num_dict[name] <= obj_num:
                    test_obj_num_dict[name] +=1
                    test_trajectory_dict[category_name].append(item)
                else:
                    continue
                 
    
    for key, value in test_trajectory_dict.items():
        print(f"{key}: {len(value)}")
        if save:
            test_save_path = os.path.splitext(seq_data_name)[0] + f"_{key}_test.p"
            with open(test_save_path, 'wb') as ofs:
                joblib.dump(test_trajectory_dict[key], ofs)

def traverse_dataset(set_num=2, obj_num=2, seq_data_name="data/seq_data/seq_taco_2_train.p", save=False):
    seq_data = joblib.load(seq_data_name)
    for seq_data in seq_data:
        seq_data['rh_joints']

 

if __name__ == '__main__':
    # for num in tqdm.tqdm(range(2, 9, 2)):
    #     spilt_dataset(num)

    spilt_dataset(2, seq_data_name="Taco/human_save/seq_feature_1.p", save=True)
    # traverse_dataset(set_num=2, obj_num=2, seq_data_name="Taco/human_save/seq_feature_1.p", save=False)
