import pickle

# with open('/home/wangminqi/workspace/test/data/Taco/dex_save1002/shadow_1.p', 'rb') as f:
with open("/home/wangminqi/workspace/test/data/Taco/human_save0929/seq_mirror_correct_1.p", "rb") as f:
    data = pickle.load(f)

print(len(data))

# all_set = [data[i//2]['which_sequence'] for i in [1,3,75,77,147,159,161,221,223,281,283,393,395,415,417,423,425,439,453]]

# # 261, 353, 

# for ii in all_set:
#     print(f"'{ii}',")


# from utils.vis_utils import visualize_dex_hand_sequence

# import os
# os.makedirs('/home/wangminqi/workspace/test/DexPose/retarget_wmq/Taco_vis/seq_vis/',exist_ok=True)
# for d in data:
#     visualize_dex_hand_sequence(d, f"/home/wangminqi/workspace/test/DexPose/retarget_wmq/Taco_vis/seq_vis/{d['uid']}")