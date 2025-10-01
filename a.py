import pickle

with open("/home/qianxu/Desktop/Project/DexPose/data/Taco/human_save0929/seq_mirror_correct_1.p", "rb") as f:
    data = pickle.load(f)


all_set = set(data[i]['which_sequence'] for i in [0,1,2,3,74,75,76,77,146,147,158,159,160,161,202,203,220,221,282,283])

for ii in all_set:
    print(f"'{ii}',")