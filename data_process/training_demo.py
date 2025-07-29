import torch
import sys
import os
import pickle
import numpy as np
import trimesh
from manotorch.manolayer import ManoLayer
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import WORKING_DIR
from models.models import HandInterfaceTransformer


def load_data():

    oakink2_results = f'{WORKING_DIR}data/seq_data/seq_oakinkv2_0704_5.p'
    taco_results = f'{WORKING_DIR}data/seq_data/seq_taco_0704_2.p'

    with open(oakink2_results, 'rb') as f:
        oakink2_data = pickle.load(f)

    with open(taco_results, 'rb') as f:
        taco_data = pickle.load(f)

    return oakink2_data + taco_data



class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_list=None):
        """
            data: List of dictionaries containing sequence data.
        """
        loaded_ori_data = load_data()
        self.data = []

        for seq_id, seq in enumerate(loaded_ori_data):
        
            side = 'l' if seq['side']==0 else 'r'
            self.data.append((seq[f'{side}o_points_ori'],
                    seq[f'{side}o_num_ori'],
                    seq[f'{side}o_transf'],
                    seq[f'{side}h_joints']))
            
            print(seq_id, seq['extra_desc'], seq['side'])

        # already in torch.tensor
        self.data_len = [len(seq[-1]) for seq in self.data]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get item by index.
        Args:
            idx: Index of the item to retrieve.
        Returns:
            A tuple containing:
                - object_pc: torch.Tensor of shape (num_points, 3)
                - hand_data: torch.Tensor of shape (num_frames, num_keypoints, 3)
        """
        object_pc, obj_num, obj_transformations, hand_joints = self.data[idx]
        # frame_id = torch.randint(0, len(hand_joints), (1,)).item()  # Randomly select a frame
        frame_id = int(.5 * len(hand_joints))  # For demo, always use the first frame

        def pc_i(f_id):
            pc = object_pc.clone()  # Copy the object point cloud
            cur_id = 0
            for i in range(obj_num.shape[0]):
                transformation = obj_transformations[i][f_id]
                pc[cur_id:cur_id+obj_num[i]] = torch.matmul(pc[cur_id:cur_id+obj_num[i]], transformation[:3, :3].T) + transformation[:3, 3]
                cur_id += obj_num[i]
            return pc

        VIS_ONE_FRAME = False
        VIS_ALL = True

        if VIS_ONE_FRAME == True:
            from loss import vis_pc_coor_plotly
            vis_pc_coor_plotly([pc_i(frame_id)], 
                        gt_hand_joints=hand_joints[frame_id], 
                        show_axis=True,
                        filename=f'visualize_0705_test_dataset/1000_points/{idx}',
                        )
        if VIS_ALL == True:
            from loss import vis_frames_plotly
            vis_frames_plotly([np.asarray([pc_i(i) for i in range(len(hand_joints))])],
                            gt_hand_joints=hand_joints,
                            show_axis=True,
                            filename=f'logs/visualize_0705_test_dataset/debug_camera_view_root/{idx}',
                            )
                            
        
        return pc_i(frame_id), hand_joints[frame_id]



def loss(k_hand_interface: torch.Tensor, origin_hand):
    """
    Calculate the loss between the predicted hand interface and the original hand data.
    Args:
        k_hand_interface: Predicted hand interface data. 
            torch.Tensor of shape (batch_size, num_function, dim_interface).
        origin_hand: Original hand data. 
            ???
    Returns:
        loss_value: Computed loss value.
    """

    pass


def main():
    training_dataset = SimpleDataset()  # Pass an empty list for now, will be filled with data later
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, batch_size=1, shuffle=True, num_workers=0
    )

    for i, data in enumerate(training_dataloader):
        object_pc, hand_data = data

    return 
    
    model = HandInterfaceTransformer()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 0703 temp! wrap loss into torch.nn.Module in the future
    criterion = loss

    for epoch in range(10):  # number of epochs
        for batch in training_dataloader:
            # Unpack the batch data
            object_pc, hand_data = batch

            # rand int from 2 to 6
            k_query = torch.randint(2, 6, (1,)).item()
            
            hand_interface = model(object_pc, k_query)
            # hand_interface: torch.Tensor of shape (batch_size, k_query, dim_interface)

            # loss_value = criterion(hand_interface, hand_data)

            # optimizer.zero_grad()
            # loss_value.backward()
            # optimizer.step()    
            # print(f'Epoch [{epoch+1}/10], Loss: {loss_value.item():.4f}')
            print(f'Epoch [{epoch+1}/10], Batch [{i+1}/{len(training_dataloader)}], k_query: {k_query}, Hand Interface Shape: {hand_interface.shape}')


if __name__ == "__main__":
    main()




