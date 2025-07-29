from data_process.mesh_renderer import MeshRenderer3D
from data_process.vis_utils import visualize_pointclouds_and_mesh, get_multiview_dff, extract_pca, test_pca_matching
import os
import torch
import numpy as np

def demo_usage():
    """Test with actual MeshRenderer3D data if available"""
    
    # Check if we have a sample OBJ file
    obj_path = "/home/qianxu/dataset/models/003_cracker_box/textured_simple.obj"
    # obj_path = "/home/qianxu/Desktop/New_Folder/DRO-Grasp/data/data_urdf/object/contactdb/airplane/coacd_allinone.obj"
    if not os.path.exists(obj_path):
        print(f"Sample OBJ file not found at {obj_path}")
        print("Skipping MeshRenderer3D test")
        return None
        
    print(f"Loading mesh from {obj_path}")
    
    # Create renderer and load mesh
    
    renderer = MeshRenderer3D()  # Use CPU to avoid GPU issues
    renderer.load_mesh(obj_path)
    
    # Setup cameras
    camera_params = [
        {'elev': 30, 'azim': 30, 'fov': 60},     # Front
        {'elev': 30, 'azim': 120, 'fov': 60},    # Right
        {'elev': 30, 'azim': 210, 'fov': 60},   # Back
        {'elev': 30, 'azim': 300, 'fov': 60},   # Left
    ]
    renderer.setup_cameras(camera_params, auto_distance=True)
    renderer.render_views(image_size=256, lighting_type='ambient')
    # renderer.save_data("/home/qianxu/Project25/RAM_code/test_resdir")
    renderer.load_featurizer(ftype='sd_dinov2')
    renderer.extract_features(prompt="a product package, cracker box")
    # renderer.visualize_all_views(save_dir="./test_resdir", show_features=True)
    
    
    rgbs, depths, masks, points_ls, features = renderer.get_rendered_data()
    '''
    Points_ls: [torch.Size([N_i, 3]), ...] for each view 
    Features: torch.Size([view_num, feature_dim, H, W])
    rgbs: torch.Size([view_num, H, W, 3])
    '''
    
    all_points_dff, all_features_dff = get_multiview_dff(points_ls, masks, features,
                                                 n_points=1000)
    '''
    all_points_dff: torch.Size([N_points, 3])
    all_features_dff: torch.Size([N_points, feature_dim])
    '''
    
    pca = extract_pca(all_features_dff, n_components=3)
    # pca.components_: torch.Size([n_components, feature_dim])

    points_dff_ls = [all_points_dff]
    features_dff_ls = [all_features_dff]
    
    for idx in range(len(rgbs)):
        points_dff, features_dff = get_multiview_dff(
            [points_ls[idx]], masks[idx].unsqueeze(0), features[idx].unsqueeze(0),
            n_points=250,
        )
        points_dff_ls.append(points_dff)
        features_dff_ls.append(features_dff)
    
    test_pca_matching(features_dff_ls, points_dff_ls, pca=pca)
    
    # visualize_pointclouds_and_mesh(
    #     pointcloud_list=[point.cpu().numpy()],
    #     color_list=[rgb[mask].cpu().numpy()],
    #     mesh=renderer.mesh,
    #     view_names=['Rendered Mesh'],
    #     title='Mesh Renderer RGBD Fusion',
    #     save_path='./test_mesh_renderer_fusion.html',
    #     use_plotly=True,
    #     point_size=4
    # )
    visualize_pointclouds_and_mesh(
        pointcloud_list=[point.cpu().numpy() for point in points_ls],
        color_list=[rgb[mask].cpu().numpy() for rgb, mask in zip(rgbs, masks)],
        mesh=renderer.mesh,
        # view_names=['Rendered Mesh'],
        view_names = [f'View {i+1}' for i in range(len(rgbs))],
        title='Mesh Renderer RGBD Fusion',
        save_path='./test_mesh_renderer_fusion.html',
        # use_plotly=True,
        point_size=4
    )


if __name__ == "__main__":
    demo_usage()
