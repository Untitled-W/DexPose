"""
Mesh Rendering and Feature Extraction System using PyTorch3D

This module provides functionality to:
1. Load 3D meshes from .obj files
2. Render RGB and depth images from multiple viewpoints
3. Extract features from RGB images using existing featurizers
4. Convert depth to point clouds with proper coordinate alignment
5. Visualize the results

Author: Qianxu, Assistant
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime
import json
from vision.featurizer import SDFeaturizer, DINOFeaturizer, CLIPFeaturizer, DINOv2Featurizer, RADIOFeaturizer, SD_DINOv2Featurizer
# from .vis_utils import visualize_colorpcs, project_pca, extract_pca, get_matched_colors, test_pca_matching, get_multiview_dff

# PyTorch3D imports
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, load_ply
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    BlendParams,
    PointLights,
    AmbientLights,
    look_at_view_transform
)
from pytorch3d.renderer.mesh.shader import HardDepthShader
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_farthest_points

# Open3D for enhanced point cloud visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. Install with: pip install open3d")

# Import existing featurizers
import sys
# sys.path.append('/home/qianxu/Project25/RAM_code')
sys.path.append('/home/qianxu/Desktop/New_Folder/data_process')
sys.path.append('/home/qianxu/Project25/feature_extract')
from vision.featurizer.run_featurizer import extract_ft_tensor
from vision.featurizer.utils.visualization import IMG_SIZE
from scipy.spatial.transform import Rotation as R

from data_process.texture_augmentation import TextureAugmentator, get_texture_prompts

class MeshRenderer3D:
    """
    A class for rendering 3D meshes from multiple viewpoints and extracting features
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize the renderer
        
        Args:
            device: torch device for computation
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mesh = None
        self.camera_positions = []
        self.rendered_data = []
        
        # Initialize enhanced modules
        self.texture_augmentator = None
        self.feature_pca_visualizer = None
        self.pointcloud_processor = None
        
    def load_mesh(self, obj_path: str, scale=1) -> None:
        """
        Load mesh from .obj file
        
        Args:
            obj_path: path to the .obj file
        """
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Mesh file not found: {obj_path}")
            
        # Load the obj file with texture loading enabled
        # verts, faces, aux = load_obj(
        #     obj_path, 
        #     device=self.device,
        #     load_textures=True,
        #     create_texture_atlas=False,  # Use UV mapping instead of atlas
        # )
        
        ### 0701 WMQ fixed!
        # if obj file, using load obj, if ply file using load_ply, if stl file using load_stl
        if obj_path.endswith('.obj'):
            verts, faces, aux = load_obj(
                obj_path, 
                device=self.device,
                load_textures=True,
                create_texture_atlas=False,  # Use UV mapping instead of atlas
            )
            mesh_faces = faces.verts_idx
        elif obj_path.endswith('.ply'):
            verts, faces = load_ply(
                obj_path, 
                # load_textures=True,
                # create_texture_atlas=False,  # Use UV mapping instead of atlas
            )
            aux = None  # PLY does not have aux data like textures
            mesh_faces = faces  # faces is already a tensor of indices
        else:
            raise ValueError(f"Unsupported file format: {obj_path}. Supported formats are .obj, .ply")

        # Scale vertices if needed
        verts = verts * scale
        
        # Create mesh
        self.mesh = Meshes(verts=[verts], faces=[mesh_faces])
        
        # Try to load proper UV textures first
        if (hasattr(aux, 'texture_images') and aux.texture_images is not None and
            hasattr(aux, 'verts_uvs') and aux.verts_uvs is not None and
            hasattr(faces, 'textures_idx') and faces.textures_idx is not None):
            
            # print("Loading UV textures...")
            from pytorch3d.renderer import TexturesUV
            
            # Load texture images if they are paths
            texture_maps = []
            for i, tex in enumerate(aux.texture_images):
                if isinstance(tex, str):
                    # Try to find the actual texture file
                    texture_candidates = [
                        "texture_map.png", "texture.png", "diffuse.png", "albedo.png",
                        f"{tex}.png", f"{tex}.jpg"
                    ]
                    
                    tex_path = None
                    model_dir = os.path.dirname(obj_path)
                    
                    for candidate in texture_candidates:
                        candidate_path = os.path.join(model_dir, candidate)
                        if os.path.exists(candidate_path):
                            tex_path = candidate_path
                            break
                    
                    if tex_path and os.path.exists(tex_path):
                        # Load image
                        from PIL import Image
                        import numpy as np
                        img = Image.open(tex_path).convert('RGB')
                        img_tensor = torch.tensor(np.array(img), dtype=torch.float32, device=self.device) / 255.0
                        texture_maps.append(img_tensor)
                        print(f"Loaded texture from {tex_path}, shape: {img_tensor.shape}")
                    else:
                        print(f"Texture file not found for material: {tex}")
                        break
                else:
                    texture_maps.append(tex)
            
            if texture_maps:
                textures = TexturesUV(
                    maps=texture_maps,
                    faces_uvs=[faces.textures_idx],
                    verts_uvs=[aux.verts_uvs]
                )
                self.mesh.textures = textures
                print("✓ Successfully loaded UV textures")
            else:
                # Fallback to vertex colors
                self._add_default_vertex_colors(verts)
        else:
            # Add default vertex colors if no textures are available
            self._add_default_vertex_colors(verts)

    def _calculate_optimal_camera_distance(self, fov: float = 60.0, coverage_ratio: float = 0.8) -> float:
        """
        Calculate optimal camera distance to fit the object in the view based on actual mesh size
        
        Args:
            fov: field of view in degrees
            coverage_ratio: what fraction of the image should be covered by the object (0.8 = 80%)
            
        Returns:
            optimal distance from object center
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Cannot calculate camera distance without mesh.")
            
        # Get the bounding box of the mesh
        # Returns tensor of shape (1, 3, 2) where bbox[0, j] gives [min, max] for axis j
        bbox = self.mesh.get_bounding_boxes()  # Shape: (1, 3, 2)
        
        # Calculate the extent (size) along each axis
        extents = bbox[0, :, 1] - bbox[0, :, 0]  # Shape: (3,) for [x_extent, y_extent, z_extent]
        
        # Use the maximum extent to ensure the entire object fits in view
        max_extent = torch.max(extents).item()
        
        # Convert FOV to radians
        fov_rad = np.radians(fov)
        
        # Calculate optimal distance using perspective projection geometry
        # The object should occupy coverage_ratio of the image
        # For an object with extent 'max_extent', we want it to occupy coverage_ratio of the viewport
        optimal_dist = max_extent / (2 * np.tan(fov_rad / 2) * coverage_ratio)
        
        # Add a safety margin to avoid clipping
        optimal_dist *= 1.2
        
        # Ensure minimum distance to avoid numerical issues and clipping
        min_dist = max_extent * 1.5  # At least 1.5x the object size
        optimal_dist = max(optimal_dist, min_dist)
        
        # print(f"Mesh extents: {extents.cpu().numpy()}")
        # print(f"Max extent: {max_extent:.4f}")
        # print(f"Calculated optimal distance: {optimal_dist:.4f}")
        
        return optimal_dist
        
    def setup_cameras(self, camera_params: List[Dict], auto_distance: bool = True) -> None:
        """
        Setup cameras for rendering
        
        Args:
            camera_params: List of camera parameter dictionaries
                Each dict should contain:
                - elev: elevation angle in degrees
                - azim: azimuth angle in degrees
                - fov: field of view in degrees (optional, default 60)
                - dist: distance from object (optional, will be auto-calculated if auto_distance=True)
            auto_distance: if True, automatically calculate optimal distance based on mesh size
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
            
        self.cameras = []
        self.camera_infos = []
        
        for params in camera_params:
            elev = params.get('elev', 0.0)
            azim = params.get('azim', 0.0)
            fov = params.get('fov', 60.0)
            if auto_distance or 'dist' not in params:
                dist = self._calculate_optimal_camera_distance(fov=fov)
            else:
                dist = params.get('dist', 2.0)
            
            # Generate camera position using PyTorch3D's look_at_view_transform
            # This returns world-to-view transformation components (R, T)
            R_w2v_yup, T_w2v_yup = look_at_view_transform(
                dist=dist, 
                elev=elev, 
                azim=azim,
                device=self.device
            )

            ### Align the y-axis of the camera to the z-axis of the mesh
            w2v_yup = torch.eye(4).unsqueeze(0).to(R_w2v_yup.device)
            w2v_yup[..., :3, :3] = R_w2v_yup
            w2v_yup[..., :3, 3] = T_w2v_yup
            v2w_yup = torch.inverse(w2v_yup)
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = R.from_rotvec([0, 0, np.pi/2]).as_matrix()
            v2w_xup = v2w_yup @ torch.from_numpy(rotation_matrix).to(R_w2v_yup.device).unsqueeze(0).to(torch.float32)
            w2v_xup = torch.inverse(v2w_xup)
            R_w2v = w2v_xup[..., :3, :3]
            T_w2v = w2v_xup[..., :3, 3]

            camera = FoVPerspectiveCameras(
                device=self.device,
                R=R_w2v,
                T=T_w2v,
                fov=fov,
                znear=0.01,  # Near plane to avoid clipping
            )
            
            self.cameras.append(camera)
            self.camera_infos.append({
                'elev': elev,
                'azim': azim,
                'dist': dist,
                'fov': fov,
            })
            return R_w2v, T_w2v
    
    def _setup_lighting(self, lighting_type: str = 'multi_point') -> object:
        """
        Setup lighting for natural and balanced illumination
        
        Args:
            lighting_type: 'multi_point', 'ambient', or 'single_point'
            
        Returns:
            PyTorch3D lights object
        """
        if lighting_type == 'multi_point':
            # Multiple point lights positioned around the object for balanced illumination
            # Position lights at different angles and distances
            light_positions = [
                [2.0, 2.0, 2.0],    # Front-top-right
                [-2.0, 2.0, 2.0],   # Front-top-left  
                [2.0, 2.0, -2.0],   # Back-top-right
                [-2.0, 2.0, -2.0],  # Back-top-left
                [0.0, -1.0, 2.0],   # Bottom-front (fill light)
            ]
            
            # Balanced color intensities - slightly warm light
            ambient_intensity = 0.4
            diffuse_intensity = 0.5
            specular_intensity = 0.1
            
            lights = PointLights(
                location=light_positions,
                ambient_color=[[ambient_intensity] * 3] * len(light_positions),
                diffuse_color=[[diffuse_intensity] * 3] * len(light_positions), 
                specular_color=[[specular_intensity] * 3] * len(light_positions),
                device=self.device
            )
            
        elif lighting_type == 'ambient':
            # Enhanced ambient lighting - brighter and more natural
            lights = AmbientLights(
                ambient_color=[[1.0, 1.0, 1.0]],  # Full brightness white light
                device=self.device
            )
            
        else:  # single_point
            # Original single point light setup
            lights = PointLights(
                location=[[2.0, 2.0, 2.0]],
                ambient_color=[[0.3, 0.3, 0.3]],
                diffuse_color=[[0.7, 0.7, 0.7]],
                specular_color=[[0.2, 0.2, 0.2]],
                device=self.device
            )
            
        return lights
    
    def render_views(self, image_size: Union[int, Tuple[int, int]] = 512, background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0), lighting_type: str = 'auto') -> None:
        """
        Render RGB and depth images from all camera viewpoints
        
        Args:
            image_size: size of rendered images (square) or (height, width)
            background_color: RGB background color (0-1 range)
            lighting_type: 'multi_point', 'ambient', 'single_point', or 'auto' (auto-select based on mesh complexity)
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
        if not self.cameras:
            raise ValueError("No cameras setup. Call setup_cameras() first.")
            
        self.rendered_data = []
        # self.clear_gpu_cache()
        
        # Setup rasterization settings
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=20000
        )
        
        # Setup improved lighting
        lights = self._setup_lighting(lighting_type)
        # print(f"Using {lighting_type} lighting for balanced illumination")
    
        # Setup blend parameters for black background
        blend_params = BlendParams(
            sigma=1e-4,
            gamma=1e-4,
            background_color=background_color
        )
        
        ### here we can also duplicate the mesh to render multiple views in parallel
        for i, camera in enumerate(self.cameras):
            
            # RGB renderer
            rgb_rasterizer = MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            )
            
            rgb_renderer = MeshRenderer(
                rasterizer=rgb_rasterizer,
                shader=HardPhongShader(
                    device=self.device,
                    cameras=camera,
                    lights=lights,
                    blend_params=blend_params
                )
            )
            
            # Depth renderer
            depth_rasterizer = MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            )
            depth_renderer = MeshRenderer(
                rasterizer=depth_rasterizer,
                shader=HardDepthShader(device=self.device)
            )
            
            with torch.no_grad():
                rgba_image = rgb_renderer(self.mesh, cameras=camera)  # (B, H, W, 4) RGBA
                depth_image = depth_renderer(self.mesh, cameras=camera)  # (B, H, W, 1)

            pointcloud = get_rgbd_point_cloud(
                camera=camera,
                image_rgb=rgba_image[..., :3].permute((0, 3, 1, 2)),
                depth_map=depth_image.permute((0, 3, 1, 2)),
                mask=(rgba_image[..., 3:] > 0).permute((0, 3, 1, 2)).to(torch.bool),
            )
            points = pointcloud.points_packed()

            mask=(rgba_image[0, ..., 3] > 0).to(torch.bool) # (H, W) Mask where alpha > 0
            # print(f"Camera {i}: mask pixels = {mask.sum()} / {mask.numel()} "
                #   f"({100 * mask.sum() / mask.numel():.1f}%)")
            
            # visualize_pointclouds_and_mesh(
            #     pointcloud_list=[points.cpu().numpy()],
            #     color_list=[rgba_image[0, ..., :3][mask].cpu().numpy()],
            #     mesh=self.mesh,
            #     view_names=['Rendered Mesh'],
            #     title='Mesh Renderer RGBD Fusion',
            #     save_path='/home/qianxu/Project25/RAM_code/test_mesh_renderer_fusion.html',
            #     use_plotly=True,
            #     point_size=4
            # )

            self.rendered_data.append({
                'camera_idx': i,
                'camera': camera,
                'rgb': rgba_image[0, ..., :3],
                'depth': depth_image[0, ..., 0],
                'points': points, # (N, 3) Point cloud coordinates after masking
                'mask': mask,
            })
            
        # print(f"Rendered {len(self.rendered_data)} views")
    
    def load_featurizer(self, ftype):
        featurizers = {
            'sd': SDFeaturizer,
            'clip': CLIPFeaturizer,
            'dino': DINOFeaturizer,
            'dinov2': DINOv2Featurizer,
            'radio': RADIOFeaturizer,
            'sd_dinov2': SD_DINOv2Featurizer
        }
        assert ftype in ['sd', 'clip', 'dino', 'dinov2', 'radio', 'sd_dinov2']
        self.ftype = ftype
        self.featurizer = featurizers[ftype]()

    def extract_features(self, prompt: Optional[str] = None, w_aug_texture:bool=True) -> None:
        """
        Extract features from RGB images using existing featurizers
        
        Args:
            ftype: feature type ('sd', 'clip', 'dino', 'dinov2', 'radio', 'sd_dinov2')
            prompt: prompt for SD-based featurizers
        """
        if not self.rendered_data:
            raise ValueError("No rendered data. Call render_views() first.")
            
        for data in self.rendered_data:
            ###0701 fix! due to python3.8 type error, load torch model may fail.
            if w_aug_texture:
                imgs = data['aug_rgbs']
                features_ls = []
                for img in imgs:
                    features = extract_ft_tensor(img.permute(2, 0, 1), self.featurizer,
                                         prompt=prompt, 
                                         dest_size=img.shape[:2])
                    features_ls.append(features)
                data['features'] = torch.cat(features_ls, dim=0)
                data['feature_type'] = self.ftype
            else:
                features = extract_ft_tensor(data['rgb'].permute(2, 0, 1), self.featurizer,
                                            prompt=prompt, 
                                            dest_size=data['rgb'].shape[:2])
                data['features'] = features
                data['feature_type'] = self.ftype
            
        # print(f"Extracted {ftype} features for {len(self.rendered_data)} views")
    
    def visualize_view(self, camera_idx: int, save_path: Optional[str] = None, show_features: bool = True) -> None:
        """
        Visualize a specific rendered view
        
        Args:
            camera_idx: index of the camera view
            save_path: path to save the visualization (optional)
            show_features: whether to show feature visualization
        """
        if camera_idx >= len(self.rendered_data):
            raise ValueError(f"Invalid camera index: {camera_idx}")
            
        data = self.rendered_data[camera_idx]
        
        # Setup figure
        n_cols = 4 if (show_features and 'features' in data) else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
        
        # RGB image
        axes[0].imshow(data['rgb'].cpu().numpy())
        axes[0].set_title(f"RGB - Camera {camera_idx}")
        axes[0].axis('off')
        
        # Depth image
        depth_vis = data['depth'].cpu().numpy().copy()
        depth_vis[data['mask'].cpu().numpy() == 0] = np.nan  # Set background to NaN for better visualization
        im = axes[1].imshow(depth_vis, cmap='jet')
        axes[1].set_title(f"Depth - Camera {camera_idx}")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Mask
        axes[2].imshow(data['mask'].cpu().numpy(), cmap='gray')
        axes[2].set_title(f"Mask - Camera {camera_idx}")
        axes[2].axis('off')
        
        # Features (if available)
        if show_features and 'features' in data:
            # Visualize first channel of features
            features = data['features'][0].cpu().numpy()  # (C, H, W)
            # Average across channels for visualization
            feat_vis = np.mean(features, axis=0)
            im = axes[3].imshow(feat_vis, cmap='viridis')
            axes[3].set_title(f"Features - {data.get('feature_type', 'Unknown')}")
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def visualize_all_views(self, save_dir: Optional[str] = None, show_features: bool = True) -> None:
        """
        Visualize all rendered views
        
        Args:
            save_dir: directory to save visualizations (optional)
            show_features: whether to show feature visualizations
        """
        # if save_dir and not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
            
        for i in range(len(self.rendered_data)):
            save_path = None
            if save_dir:
                # save_path = os.path.join(save_dir, f"view_{i:03d}.png")
                save_path = os.path.join(save_dir)
            self.visualize_view(i, save_path, show_features)
    
    def save_data(self, save_dir: str) -> None:
        """
        Save all rendered data to disk
        
        Args:
            save_dir: directory to save data
        """
        assert self.rendered_data, "No rendered data to save. Call render_views() first."

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save metadata
        metadata = {
            'num_views': len(self.rendered_data),
            'camera_params': [data for data in self.camera_infos],
            'feature_type': self.rendered_data[0].get('feature_type', None) if self.rendered_data else None
        }
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        ### For visualization
        self.visualize_all_views(
            save_dir=save_dir, 
            show_features='features' in self.rendered_data[0]
        )

        ### Save rendered data
        all_views_rgb = torch.stack([data['rgb'] for data in self.rendered_data], dim=0)  # (N, H, W, 3)
        all_views_depth = torch.stack([data['depth'] for data in self.rendered_data], dim=0)  # (N, H, W)
        all_views_mask = torch.stack([data['mask'] for data in self.rendered_data], dim=0)  # (N, H, W)
        all_views_points = torch.cat([data['points'] for data in self.rendered_data], dim=0)  # (P, 3)
        
        torch.save(all_views_rgb.cpu(), os.path.join(save_dir, 'all_views_rgb.pt'))
        torch.save(all_views_depth.cpu(), os.path.join(save_dir, 'all_views_depth.pt'))
        torch.save(all_views_mask.cpu(), os.path.join(save_dir, 'all_views_mask.pt'))
        torch.save(all_views_points.cpu(), os.path.join(save_dir, 'all_views_points.pt'))
        if 'features' in self.rendered_data[0]:
            # Stack features if available
            all_views_features = torch.cat([data['features'] for data in self.rendered_data], dim=0)
            torch.save(all_views_features.cpu(), os.path.join(save_dir, 'all_views_features.pt'))
        print(f"Saved all data to {save_dir}")

    def get_rendered_data(self) -> None:
        """
        Return all rendered data
        """
        assert self.rendered_data, "No rendered data to return. Call render_views() first."

        # prepare metadata
        # metadata = {
        #     'num_views': len(self.rendered_data),
        #     'camera_params': [data for data in self.camera_infos],
        #     'feature_type': self.rendered_data[0].get('feature_type', None) if self.rendered_data else None
        # }

        ### prepare rendered data
        all_views_rgb = torch.stack([data['rgb'] for data in self.rendered_data], dim=0)  # (N, H, W, 3)
        all_views_depth = torch.stack([data['depth'] for data in self.rendered_data], dim=0)  # (N, H, W)
        all_views_mask = torch.stack([data['mask'] for data in self.rendered_data], dim=0)  # (N, H, W)
        all_views_points = [data['points'] for data in self.rendered_data]

        if 'features' in self.rendered_data[0]:
            # Stack features if available
            all_views_features = torch.stack([data['features'] for data in self.rendered_data], dim=0)
        else:
            all_views_features = None

        return all_views_rgb, all_views_depth, all_views_mask, all_views_points, all_views_features

    def _add_default_vertex_colors(self, verts):
        """Add default gray vertex colors to mesh"""
        from pytorch3d.renderer import TexturesVertex
        # Create simple gray vertex colors
        verts_rgb = torch.ones_like(verts)[None] * 0.7  # Gray color
        textures = TexturesVertex(verts_features=verts_rgb)
        self.mesh.textures = textures
        # print("Added default vertex colors")
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free up memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # print("Cleared GPU cache")
    

    def augment_textures(self, 
                        texture_prompts: List[str],
                        strength: float = 0.8,
                        guidance_scale: float = 7.5,
                        save_dir: Optional[str] = None,
                        **kwargs) -> Dict[str, List[np.ndarray]]:
        """
        Apply ControlNet-based texture augmentation to rendered images
        
        Args:
            texture_prompts: list of texture descriptions for each view
            num_variations: number of texture variations per view
            strength: ControlNet conditioning strength (0-1)
            guidance_scale: Stable Diffusion guidance scale
            save_dir: optional directory to save augmented images
            
        Returns:
            Dictionary mapping view indices to lists of augmented images
        """
        if not self.rendered_data:
            raise ValueError("No rendered data available. Call render_views() first.")
            
        if not self.texture_augmentator:
            print("Initializing texture augmentator...")
            self.texture_augmentator = TextureAugmentator(device=self.device)
        
        for i, data in enumerate(self.rendered_data):
            if 'rgb' not in data or 'depth' not in data:
                print(f"Skipping view {i}: missing RGB or depth data")
                continue
            
            # Prepare RGB and depth as numpy arrays for texture augmentation
            rgb_np = data['rgb'].cpu().numpy()  # in 0-1 range
            depth_np = data['depth'].cpu().numpy()  # in 0-100 range
            depth_np[depth_np == 100] = 0
            mask_np = data['mask'].cpu().numpy()  # (H, W) Mask where alpha > 0
            
            # Generate augmented textures (multiple variations)
            augmented_images_ls = []
            for prompt in texture_prompts:
                augmented_img_np = self.texture_augmentator.augment_texture(
                    rgb_image=rgb_np, # (h, w, 3) RGB image in 0-1 range
                    depth_image=depth_np, # (h, w) Depth image in 0-100 range
                    mask=mask_np,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=strength,
                )
                augmented_img = torch.from_numpy(augmented_img_np).to(torch.float32).to(data['rgb'].device)
                augmented_images_ls.append(augmented_img)
            
            augmented_images = torch.stack(augmented_images_ls, dim=0)  # (num_variations, H, W, 3)
            data['aug_rgbs'] = augmented_images
            # Save if directory provided
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                for j, aug_img in enumerate(augmented_images_ls):
                    save_path = os.path.join(save_dir, f"view_{i:03d}_texture_{j}.png")

                    aug_img_pil = Image.fromarray((aug_img.cpu().numpy() * 255).astype(np.uint8))
                    aug_img_pil.save(save_path)