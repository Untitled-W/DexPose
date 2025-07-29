"""
Texture Augmentation using ControlNet-based Stable Diffusion

This module provides functionality to:
1. Replace textures in rendered images using ControlNet + Stable Diffusion
2. Use depth and mask as control conditions
3. Generate texture variations based on natural language descriptions
4. Enable data augmentation for feature extraction

Author: Assistant
Date: 2025-06-20
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
import cv2

try:
    from diffusers import (
        StableDiffusionControlNetPipeline, 
        ControlNetModel,
        AutoencoderKL,
        DDIMScheduler
    )
    from diffusers.utils import load_image
    DIFFUSION_AVAILABLE = True
except ImportError:
    print("Warning: diffusers not available. Install with: pip install diffusers transformers accelerate")
    DIFFUSION_AVAILABLE = False

class TextureAugmentator:
    """
    A class for texture augmentation using ControlNet-based Stable Diffusion
    """
    
    def __init__(self, device='cuda', model_id="runwayml/stable-diffusion-v1-5"):
        """
        Initialize the texture augmentator
        
        Args:
            device: torch device for computation
            model_id: Stable Diffusion model identifier
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        self.pipeline = None
        self.controlnet = None
        
        if DIFFUSION_AVAILABLE:
            self._setup_pipeline()
        else:
            print("Warning: Diffusion models not available. Texture augmentation disabled.")
    
    def _setup_pipeline(self):
        """Setup the ControlNet + Stable Diffusion pipeline"""
        try:
            print("Loading ControlNet models...")
            
            # Load ControlNet for depth
            depth_controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth",
                torch_dtype=torch.float16
            )
            
            # Load Stable Diffusion pipeline with ControlNet
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=depth_controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Optimize for memory
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_xformers_memory_efficient_attention()
            
            # Use faster scheduler
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            
            print("✓ ControlNet pipeline loaded successfully")
            
        except Exception as e:
            print(f"Failed to load ControlNet pipeline: {e}")
            print("Texture augmentation will be disabled.")
            self.pipeline = None
    
    def prepare_control_image(self, depth: np.ndarray, mask: np.ndarray) -> Image.Image:
        """
        Prepare control image from depth and mask
        
        Args:
            depth: depth image as numpy array (H, W)
            mask: object mask as numpy array (H, W)
            
        Returns:
            PIL Image for ControlNet conditioning
        """
        # Normalize depth to 0-255 range
        depth_normalized = depth.copy()
        
        # Only consider depth values where mask is valid
        valid_depth = depth_normalized[mask > 0]
        if len(valid_depth) > 0:
            depth_min, depth_max = valid_depth.min(), valid_depth.max()
            if depth_max > depth_min:
                depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.ones_like(depth_normalized) * 0.5
        
        # Set background to 0 (far depth)
        depth_normalized[mask == 0] = 0.0
        
        # Convert to 0-255 uint8
        depth_control = (depth_normalized * 255).astype(np.uint8)
        
        # Convert to 3-channel for ControlNet
        depth_control_rgb = cv2.cvtColor(depth_control, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(depth_control_rgb)
    
    def augment_texture(self, 
                       rgb_image: np.ndarray,
                       depth_image: np.ndarray, 
                       mask: np.ndarray,
                       prompt: str,
                       negative_prompt: str = "blurry, low quality, distorted",
                       num_inference_steps: int = 20,
                       guidance_scale: float = 7.5,
                       controlnet_conditioning_scale: float = 1.0,
                       strength: float = 0.8) -> np.ndarray:
        """
        Augment texture using ControlNet + Stable Diffusion
        
        Args:
            rgb_image: original RGB image (H, W, 3)
            depth_image: depth image (H, W)
            mask: object mask (H, W)
            prompt: text description for new texture
            negative_prompt: negative prompt
            num_inference_steps: number of diffusion steps
            guidance_scale: classifier-free guidance scale
            controlnet_conditioning_scale: ControlNet conditioning strength
            strength: how much to change from original image
            
        Returns:
            Augmented RGB image as numpy array
        """
        if self.pipeline is None:
            print("Warning: Pipeline not available, returning original image")
            return rgb_image
        
        try:
            # Convert inputs to PIL Images
            rgb_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))
            control_image = self.prepare_control_image(depth_image, mask)
            
            # Resize to model input size (512x512 for most SD models)
            target_size = (512, 512)
            rgb_pil = rgb_pil.resize(target_size, Image.Resampling.LANCZOS)
            control_image = control_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Generate augmented image using ControlNet (text2img with depth conditioning)
            with torch.autocast(self.device.type):
                result = self.pipeline(
                    prompt=prompt,
                    image=control_image,  # Control image parameter for ControlNet
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).images[0]
            
            # Resize back to original size
            original_size = (rgb_image.shape[1], rgb_image.shape[0])  # (W, H)
            result = result.resize(original_size, Image.Resampling.LANCZOS)
            
            # Convert back to numpy array
            result_np = np.array(result) / 255.0
            
            # Blend with original image using mask
            mask_3d = np.stack([mask] * 3, axis=-1).astype(np.float32)
            blended = rgb_image * (1 - mask_3d) + result_np * mask_3d
            
            return blended
            
        except Exception as e:
            print(f"Error during texture augmentation: {e}")
            return rgb_image
    
    def batch_augment_textures(self,
                              rendered_data: List[Dict],
                              texture_prompts: List[str],
                              **kwargs) -> List[Dict]:
        """
        Augment textures for multiple rendered views
        
        Args:
            rendered_data: list of rendered view dictionaries
            texture_prompts: list of texture description prompts
            **kwargs: additional arguments for augment_texture
            
        Returns:
            List of augmented view dictionaries
        """
        augmented_data = []
        
        for i, data in enumerate(rendered_data):
            print(f"Augmenting texture for view {i+1}/{len(rendered_data)}...")
            
            augmented_views = []
            for j, prompt in enumerate(texture_prompts):
                print(f"  Generating texture variation {j+1}: '{prompt}'")
                
                augmented_rgb = self.augment_texture(
                    rgb_image=data['rgb'],
                    depth_image=data['depth'],
                    mask=data['mask'],
                    prompt=prompt,
                    **kwargs
                )
                
                # Create new data entry for augmented view
                augmented_view = data.copy()
                augmented_view['rgb'] = augmented_rgb
                augmented_view['texture_prompt'] = prompt
                augmented_view['augmentation_id'] = j
                augmented_view['original_view_id'] = i
                
                augmented_views.append(augmented_view)
            
            augmented_data.extend(augmented_views)
        
        return augmented_data
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Example texture prompts for different materials
TEXTURE_PROMPTS = {
    'wood': [
        "wooden texture, natural wood grain, oak wood surface",
        "dark walnut wood texture, rich brown wooden surface",
        "bamboo texture, natural bamboo surface pattern"
    ],
    'metal': [
        "brushed metal texture, steel surface, metallic finish",
        "copper texture, oxidized copper surface, patina",
        "gold metal texture, shiny golden surface"
    ],
    'fabric': [
        "fabric texture, cotton textile, soft cloth surface",
        "denim texture, blue jean fabric pattern",
        "leather texture, brown leather surface"
    ],
    'stone': [
        "marble texture, white marble surface with veins",
        "granite texture, speckled stone surface",
        "concrete texture, rough cement surface"
    ],
    'paper': [
        "paper texture, white paper surface, cardboard",
        "vintage paper texture, aged parchment surface",
        "newspaper texture, printed text background"
    ]
}

def get_texture_prompts(material_type: str = 'wood') -> List[str]:
    """Get predefined texture prompts for a material type"""
    return TEXTURE_PROMPTS.get(material_type, TEXTURE_PROMPTS['wood'])
