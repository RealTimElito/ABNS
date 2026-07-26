"""Robustness testing utilities"""

import numpy as np
import torch
from PIL import Image
import cv2
import io


def apply_jpeg_compression(image_tensor, quality=85):
    """Apply JPEG compression to test robustness.
    
    Args:
        image_tensor: Tensor [1, 3, H, W] or [3, H, W] in range [-1, 1]
        quality: JPEG quality (1-100)
        
    Returns:
        compressed_tensor: Tensor with same shape as input
    """
    # Convert to PIL Image
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # Denormalize
    image_np = ((image_tensor.cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0 * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    
    # Apply JPEG compression
    buffer = io.BytesIO()
    image_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed_pil = Image.open(buffer).convert('RGB')
    
    # Convert back to tensor
    compressed_np = np.array(compressed_pil).astype(np.float32) / 255.0
    compressed_np = compressed_np.transpose(2, 0, 1)
    compressed_tensor = torch.from_numpy(compressed_np) * 2.0 - 1.0
    
    if image_tensor.dim() == 3:
        return compressed_tensor.unsqueeze(0)
    return compressed_tensor


def apply_resize(image_tensor, scale_factor=0.8):
    """Apply resizing to test robustness.
    
    Args:
        image_tensor: Tensor [1, 3, H, W] or [3, H, W] in range [-1, 1]
        scale_factor: Scale factor (e.g., 0.8 for 80% size)
        
    Returns:
        resized_tensor: Tensor resized and then rescaled back to original size
    """
    from torchvision.transforms import functional as F
    
    if image_tensor.dim() == 4:
        batch_size = image_tensor.shape[0]
        image_tensor = image_tensor.squeeze(0)
    else:
        batch_size = 1
    
    # Denormalize
    image_tensor = (image_tensor + 1.0) / 2.0
    
    # Resize down
    h, w = image_tensor.shape[1], image_tensor.shape[2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized = F.resize(image_tensor.unsqueeze(0), (new_h, new_w))
    
    # Resize back up
    resized = F.resize(resized, (h, w))
    
    # Renormalize
    resized = resized * 2.0 - 1.0
    
    if batch_size == 1:
        return resized.squeeze(0)
    return resized


def apply_color_shift(image_tensor, shift_range=0.1):
    """Apply color shifting to test robustness.
    
    Args:
        image_tensor: Tensor [1, 3, H, W] or [3, H, W] in range [-1, 1]
        shift_range: Maximum shift amount in normalized range
        
    Returns:
        shifted_tensor: Tensor with same shape as input
    """
    if image_tensor.dim() == 4:
        batch_size = image_tensor.shape[0]
    else:
        batch_size = 1
        image_tensor = image_tensor.unsqueeze(0)
    
    # Generate random shifts for each channel
    shifts = torch.rand(3, 1, 1, device=image_tensor.device) * 2 * shift_range - shift_range
    
    # Apply shifts
    shifted = image_tensor + shifts
    
    # Clamp to valid range
    shifted = torch.clamp(shifted, -1, 1)
    
    if batch_size == 1:
        return shifted.squeeze(0)
    return shifted


def calculate_psnr(image1, image2):
    """Calculate PSNR between two images.
    
    Args:
        image1: Tensor [3, H, W] or [1, 3, H, W] in range [-1, 1]
        image2: Tensor [3, H, W] or [1, 3, H, W] in range [-1, 1]
        
    Returns:
        psnr: PSNR value in dB
    """
    if image1.dim() == 4:
        image1 = image1.squeeze(0)
    if image2.dim() == 4:
        image2 = image2.squeeze(0)
    
    # Convert to [0, 1]
    img1 = (image1 + 1.0) / 2.0
    img2 = (image2 + 1.0) / 2.0
    
    # Calculate MSE
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr.item()

