"""Image processing utilities"""

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


def load_image(image_path, size=(256, 256)):
    """Load and preprocess an image.
    
    Args:
        image_path: Path to image file
        size: Target size (width, height)
        
    Returns:
        image_tensor: Tensor [3, H, W] in range [-1, 1]
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    image_tensor = transform(image)
    return image_tensor


def save_image(image_tensor, output_path):
    """Save an image tensor to file.
    
    Args:
        image_tensor: Tensor [3, H, W] or [1, 3, H, W] in range [-1, 1]
        output_path: Path to save image
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # Denormalize to [0, 1]
    image_tensor = (image_tensor + 1.0) / 2.0
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    image = to_pil(image_tensor)
    
    image.save(output_path)


def prepare_image(image_tensor, device='cpu'):
    """Prepare image tensor for model input.
    
    Args:
        image_tensor: Tensor [3, H, W] or [1, 3, H, W]
        device: Device to move tensor to
        
    Returns:
        batch_tensor: Tensor [1, 3, H, W] on specified device
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    return image_tensor.to(device)


def recover_image(image_tensor):
    """Recover image tensor from model output.
    
    Args:
        image_tensor: Tensor [1, 3, H, W] or [3, H, W]
        
    Returns:
        image_tensor: Tensor [3, H, W]
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    return image_tensor.cpu()


def file_to_tensor(file_path, image_size=(256, 256)):
    """Convert a file to a binary tensor suitable for embedding.
    
    Args:
        file_path: Path to file to embed
        image_size: Size of the image (width, height)
        
    Returns:
        data_tensor: Binary tensor [1, H, W] in range [-1, 1]
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Convert to bits
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    
    # Pad to image_size
    total_pixels = image_size[0] * image_size[1]
    bits = bits[:total_pixels]  # Truncate if too long
    bits.extend([0] * (total_pixels - len(bits)))  # Pad if too short
    
    # Reshape to image
    bit_array = np.array(bits, dtype=np.float32).reshape(image_size[1], image_size[0])
    
    # Convert to tensor: 0 -> -1, 1 -> 1
    data_tensor = torch.from_numpy(bit_array).unsqueeze(0) * 2.0 - 1.0
    
    return data_tensor


def tensor_to_file(data_tensor, output_path, original_size=None):
    """Convert a binary tensor back to a file.
    
    Args:
        data_tensor: Binary tensor [1, H, W] in range [-1, 1]
        output_path: Path to save recovered file
        original_size: Original file size in bytes (if known, for better trimming)
    """
    import numpy as np
    
    # Convert from [-1, 1] to [0, 1]
    data_tensor = (data_tensor + 1.0) / 2.0
    
    # Use a more lenient threshold for low-confidence recovery
    # If values are close to 0.5, the model isn't confident
    # We can try using the raw values with a threshold, or use a sigmoid-like approach
    threshold = 0.5
    
    # Check confidence - if low, we might need to adjust
    if data_tensor.dim() == 3:
        confidence_check = data_tensor.squeeze(0)
    else:
        confidence_check = data_tensor
    mean_confidence = torch.abs(confidence_check - 0.5).mean().item()
    
    # If confidence is very low, the model isn't recovering well
    # Still use threshold, but this will likely produce errors
    if mean_confidence < 0.1:
        # Very low confidence - model outputs are near 0.5 (random)
        # This will produce many bit errors
        pass  # Continue with threshold anyway
    
    # Threshold to binary
    data_tensor = (data_tensor > threshold).float()
    
    # Convert to numpy
    if data_tensor.dim() == 3:
        data_tensor = data_tensor.squeeze(0)
    bit_array = data_tensor.cpu().numpy()
    
    # Flatten and convert to bytes
    bits = bit_array.flatten().astype(np.uint8)
    
    # Convert bits to bytes
    bytes_data = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= (bits[i + j] << (7 - j))
        bytes_data.append(byte)
    
    # Write to file (remove trailing zeros)
    bytes_data = bytes(bytes_data)
    
    # Find last non-zero byte to trim padding
    # If original_size is known, use it; otherwise find last non-zero
    if original_size and original_size < len(bytes_data):
        # Use known original size
        last_nonzero = original_size
    else:
        # Find last non-zero byte
        last_nonzero = len(bytes_data)
        for i in range(len(bytes_data) - 1, -1, -1):
            if bytes_data[i] != 0:
                last_nonzero = i + 1
                break
    
    with open(output_path, 'wb') as f:
        f.write(bytes_data[:last_nonzero])
    
    return mean_confidence

