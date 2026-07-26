"""CNN Encoder for embedding data into images"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder network that embeds secret data into cover images.
    
    Architecture inspired by HiDDeN (High-capacity Deep Data Hiding).
    Embeds 1 bit per pixel into 256x256 images with minimal distortion.
    """
    
    def __init__(self, data_depth=1):
        """Initialize the encoder.
        
        Args:
            data_depth: Number of channels in the secret data (default: 1 for binary)
        """
        super(Encoder, self).__init__()
        
        # First conv block - processes cover image
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Second conv block - processes secret data
        self.conv2 = nn.Sequential(
            nn.Conv2d(data_depth, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Fusion block - combines cover and secret
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Output block - generates steganographic image
        self.output = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
    def forward(self, cover_image, secret_data):
        """Embed secret data into cover image.
        
        Args:
            cover_image: Cover image tensor [B, 3, H, W] in range [-1, 1]
            secret_data: Secret data tensor [B, 1, H, W] in range [-1, 1]
            
        Returns:
            stego_image: Steganographic image [B, 3, H, W] in range [-1, 1]
        """
        # Process cover image
        cover_features = self.conv1(cover_image)
        
        # Process secret data
        secret_features = self.conv2(secret_data)
        
        # Concatenate and fuse
        combined = torch.cat([cover_features, secret_features], dim=1)
        fused = self.fusion(combined)
        
        # Generate stego image (residual approach)
        residual = self.output(fused)
        stego_image = cover_image + residual * 0.1  # Small residual for minimal distortion
        
        # Clamp to valid range
        stego_image = torch.clamp(stego_image, -1, 1)
        
        return stego_image

