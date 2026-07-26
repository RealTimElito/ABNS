"""CNN Decoder for extracting data from steganographic images"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Decoder network that extracts secret data from steganographic images.
    
    Uses a U-Net-like architecture to recover the embedded data.
    """
    
    def __init__(self, data_depth=1):
        """Initialize the decoder.
        
        Args:
            data_depth: Number of channels in the secret data (default: 1 for binary)
        """
        super(Decoder, self).__init__()
        
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder path
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Output layer (now receives 128 channels: 64 from dec2 + 64 from e1)
        self.output = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, data_depth, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
    def forward(self, stego_image):
        """Extract secret data from steganographic image.
        
        Args:
            stego_image: Steganographic image tensor [B, 3, H, W] in range [-1, 1]
            
        Returns:
            secret_data: Recovered secret data [B, 1, H, W] in range [-1, 1]
        """
        # Encoder path
        e1 = self.enc1(stego_image)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Decoder path with skip connections
        d1 = self.dec1(e3)  # 64x64 -> 128x128
        d1 = torch.cat([d1, e2], dim=1)  # Skip connection with e2 (128x128)
        d2 = self.dec2(d1)  # 128x128 -> 256x256
        d2 = torch.cat([d2, e1], dim=1)  # Skip connection with e1 (256x256)
        
        # Output
        secret_data = self.output(d2)
        
        return secret_data

