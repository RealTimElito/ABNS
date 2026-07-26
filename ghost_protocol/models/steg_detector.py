"""Steganalysis detector for adversarial testing"""

import torch
import torch.nn as nn


class StegDetector(nn.Module):
    """Steganalysis detector network.
    
    Binary classifier that attempts to detect if an image contains hidden data.
    Used for adversarial testing of the steganographic system.
    """
    
    def __init__(self):
        """Initialize the steg-detector."""
        super(StegDetector, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability of containing hidden data
        )
        
    def forward(self, image):
        """Detect if image contains hidden data.
        
        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1]
            
        Returns:
            probability: Probability that image contains hidden data [B, 1]
        """
        features = self.features(image)
        probability = self.classifier(features)
        return probability

