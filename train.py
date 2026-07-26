"""Training script for the neural steganography models"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from ghost_protocol.models import Encoder, Decoder
from ghost_protocol.utils.robustness import calculate_psnr


class SteganographyDataset(Dataset):
    """Dataset for training steganography models."""

    def __init__(self, image_dir, image_size=(256, 256), transform=None,
                 recursive=False):
        """Initialize dataset.

        Args:
            image_dir: Directory containing images
            image_size: Target image size
            transform: Optional transform
            recursive: If True, search for images recursively in subdirectories
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load image paths (recursively if requested)
        self.image_paths = []
        image_dir_path = Path(image_dir)

        if recursive:
            # Search recursively in subdirectories
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.image_paths.extend(
                    list(image_dir_path.rglob(ext)))
                self.image_paths.extend(
                    list(image_dir_path.rglob(ext.upper())))
        else:
            # Search only in the specified directory
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.image_paths.extend(
                    list(image_dir_path.glob(ext)))
                self.image_paths.extend(
                    list(image_dir_path.glob(ext.upper())))

        # Remove duplicates and sort
        self.image_paths = sorted(set(self.image_paths))

        if len(self.image_paths) == 0:
            msg = (
                f"No images found in {image_dir}. "
                f"Please ensure the directory contains "
                f"JPG, PNG, or BMP files. "
                f"Use --recursive to search subdirectories.")
            raise ValueError(msg)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load cover image
        image_path = self.image_paths[idx]
        cover_image = Image.open(image_path).convert('RGB')
        cover_image = self.transform(cover_image)

        # Generate random secret data (binary)
        secret_shape = (1, self.image_size[1], self.image_size[0])
        secret_data = torch.randint(0, 2, secret_shape).float()
        # Convert to [-1, 1]
        secret_data = secret_data * 2.0 - 1.0

        return cover_image, secret_data


def train_epoch(encoder, decoder, dataloader, criterion,
                optimizer_enc, optimizer_dec, device):
    """Train for one epoch."""
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_psnr = 0.0
    total_recon = 0.0
    total_binary_acc = 0.0

    for cover_image, secret_data in tqdm(dataloader, desc="Training"):
        cover_image = cover_image.to(device)
        secret_data = secret_data.to(device)

        # Forward pass
        stego_image = encoder(cover_image, secret_data)
        recovered_data = decoder(stego_image)

        # Calculate losses
        # 1. Reconstruction loss (secret data) - MSE
        recon_loss = criterion(recovered_data, secret_data)

        # 2. Binary confidence loss - encourage outputs far from 0 (confident)
        # Convert to [0, 1] range for confidence calculation
        recovered_01 = (recovered_data + 1.0) / 2.0
        secret_01 = (secret_data + 1.0) / 2.0
        # Penalize outputs close to 0.5 (uncertain)
        # For each pixel: if target is 1, output should be >> 0.5; if 0, << 0.5
        confidence_loss = torch.mean(
            torch.abs(recovered_01 - 0.5) * torch.abs(secret_01 - 0.5))
        # Invert: we want HIGH confidence, so penalize LOW confidence
        confidence_loss = 1.0 - confidence_loss

        # 3. Binary accuracy loss - encourage correct binary decisions
        # Threshold at 0.5 and compare
        recovered_binary = (recovered_01 > 0.5).float()
        secret_binary = secret_01
        binary_acc_loss = nn.MSELoss()(recovered_binary, secret_binary)

        # 4. Distortion loss (cover vs stego)
        distortion_loss = nn.MSELoss()(stego_image, cover_image)

        # Combined loss - weight confidence and binary accuracy
        loss = (recon_loss +
                0.5 * confidence_loss +  # Encourage confident outputs
                0.5 * binary_acc_loss +  # Encourage correct binary decisions
                0.1 * distortion_loss)  # Keep image quality

        # Backward pass
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        loss.backward()
        optimizer_enc.step()
        optimizer_dec.step()

        # Metrics
        total_loss += loss.item()
        total_recon += recon_loss.item()
        psnr = calculate_psnr(cover_image, stego_image)
        total_psnr += psnr

        # Calculate binary accuracy
        recovered_01 = (recovered_data + 1.0) / 2.0
        secret_01 = (secret_data + 1.0) / 2.0
        recovered_binary = (recovered_01 > 0.5).float()
        binary_acc = (recovered_binary == secret_01).float().mean().item()
        total_binary_acc += binary_acc

    return {
        'loss': total_loss / len(dataloader),
        'recon_loss': total_recon / len(dataloader),
        'psnr': total_psnr / len(dataloader),
        'binary_acc': total_binary_acc / len(dataloader)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train neural steganography models')
    parser.add_argument(
        '--data-dir', type=str, default='data/images',
        help='Directory containing training images')
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs')
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate')
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument(
        '--device', type=str, default=default_device,
        help='Device to use (cuda/cpu)')
    parser.add_argument(
        '--save-dir', type=str, default='checkpoints',
        help='Directory to save model checkpoints')
    parser.add_argument(
        '--image-size', type=int, default=256,
        help='Image size (assumes square images)')
    parser.add_argument(
        '--recursive', action='store_true',
        help='Search for images recursively in subdirectories')
    parser.add_argument(
        '--num-workers', type=int, default=16,
        help='Number of data loading workers (default: 16)')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Models
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    # Loss and optimizers
    # Use MSE loss since decoder outputs tanh (range [-1, 1])
    criterion = nn.MSELoss()
    optimizer_enc = optim.Adam(encoder.parameters(), lr=args.lr)
    optimizer_dec = optim.Adam(decoder.parameters(), lr=args.lr)

    # Dataset and dataloader
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    # Create dataset (will raise error if no images found)
    image_size_tuple = (args.image_size, args.image_size)
    try:
        dataset = SteganographyDataset(
            args.data_dir, image_size_tuple, recursive=args.recursive)
        print(f"Found {len(dataset)} images in {args.data_dir}")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTips:")
        print("1. Check that the directory path is correct")
        print("2. Use --recursive to search subdirectories")
        print("3. Ensure images are in JPG, PNG, or BMP format")
        return

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_psnr = 0.0

    for epoch in range(args.epochs):
        metrics = train_epoch(
            encoder, decoder, dataloader, criterion,
            optimizer_enc, optimizer_dec, device)

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Recon Loss: {metrics['recon_loss']:.4f}")
        print(f"  Binary Accuracy: {metrics['binary_acc']:.4f} "
              f"({metrics['binary_acc']*100:.2f}%)")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")

        # Save checkpoint
        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            checkpoint_path = os.path.join(
                args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_enc_state_dict': optimizer_enc.state_dict(),
                'optimizer_dec_state_dict': optimizer_dec.state_dict(),
                'psnr': metrics['psnr'],
            }, checkpoint_path)
            print(f"  Saved best model (PSNR: {best_psnr:.2f} dB)")

    print("Training completed!")


if __name__ == '__main__':
    main()
