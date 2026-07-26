"""Command-line interface for Ghost-Protocol"""

import argparse
import os
from pathlib import Path
import getpass

import numpy as np
import torch

from ghost_protocol.models import Encoder, Decoder, StegDetector
from ghost_protocol.utils import (
    encrypt_file, decrypt_file,
    load_image, save_image, prepare_image, recover_image,
    file_to_tensor, tensor_to_file,
    apply_jpeg_compression, apply_resize, apply_color_shift,
    calculate_psnr
)
from ghost_protocol.network import send_files, receive_files


def hide_data(input_file, cover_dir, output_dir, password=None, model_path='checkpoints/best_model.pth', device='cpu'):
    """Hide data in cover images.
    
    Args:
        input_file: Path to file to hide
        cover_dir: Directory containing cover images
        output_dir: Directory to save steganographic images
        password: Password for AES-256 encryption (optional)
        model_path: Path to trained model
        device: Device to use
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load models
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using untrained model")
    
    encoder.eval()
    
    # Encrypt file if password provided
    if password:
        encrypted_file = input_file + '.encrypted'
        encrypt_file(input_file, encrypted_file, password)
        file_to_hide = encrypted_file
        print(f"Encrypted file saved to {encrypted_file}")
    else:
        file_to_hide = input_file
    
    # Convert file to tensor
    secret_tensor = file_to_tensor(file_to_hide)
    
    # Get cover images
    cover_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        cover_paths.extend(list(Path(cover_dir).glob(ext)))
        cover_paths.extend(list(Path(cover_dir).glob(ext.upper())))
    
    if not cover_paths:
        raise ValueError(f"No images found in {cover_dir}")
    
    # Calculate capacity per image (1 bit per pixel = 256*256/8 bytes)
    capacity_per_image = (256 * 256) // 8  # ~8KB per image
    file_size = os.path.getsize(file_to_hide)
    
    # Determine how many images we need
    if file_size > capacity_per_image:
        num_images_needed = (file_size + capacity_per_image - 1) // capacity_per_image
        print(f"File size: {file_size} bytes")
        print(f"Capacity per image: {capacity_per_image} bytes")
        print(f"Need {num_images_needed} images to hide entire file")
        
        if len(cover_paths) < num_images_needed:
            raise ValueError(
                f"Not enough cover images. Need {num_images_needed}, "
                f"but only {len(cover_paths)} found.")
        
        # Use only the number of images needed
        cover_paths = cover_paths[:num_images_needed]
        print(f"Using {len(cover_paths)} cover images...")
    else:
        # File fits in one image - use just the first one
        cover_paths = [cover_paths[0]]
        print(f"File fits in one image. Using: {cover_paths[0].name}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Hide data in image(s)
    if len(cover_paths) == 1:
        # Single image case
        cover_path = cover_paths[0]
        cover_image = load_image(str(cover_path))
        cover_batch = prepare_image(cover_image, device)
        secret_batch = secret_tensor.unsqueeze(0).to(device)
        
        # Embed
        with torch.no_grad():
            stego_image = encoder(cover_batch, secret_batch)
        
        # Save stego image
        output_path = Path(output_dir) / f"stego_{cover_path.name}"
        save_image(recover_image(stego_image), str(output_path))
        
        # Calculate PSNR
        psnr = calculate_psnr(cover_image, recover_image(stego_image))
        print(f"  Saved: {output_path.name} (PSNR: {psnr:.2f} dB)")
        print(f"\nData hidden successfully in 1 image")
    else:
        # Multiple images case - split file across images
        print(f"Hiding data across {len(cover_paths)} images...")
        with open(file_to_hide, 'rb') as f:
            file_data = f.read()
        
        for i, cover_path in enumerate(cover_paths):
            # Get chunk for this image
            start_idx = i * capacity_per_image
            end_idx = min((i + 1) * capacity_per_image, len(file_data))
            chunk = file_data[start_idx:end_idx]
            
            # Save chunk to temp file
            chunk_file = output_dir + f"/chunk_{i}.tmp"
            with open(chunk_file, 'wb') as cf:
                cf.write(chunk)
            
            # Convert chunk to tensor
            chunk_tensor = file_to_tensor(chunk_file)
            os.remove(chunk_file)
            
            # Load cover image
            cover_image = load_image(str(cover_path))
            cover_batch = prepare_image(cover_image, device)
            secret_batch = chunk_tensor.unsqueeze(0).to(device)
            
            # Embed
            with torch.no_grad():
                stego_image = encoder(cover_batch, secret_batch)
            
            # Save stego image
            output_path = Path(output_dir) / f"stego_{i:04d}_{cover_path.name}"
            save_image(recover_image(stego_image), str(output_path))
            
            # Calculate PSNR
            psnr = calculate_psnr(cover_image, recover_image(stego_image))
            print(f"  [{i+1}/{len(cover_paths)}] {cover_path.name} -> {output_path.name} (PSNR: {psnr:.2f} dB)")
        
        print(f"\nData hidden successfully across {len(cover_paths)} images")
        print(f"To extract, use: python cli.py extract --input {output_dir}")
    
    # Clean up encrypted file
    if password and os.path.exists(encrypted_file):
        os.remove(encrypted_file)
    
    print(f"Output directory: {output_dir}")


def extract_data(input_dir, output_file, password=None, model_path='checkpoints/best_model.pth', device='cpu'):
    """Extract data from steganographic images.
    
    Args:
        input_dir: Directory containing steganographic images
        output_file: Path to save extracted file
        password: Password for AES-256 decryption (optional)
        model_path: Path to trained model
        device: Device to use
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load models
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using untrained model")
    
    decoder.eval()
    
    # Get stego images (sort to ensure correct order if split across multiple)
    stego_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        stego_paths.extend(list(Path(input_dir).glob(ext)))
        stego_paths.extend(list(Path(input_dir).glob(ext.upper())))
    
    if not stego_paths:
        raise ValueError(f"No images found in {input_dir}")
    
    # Sort by name to handle multi-image files correctly
    stego_paths = sorted(stego_paths)
    
    # Check if we have multiple stego images (file was split)
    if len(stego_paths) > 1 and any('stego_' in str(p) for p in stego_paths):
        # Multiple images - reconstruct file from chunks
        print(f"Extracting data from {len(stego_paths)} images...")
        all_chunks = []
        
        for i, stego_path in enumerate(stego_paths):
            print(f"  Processing [{i+1}/{len(stego_paths)}] {stego_path.name}...")
            stego_image = load_image(str(stego_path))
            stego_batch = prepare_image(stego_image, device)
            
            # Extract
            with torch.no_grad():
                recovered_tensor = decoder(stego_batch)
            
            # Convert tensor to file chunk
            chunk_file = output_file + f'.chunk_{i}'
            tensor_to_file(recover_image(recovered_tensor), chunk_file)
            
            # Read chunk
            with open(chunk_file, 'rb') as f:
                chunk_data = f.read()
            all_chunks.append(chunk_data)
            os.remove(chunk_file)
        
        # Combine all chunks
        temp_file = output_file + '.temp'
        with open(temp_file, 'wb') as f:
            for chunk in all_chunks:
                f.write(chunk)
        
        print(f"Reconstructed file from {len(stego_paths)} images")
    else:
        # Single image
        print(f"Extracting data from {stego_paths[0].name}...")
        stego_image = load_image(str(stego_paths[0]))
        stego_batch = prepare_image(stego_image, device)
        
        # Extract
        with torch.no_grad():
            recovered_tensor = decoder(stego_batch)
        
        # Check recovery quality (for diagnostics)
        try:
            recovered_np = (recovered_tensor.cpu().squeeze(0).squeeze(0).numpy() + 1.0) / 2.0
            confidence = np.abs(recovered_np - 0.5).mean()
            print(f"  Recovery confidence: {confidence:.4f} (higher is better, >0.3 is good)")
        except NameError:
            # Fallback if numpy import failed
            import numpy as np
            recovered_np = (recovered_tensor.cpu().squeeze(0).squeeze(0).numpy() + 1.0) / 2.0
            confidence = np.abs(recovered_np - 0.5).mean()
            print(f"  Recovery confidence: {confidence:.4f} (higher is better, >0.3 is good)")
        
        # Convert tensor to file
        temp_file = output_file + '.temp'
        tensor_to_file(recover_image(recovered_tensor), temp_file)
        
        # Check file size
        recovered_size = os.path.getsize(temp_file)
        print(f"  Recovered file size: {recovered_size} bytes")
    
    # Handle decryption - try if password provided, otherwise save as-is
    # We can't reliably detect if a file is encrypted without trying to decrypt it
    # So we'll only attempt decryption if a password is provided
    if password:
        # Try to decrypt if password provided
        try:
            from cryptography.exceptions import InvalidTag
            decrypt_file(temp_file, output_file, password)
            os.remove(temp_file)
            print(f"Decrypted file saved to {output_file}")
        except InvalidTag:
            # Decryption failed - file might not be encrypted, or wrong password, or corrupted
            print("\nWarning: Decryption failed. Possible reasons:")
            print("  1. File was not encrypted (no password used when hiding)")
            print("  2. Wrong password")
            print("  3. Data corruption (low recovery confidence detected)")
            
            if confidence < 0.1:
                print(f"\n  Recovery confidence is very low ({confidence:.4f})")
                print("  This suggests the model didn't recover the data well.")
                print("  The file may be corrupted due to poor neural decoder recovery.")
                print("\n  Solutions:")
                print("  - Retrain the model for better binary recovery")
                print("  - Try hiding/extracting without encryption first")
                print("  - Check if the password is correct")
            
            # Ask user what to do
            print("\nSaving file as-is (without decryption)...")
            print("If the file is corrupted, it won't be usable.")
            os.rename(temp_file, output_file)
            print(f"Extracted file saved to {output_file} (may be corrupted)")
        except Exception as e:
            # Other decryption errors
            print(f"\nError during decryption: {e}")
            print("Saving file as-is...")
            os.rename(temp_file, output_file)
            print(f"Extracted file saved to {output_file}")
    else:
        # No password provided - save as-is
        os.rename(temp_file, output_file)
        print(f"Extracted file saved to {output_file}")
    
    print("Data extraction completed!")


def test_robustness(input_dir, model_path='checkpoints/best_model.pth', device='cpu'):
    """Test robustness against various attacks.
    
    Args:
        input_dir: Directory containing steganographic images
        model_path: Path to trained model
        device: Device to use
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load models
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    else:
        print(f"Warning: Model not found at {model_path}")
        return
    
    decoder.eval()
    
    # Get stego images
    stego_paths = list(Path(input_dir).glob('*.jpg')) + list(Path(input_dir).glob('*.png'))
    if not stego_paths:
        print(f"No images found in {input_dir}")
        return
    
    stego_image = load_image(str(stego_paths[0]))
    stego_batch = prepare_image(stego_image, device)
    
    # Original extraction
    with torch.no_grad():
        original_recovered = decoder(stego_batch)
    
    print("Testing robustness...")
    print("\n1. JPEG Compression:")
    for quality in [90, 75, 50]:
        compressed = apply_jpeg_compression(stego_batch, quality=quality)
        with torch.no_grad():
            recovered = decoder(compressed.to(device))
        mse = torch.mean((original_recovered - recovered) ** 2).item()
        print(f"   Quality {quality}: MSE = {mse:.6f}")
    
    print("\n2. Resizing:")
    for scale in [0.9, 0.8, 0.7]:
        resized = apply_resize(stego_batch, scale_factor=scale)
        with torch.no_grad():
            recovered = decoder(resized.to(device))
        mse = torch.mean((original_recovered - recovered) ** 2).item()
        print(f"   Scale {scale}: MSE = {mse:.6f}")
    
    print("\n3. Color Shifting:")
    for shift in [0.05, 0.1, 0.15]:
        shifted = apply_color_shift(stego_batch, shift_range=shift)
        with torch.no_grad():
            recovered = decoder(shifted.to(device))
        mse = torch.mean((original_recovered - recovered) ** 2).item()
        print(f"   Shift {shift}: MSE = {mse:.6f}")


def test_detection(input_dir, model_path='checkpoints/best_model.pth', device='cpu'):
    """Test against steg-detector.
    
    Args:
        input_dir: Directory containing steganographic images
        model_path: Path to trained model
        device: Device to use
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load steg-detector
    detector = StegDetector().to(device)
    detector.eval()
    
    # Get images
    image_paths = list(Path(input_dir).glob('*.jpg')) + list(Path(input_dir).glob('*.png'))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print("Testing against steg-detector...")
    print("(Note: Detector needs to be trained separately)")
    
    detected_count = 0
    for img_path in image_paths[:10]:  # Test first 10 images
        image = load_image(str(img_path))
        image_batch = prepare_image(image, device)
        
        with torch.no_grad():
            probability = detector(image_batch)
        
        if probability.item() > 0.5:
            detected_count += 1
        print(f"  {img_path.name}: {probability.item():.4f} (detected: {probability.item() > 0.5})")
    
    print(f"\nDetection rate: {detected_count}/{min(10, len(image_paths))}")


def main():
    parser = argparse.ArgumentParser(description='Ghost-Protocol CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Hide command
    hide_parser = subparsers.add_parser('hide', help='Hide data in images')
    hide_parser.add_argument('--input', type=str, required=True, help='File to hide')
    hide_parser.add_argument('--cover-dir', type=str, required=True, help='Directory with cover images')
    hide_parser.add_argument('--output', type=str, required=True, help='Output directory for stego images')
    hide_parser.add_argument('--password', type=str, help='Password for encryption (optional)')
    hide_parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='Model path')
    hide_parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract data from images')
    extract_parser.add_argument('--input', type=str, required=True, help='Directory with stego images')
    extract_parser.add_argument('--output', type=str, required=True, help='Output file path')
    extract_parser.add_argument('--password', type=str, help='Password for decryption (optional)')
    extract_parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='Model path')
    extract_parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    # Robustness test command
    robust_parser = subparsers.add_parser('test-robustness', help='Test robustness')
    robust_parser.add_argument('--input', type=str, required=True, help='Directory with stego images')
    robust_parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='Model path')
    robust_parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    # Detection test command
    detect_parser = subparsers.add_parser('test-detection', help='Test against steg-detector')
    detect_parser.add_argument('--input', type=str, required=True, help='Directory with stego images')
    detect_parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='Model path')
    detect_parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    # Tunnel command
    tunnel_parser = subparsers.add_parser('tunnel', help='Secure tunneling')
    tunnel_parser.add_argument('--mode', type=str, choices=['server', 'client'], required=True, help='Server or client mode')
    tunnel_parser.add_argument('--host', type=str, default='localhost', help='Host (for client mode)')
    tunnel_parser.add_argument('--port', type=int, default=8080, help='Port number')
    tunnel_parser.add_argument('--send', type=str, help='Directory to send (client mode)')
    
    args = parser.parse_args()
    
    if args.command == 'hide':
        device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
        password = args.password or (getpass.getpass("Enter encryption password (or press Enter to skip): ") or None)
        hide_data(args.input, args.cover_dir, args.output, password, args.model, device)
    
    elif args.command == 'extract':
        device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
        password = args.password or (getpass.getpass("Enter decryption password (or press Enter to skip): ") or None)
        extract_data(args.input, args.output, password, args.model, device)
    
    elif args.command == 'test-robustness':
        device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
        test_robustness(args.input, args.model, device)
    
    elif args.command == 'test-detection':
        device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
        test_detection(args.input, args.model, device)
    
    elif args.command == 'tunnel':
        if args.mode == 'server':
            print(f"Starting server on port {args.port}...")
            receive_files('received', args.port)
        else:
            if not args.send:
                print("Error: --send directory required for client mode")
                return
            print(f"Connecting to {args.host}:{args.port}...")
            send_files(list(Path(args.send).glob('*')), args.host, args.port)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

