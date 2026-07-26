"""Helper script to download sample images for training"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import time


def download_unsplash_sample(output_dir='data/images', num_images=100):
    """Download sample images from Unsplash (free, no API key needed).
    
    Note: This uses Unsplash Source API which doesn't require authentication
    but has rate limits. For production, get an API key.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_images} sample images from Unsplash...")
    print("Note: Using Unsplash Source (no API key). Rate limited.")
    
    base_url = "https://source.unsplash.com/256x256/?nature"
    
    for i in tqdm(range(num_images)):
        try:
            # Add random query to get different images
            queries = ['nature', 'landscape', 'city', 'portrait', 'animal',
                      'architecture', 'food', 'travel', 'abstract', 'water']
            query = queries[i % len(queries)]
            url = f"https://source.unsplash.com/256x256/?{query}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                image_path = output_path / f"unsplash_{i:05d}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Warning: Failed to download image {i}")
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error downloading image {i}: {e}")
            continue
    
    print(f"\nDownloaded images to {output_dir}")
    print(f"Found {len(list(output_path.glob('*.jpg')))} images")


def download_placeholder_images(output_dir='data/images', num_images=100):
    """Generate placeholder images using placeholder.com API."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_images} placeholder images...")
    
    for i in tqdm(range(num_images)):
        try:
            # Use placeholder.com for simple test images
            url = f"https://picsum.photos/256/256?random={i}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                image_path = output_path / f"placeholder_{i:05d}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"Error downloading image {i}: {e}")
            continue
    
    print(f"\nDownloaded images to {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download sample images for training')
    parser.add_argument(
        '--source', type=str, choices=['unsplash', 'placeholder'],
        default='placeholder',
        help='Image source')
    parser.add_argument(
        '--num-images', type=int, default=100,
        help='Number of images to download')
    parser.add_argument(
        '--output-dir', type=str, default='data/images',
        help='Output directory')
    
    args = parser.parse_args()
    
    if args.source == 'unsplash':
        download_unsplash_sample(args.output_dir, args.num_images)
    elif args.source == 'placeholder':
        download_placeholder_images(args.output_dir, args.num_images)
    
    print("\nNext steps:")
    print(f"1. Verify images in {args.output_dir}")
    print(f"2. Train: python train.py --data-dir {args.output_dir}")


if __name__ == '__main__':
    main()

