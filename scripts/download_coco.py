"""Script to download COCO dataset for training"""

import os
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import json


COCO_URLS = {
    'train2017': {
        'images': 'http://images.cocodataset.org/zips/train2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    },
    'val2017': {
        'images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    },
    'test2017': {
        'images': 'http://images.cocodataset.org/zips/test2017.zip',
        'annotations': None  # Test set has no annotations
    }
}


def download_file(url, output_path, description="Downloading"):
    """Download a file with progress bar and resume support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and get current size for resume
    resume_pos = 0
    if output_path.exists():
        resume_pos = output_path.stat().st_size
        if resume_pos > 0:
            print(f"Resuming download from {resume_pos / 1e9:.2f} GB...")
    
    headers = {}
    if resume_pos > 0:
        headers['Range'] = f'bytes={resume_pos}-'
    
    response = requests.get(url, stream=True, timeout=30, headers=headers)
    total_size = int(response.headers.get('content-length', 0))
    
    # Adjust total size if resuming
    if resume_pos > 0 and 'content-range' in response.headers:
        # Content-Range: bytes 0-1234567/12345678
        content_range = response.headers['content-range']
        total_size = int(content_range.split('/')[-1])
    
    # Open in append mode if resuming
    mode = 'ab' if resume_pos > 0 else 'wb'
    
    with open(output_path, mode) as f, tqdm(
        desc=description,
        total=total_size,
        initial=resume_pos,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path, extract_to, remove_zip=False):
    """Extract a zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total files for progress
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)
    
    if remove_zip:
        zip_path.unlink()
        print(f"Removed {zip_path.name}")


def download_coco_split(split='val2017', output_dir='data/coco', 
                        download_images=True, download_annotations=True,
                        extract=True, remove_zips=False, training_dir=None):
    """Download a COCO dataset split.
    
    Args:
        split: One of 'train2017', 'val2017', 'test2017'
        output_dir: Base output directory
        download_images: Whether to download images
        download_annotations: Whether to download annotations
        extract: Whether to extract zip files
        remove_zips: Whether to remove zip files after extraction
        training_dir: If provided, extract images directly to this directory
    """
    if split not in COCO_URLS:
        raise ValueError(f"Invalid split: {split}. Choose from {list(COCO_URLS.keys())}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading COCO {split}")
    print(f"{'='*60}\n")
    
    # Download images
    if download_images:
        images_url = COCO_URLS[split]['images']
        images_zip = output_path / f"{split}_images.zip"
        
        if images_zip.exists():
            # Check if file is complete by getting expected size
            try:
                head_response = requests.head(images_url, timeout=10)
                expected_size = int(head_response.headers.get('content-length', 0))
                actual_size = images_zip.stat().st_size
                if actual_size < expected_size * 0.99:  # Allow 1% tolerance
                    print(f"Images zip exists but incomplete "
                          f"({actual_size/1e9:.2f}GB / {expected_size/1e9:.2f}GB)")
                    print("Resuming download...")
                    download_file(images_url, images_zip,
                                 f"Resuming {split} images")
                else:
                    print(f"Images zip already exists and appears complete")
            except Exception:
                print(f"Images zip already exists: {images_zip}")
        else:
            print(f"Downloading images from: {images_url}")
            download_file(images_url, images_zip, f"Downloading {split} images")
        
        if extract:
            if training_dir:
                # Extract directly to training directory
                target_dir = Path(training_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Check if already extracted
                existing_images = list(target_dir.glob("*.jpg"))
                if len(existing_images) > 1000:  # Assume extracted if many images
                    print(f"Images already in {target_dir} ({len(existing_images)} found)")
                else:
                    print(f"Extracting images directly to {target_dir}...")
                    with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                        # Get all JPG files from the split
                        members = [m for m in zip_ref.namelist() 
                                 if m.endswith('.jpg')]
                        for member in tqdm(members, desc="Extracting images"):
                            # Extract just the filename (no subdirectories)
                            filename = Path(member).name
                            output_file = target_dir / filename
                            if not output_file.exists():
                                with zip_ref.open(member) as source:
                                    with open(output_file, 'wb') as target:
                                        target.write(source.read())
                    print(f"✓ Extracted {len(members)} images to {target_dir}")
            else:
                # Extract to coco directory (original behavior)
                images_dir = output_path / split
                if not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) < 100:
                    extract_zip(images_zip, output_path, remove_zip=remove_zips)
                else:
                    print(f"Images already extracted to {images_dir}")
    
    # Download annotations (only for train/val, not test)
    if download_annotations and COCO_URLS[split]['annotations']:
        annotations_url = COCO_URLS[split]['annotations']
        annotations_zip = output_path / "annotations.zip"
        
        if annotations_zip.exists():
            print(f"Annotations zip already exists: {annotations_zip}")
        else:
            print(f"Downloading annotations from: {annotations_url}")
            download_file(annotations_url, annotations_zip, 
                         "Downloading annotations")
        
        if extract:
            annotations_dir = output_path / "annotations"
            if not annotations_dir.exists():
                extract_zip(annotations_zip, output_path, remove_zip=remove_zips)
            else:
                print(f"Annotations already extracted to {annotations_dir}")
    
    print(f"\n✓ COCO {split} download complete!")
    print(f"  Images: {output_path / split}")
    if download_annotations:
        print(f"  Annotations: {output_path / 'annotations'}")


def prepare_training_images(coco_dir='data/coco', output_dir='data/images',
                            split='val2017', max_images=None):
    """Copy COCO images to training directory.
    
    Args:
        coco_dir: Directory containing COCO dataset
        output_dir: Directory to copy images to
        split: Which split to use ('train2017', 'val2017', 'test2017')
        max_images: Maximum number of images to copy (None for all)
    """
    coco_path = Path(coco_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find images directory
    images_dir = coco_path / split
    if not images_dir.exists():
        # Try alternative structure
        images_dir = coco_path / split / "images"
    
    if not images_dir.exists():
        raise FileNotFoundError(
            f"COCO images not found at {images_dir}. "
            f"Please download the dataset first.")
    
    # Get all images
    image_files = list(images_dir.glob("*.jpg"))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Copying {len(image_files)} images from {images_dir} to {output_dir}...")
    
    for img_file in tqdm(image_files, desc="Copying images"):
        output_file = output_path / img_file.name
        if not output_file.exists():
            import shutil
            shutil.copy2(img_file, output_file)
    
    print(f"\n✓ Copied {len(image_files)} images to {output_dir}")
    print(f"  Ready for training: python train.py --data-dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Download COCO dataset for training')
    parser.add_argument(
        '--split', type=str, 
        choices=['train2017', 'val2017', 'test2017'],
        default='val2017',
        help='COCO split to download (default: val2017, smaller)')
    parser.add_argument(
        '--output-dir', type=str, default='data/coco',
        help='Output directory for COCO dataset')
    parser.add_argument(
        '--no-images', action='store_true',
        help='Skip downloading images')
    parser.add_argument(
        '--no-annotations', action='store_true',
        help='Skip downloading annotations')
    parser.add_argument(
        '--no-extract', action='store_true',
        help='Do not extract zip files')
    parser.add_argument(
        '--remove-zips', action='store_true',
        help='Remove zip files after extraction')
    parser.add_argument(
        '--prepare-training', action='store_true',
        help='Copy images to data/images/ for training')
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='Maximum number of images to copy (for prepare-training)')
    parser.add_argument(
        '--training-dir', type=str, default='data/images',
        help='Directory for training images (for prepare-training)')
    
    args = parser.parse_args()
    
    # Download COCO
    training_dir_arg = args.training_dir if args.prepare_training else None
    download_coco_split(
        split=args.split,
        output_dir=args.output_dir,
        download_images=not args.no_images,
        download_annotations=not args.no_annotations,
        extract=not args.no_extract,
        remove_zips=args.remove_zips,
        training_dir=training_dir_arg
    )
    
    # Prepare training directory if requested and not already extracted
    if args.prepare_training and not training_dir_arg:
        print("\n" + "="*60)
        print("Preparing training images...")
        print("="*60 + "\n")
        prepare_training_images(
            coco_dir=args.output_dir,
            output_dir=args.training_dir,
            split=args.split,
            max_images=args.max_images
        )


if __name__ == '__main__':
    main()

