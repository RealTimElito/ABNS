#!/usr/bin/env python3
"""Extract COCO zip directly to images folder (optionally half)"""

import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Extract COCO zip to images folder')
    parser.add_argument('--zip', type=str, default='data/coco/train2017_images.zip',
                       help='Path to zip file')
    parser.add_argument('--output', type=str, default='data/images',
                       help='Output directory')
    parser.add_argument('--half', action='store_true',
                       help='Extract only half of the images')
    parser.add_argument('--fraction', type=float, default=1.0,
                       help='Fraction of images to extract (0.0-1.0)')
    
    args = parser.parse_args()
    
    zip_path = Path(args.zip)
    target_dir = Path(args.output)
    
    if not zip_path.exists():
        print(f"Error: {zip_path} not found")
        exit(1)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine fraction
    fraction = 0.5 if args.half else args.fraction
    
    print(f"Extracting {zip_path.name} to {target_dir}...")
    print(f"Extracting {fraction*100:.0f}% of images...")
    print("This may take a few minutes...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all JPG files
            members = [m for m in zip_ref.namelist() if m.endswith('.jpg')]
            total = len(members)
            num_to_extract = int(total * fraction)
            members = members[:num_to_extract]
            
            print(f"Found {total} total images")
            print(f"Extracting {len(members)} images ({fraction*100:.0f}%)...")
            
            for member in tqdm(members, desc="Extracting"):
                # Extract just the filename (no subdirectories)
                filename = Path(member).name
                output_file = target_dir / filename
                
                if not output_file.exists():
                    with zip_ref.open(member) as source:
                        with open(output_file, 'wb') as target:
                            target.write(source.read())
        
        print(f"\n✓ Extracted {len(members)} images to {target_dir}")
        print(f"Ready for training!")
        
    except zipfile.BadZipFile:
        print(f"\nError: {zip_path} is not a valid zip file or is incomplete.")
        print("Please re-download the file:")
        print("  python3 scripts/download_coco.py --split train2017 --no-extract")
        exit(1)

if __name__ == '__main__':
    main()

