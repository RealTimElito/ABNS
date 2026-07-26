# Dataset Guide for Ghost-Protocol

## Recommended Datasets

For training neural steganography models, you need a diverse set of natural images. Here are the best options:

### 1. **COCO Dataset** (Recommended)
- **Why**: Large, diverse, real-world images
- **Size**: ~330K images (train2017), ~5K images (val2017)
- **Download**: 
  ```bash
  # Quick start: Download validation set (smaller, ~1GB)
  python scripts/download_coco.py --split val2017 --prepare-training
  
  # Full training set (large, ~20GB)
  python scripts/download_coco.py --split train2017 --prepare-training
  
  # Or download manually from: https://cocodataset.org/
  ```
- **Usage**: Script automatically prepares images for training
- **Pros**: High quality, diverse scenes, commonly used in research
- **Cons**: Large download size (~20GB for full set)

### 2. **ImageNet** (High Quality)
- **Why**: Standard benchmark, high-quality images
- **Size**: ~1.4M images (can use subset)
- **Download**: 
  ```bash
  # Using torchvision
  python -c "from torchvision.datasets import ImageNet; print('Use torchvision.datasets.ImageNet')"
  
  # Or: http://www.image-net.org/
  ```
- **Pros**: Very high quality, diverse categories
- **Cons**: Requires registration, very large

### 3. **DIV2K** (High Resolution)
- **Why**: High-quality images, commonly used for image processing
- **Size**: 800 training + 100 validation images
- **Download**: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- **Pros**: High resolution, clean images
- **Cons**: Smaller dataset size

### 4. **CelebA** (Faces)
- **Why**: Good for face-specific steganography
- **Size**: ~200K face images
- **Download**: 
  ```bash
  # Using torchvision
  python -c "from torchvision.datasets import CelebA; print('Use torchvision.datasets.CelebA')"
  ```
- **Pros**: Consistent format, good for specialized use
- **Cons**: Limited to faces only

### 5. **Places365** (Scenes)
- **Why**: Diverse scene categories
- **Size**: ~1.8M images
- **Download**: http://places2.csail.mit.edu/
- **Pros**: Scene diversity, good for generalization
- **Cons**: Large download

### 6. **Flickr30k / Flickr8k** (Natural Images)
- **Why**: Natural, diverse images
- **Size**: 30K / 8K images
- **Download**: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
- **Pros**: Natural scenes, manageable size
- **Cons**: May require Kaggle account

## Quick Start: Using torchvision Datasets

Create a script to download and prepare data:

```python
# download_dataset.py
from torchvision import datasets
from pathlib import Path
import shutil

def download_coco_subset(output_dir='data/images', num_images=1000):
    """Download a subset of COCO images"""
    # Note: Full COCO requires manual download
    # This is a placeholder - you'd need to implement COCO download
    pass

def download_imagenet_subset(output_dir='data/images'):
    """Download ImageNet validation set (smaller)"""
    # ImageNet requires manual download and setup
    pass

def download_places365_subset(output_dir='data/images'):
    """Download Places365 subset"""
    # Requires manual download
    pass
```

## Minimum Requirements

For effective training, you need:
- **Minimum**: 1,000 images (for basic training)
- **Recommended**: 10,000+ images (for good generalization)
- **Optimal**: 50,000+ images (for production quality)

## Image Characteristics

Good training images should have:
- ✅ **Diverse content**: Landscapes, portraits, objects, scenes
- ✅ **Natural variation**: Different lighting, colors, textures
- ✅ **High quality**: Clear, not heavily compressed
- ✅ **Various sizes**: Model will resize to 256x256 anyway
- ✅ **RGB format**: Color images work best

## Using Your Own Images

You can use any collection of images:

```bash
# Organize your images
mkdir -p data/images

# Copy images to the directory
cp /path/to/your/images/*.jpg data/images/
cp /path/to/your/images/*.png data/images/

# The training script will automatically find them
python train.py --data-dir data/images
```

## Download Script Example

Here's a helper script to download images from common sources:

```python
# scripts/download_images.py
import requests
from pathlib import Path
import json

def download_unsplash_images(query, count=100, output_dir='data/images'):
    """Download images from Unsplash (requires API key)"""
    # Note: Requires Unsplash API key
    # See: https://unsplash.com/developers
    pass

def download_pexels_images(query, count=100, output_dir='data/images'):
    """Download images from Pexels (free, requires API key)"""
    # See: https://www.pexels.com/api/
    pass
```

## Recommended Setup for Beginners

1. **Start Small** (1,000-5,000 images):
   - Use a subset of COCO validation set
   - Or collect images from free sources (Pexels, Unsplash)
   - Or use your own photo collection

2. **For Better Results** (10,000+ images):
   - Download COCO training set
   - Or ImageNet validation set
   - Mix multiple sources

3. **For Production** (50,000+ images):
   - Full COCO dataset
   - Or ImageNet training set
   - Consider data augmentation

## Data Augmentation

The training script automatically resizes images. For better generalization, consider adding augmentation:

```python
# In train.py, you could modify the transform:
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Add augmentation
    transforms.RandomRotation(10),      # Add augmentation
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

## Quick Download Commands

### Using wget (for direct downloads):
```bash
# Example: Download DIV2K (if available)
mkdir -p data/images
cd data/images
# Follow dataset-specific download instructions
```

### Using Python requests:
```python
# Example script to download sample images
import requests
from pathlib import Path

Path('data/images').mkdir(parents=True, exist_ok=True)
# Add your download logic here
```

## My Recommendation

**For getting started quickly:**
1. Use **COCO validation set** (5,000 images) - good balance
2. Or download **1,000-5,000 images** from free stock photo sites
3. Or use your own **photo collection**

**For best results:**
1. **COCO training set** (330K images) - industry standard
2. Mix with **ImageNet** for diversity
3. Add **data augmentation** during training

**For this project specifically:**
- Start with **5,000-10,000 diverse images**
- Focus on **natural scenes** (not synthetic)
- Ensure **good quality** (not heavily compressed)
- Mix **different categories** (people, objects, landscapes)

## Next Steps

1. Choose a dataset based on your needs
2. Download images to `data/images/`
3. Verify images are readable:
   ```bash
   python -c "from PIL import Image; from pathlib import Path; print(f'Found {len(list(Path(\"data/images\").glob(\"*.jpg\")))} JPG images')"
   ```
4. Start training:
   ```bash
   python train.py --data-dir data/images --epochs 50
   ```

