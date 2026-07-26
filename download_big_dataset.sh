#!/bin/bash
# Script to download the full COCO training dataset

echo "=========================================="
echo "Downloading Full COCO Training Dataset"
echo "=========================================="
echo ""
echo "This will download:"
echo "  - ~330,000 training images (~18GB)"
echo "  - Annotations (~250MB)"
echo "  - Total: ~20GB"
echo ""
echo "This may take a while depending on your connection..."
echo ""

# Download full training set and extract directly to images folder
python3 scripts/download_coco.py \
  --split train2017 \
  --output-dir data/coco \
  --prepare-training \
  --training-dir data/images \
  --remove-zips

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Images are now in: data/images/"
echo "You can start training with:"
echo "  python train.py --data-dir data/images --epochs 100"
echo ""

