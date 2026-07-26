#!/bin/bash
# Optimized training script for RTX 5090 (32GB VRAM)

echo "Training with RTX 5090 optimized settings..."
echo ""

python3 train.py \
  --data-dir data/images \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.002 \
  --device cuda \
  --num-workers 16

echo ""
echo "Training complete!"

