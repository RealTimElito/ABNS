#!/bin/bash
# Maximum performance settings for beast system
# 32 cores, 96GB RAM, RTX 5090

echo "Training with MAXIMUM performance settings..."
echo "System: 32 cores, 96GB RAM, RTX 5090"
echo ""

python3 train.py \
  --data-dir data/images \
  --epochs 100 \
  --batch-size 512 \
  --lr 0.004 \
  --device cuda \
  --num-workers 24

echo ""
echo "Training complete!"

