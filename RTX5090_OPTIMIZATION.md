# RTX 5090 Optimization Guide

## Your GPU Specs
- **GPU**: NVIDIA GeForce RTX 5090
- **VRAM**: 33.7 GB
- **Capability**: Can handle MUCH larger batch sizes!

## Recommended Batch Sizes

### Conservative (Safe)
```bash
--batch-size 128
```
- Uses ~8-10GB VRAM
- Fast training, stable
- Good starting point

### Optimal (Recommended)
```bash
--batch-size 256
```
- Uses ~15-18GB VRAM
- Much faster training
- Better gradient estimates
- **Recommended for your GPU**

### Aggressive (Maximum)
```bash
--batch-size 512
```
- Uses ~25-30GB VRAM
- Fastest training
- May need to reduce if you get OOM errors
- Test first!

## Learning Rate Adjustment

With larger batch sizes, you can use higher learning rates:

- **Batch 32**: LR = 0.001 (baseline)
- **Batch 128**: LR = 0.002 (2x)
- **Batch 256**: LR = 0.002-0.003 (2-3x)
- **Batch 512**: LR = 0.004 (4x)

**Rule of thumb**: Scale LR proportionally with batch size.

## Training Commands

### Recommended (Batch 256)
```bash
python train.py \
  --data-dir data/images \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.002 \
  --device cuda
```

### Maximum Performance (Batch 512)
```bash
python train.py \
  --data-dir data/images \
  --epochs 100 \
  --batch-size 512 \
  --lr 0.004 \
  --device cuda
```

### If You Get Out of Memory
```bash
# Reduce batch size
--batch-size 128

# Or use gradient accumulation (simulates larger batch)
# (Would need to modify train.py for this)
```

## Expected Performance

### Batch Size 32 (Old)
- Time per epoch: ~X minutes
- VRAM usage: ~2-3GB
- Underutilized GPU

### Batch Size 256 (Recommended)
- Time per epoch: ~X/8 minutes (8x faster!)
- VRAM usage: ~15-18GB
- Much better GPU utilization

### Batch Size 512 (Maximum)
- Time per epoch: ~X/16 minutes (16x faster!)
- VRAM usage: ~25-30GB
- Maximum GPU utilization

## Additional Optimizations

### Mixed Precision Training (Optional)
You could enable automatic mixed precision for even larger batches:
```python
from torch.cuda.amp import autocast, GradScaler
# Would need to modify train.py
```

### DataLoader Workers
With 32 CPU cores, you can use many workers:
```python
num_workers=16  # Optimal for 32 cores (2 per core)
num_workers=24  # Aggressive (3 per core)
num_workers=32  # Maximum (1 per core, may have overhead)
```

**Rule of thumb**: 2-4 workers per CPU core. With 32 cores:
- **16 workers**: Optimal (2 per core) ✅ Recommended
- **24 workers**: Aggressive (3 per core) ✅ Good for maximum throughput
- **32 workers**: Maximum (may have overhead)

## Memory Calculation

For 256x256 images:
- Image size: 3 × 256 × 256 × 4 bytes = ~786KB per image
- With activations/gradients: ~2-4MB per image
- Batch 256: ~512MB - 1GB for images
- Plus model weights, activations, gradients: ~15-18GB total

Your 33.7GB VRAM can easily handle batch 256, maybe even 512!

## Quick Start

After your COCO download completes:

```bash
# Use the optimized script
./train_5090.sh

# Or manually
python train.py --data-dir data/images --epochs 100 --batch-size 256 --lr 0.002
```

## Monitoring

Watch GPU utilization:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

You should see:
- **Batch 32**: ~10-20% GPU utilization
- **Batch 256**: ~80-95% GPU utilization ✅
- **Batch 512**: ~95-100% GPU utilization ✅

Enjoy your beast GPU! 🚀

