# Troubleshooting Guide

## Problem: Recovered File is Corrupted

### Symptoms
- Files differ when comparing original vs recovered
- Recovery confidence is very low (< 0.1)
- Recovered file size is wrong (e.g., 8KB instead of 26 bytes)

### Root Cause
The neural decoder isn't recovering binary data accurately. The decoder outputs values very close to 0.5 (the threshold), making bit decisions unreliable.

### Solutions

#### 1. Retrain the Model (Recommended)

The model needs better training for binary recovery:

```bash
# Train for more epochs with better loss function
python train.py --data-dir data/images --epochs 100 --batch-size 32
```

**Why**: More training helps the decoder learn to output values closer to -1 or +1 (confident), rather than near 0 (uncertain).

#### 2. Improve Training Loss

The current loss uses MSE, which might not be optimal for binary data. Consider:

- **Binary Cross-Entropy Loss**: Better for binary classification
- **Focal Loss**: Helps with hard examples
- **Combined Loss**: MSE + BCE for both reconstruction and binary accuracy

#### 3. Check Model Architecture

The decoder might need:
- More capacity (deeper/wider)
- Better skip connections
- Different activation functions

#### 4. Test Without Encryption First

To isolate the issue:

```bash
# Hide without encryption
python cli.py hide --input secret.txt --cover-dir data/images --output test1

# Extract
python cli.py extract --input test1 --output recovered.txt

# Check if it works
diff secret.txt recovered.txt
```

If this works, the issue is with encrypted file recovery (bit errors in encrypted data break decryption).

#### 5. Use Smaller Files

The model might work better with smaller files:

```bash
# Test with a very small file (few bytes)
echo "test" > small.txt
python cli.py hide --input small.txt --cover-dir data/images --output test2
python cli.py extract --input test2 --output recovered_small.txt
```

#### 6. Check Training Metrics

During training, monitor:
- **Reconstruction Loss**: Should decrease
- **PSNR**: Should be > 40dB
- **Binary Accuracy**: Should be > 95%

If these aren't improving, the model architecture or training might need adjustment.

## Expected Recovery Confidence

- **> 0.3**: Good recovery (should work)
- **0.1 - 0.3**: Marginal (may have errors)
- **< 0.1**: Poor recovery (will have many errors) ⚠️

Your current confidence: **0.0096** - This is very poor and explains the corruption.

## Quick Fix: Retrain

```bash
# Retrain with more epochs
python train.py \
  --data-dir data/images \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.0005

# Then test again
python cli.py hide --input secret.txt --cover-dir data/images --output test_new
python cli.py extract --input test_new --output recovered_new.txt
diff secret.txt recovered_new.txt
```

## Why This Happens

Neural steganography is challenging because:
1. The decoder must perfectly recover binary data (any bit error corrupts the file)
2. The model outputs continuous values that must be thresholded to binary
3. If outputs are near 0.5, thresholding becomes unreliable

This is why traditional steganography (LSB) is simpler but less robust - it directly modifies bits, while neural methods must learn to encode/decode through continuous values.

## Next Steps

1. **Retrain the model** with more epochs
2. **Monitor training metrics** - ensure binary accuracy improves
3. **Test with small files** first
4. **Consider improving the loss function** for binary data

The model needs to learn to output confident values (close to -1 or +1) rather than uncertain values (near 0).

