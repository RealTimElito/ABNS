# What to Do After Training is Complete

## Step 1: Verify Your Model

After training completes, check that your model was saved:

```bash
# Check if model checkpoint exists
ls -lh checkpoints/best_model.pth

# You should see something like:
# -rw-r--r-- 1 user user 15M checkpoints/best_model.pth
```

The model should be saved with:
- Encoder weights
- Decoder weights
- Training metrics (PSNR, epoch, etc.)

## Step 2: Test the Model

### Quick Test: Hide and Extract

```bash
# 1. Create a test file
echo "This is a secret message!" > secret.txt

# 2. Hide it in an image
python cli.py hide \
  --input secret.txt \
  --cover-dir data/images \
  --output output/stego \
  --model checkpoints/best_model.pth

# 3. Extract it back
python cli.py extract \
  --input output/stego \
  --output recovered.txt \
  --model checkpoints/best_model.pth

# 4. Verify it worked
diff secret.txt recovered.txt
# (Should be identical - no output means success!)
```

### Test with Encryption

```bash
# Hide with AES-256 encryption
python cli.py hide \
  --input secret.txt \
  --cover-dir data/images \
  --output output/stego \
  --password "my_secret_password" \
  --model checkpoints/best_model.pth

# Extract with decryption
python cli.py extract \
  --input output/stego \
  --output recovered.txt \
  --password "my_secret_password" \
  --model checkpoints/best_model.pth
```

## Step 3: Test Robustness

See how well your model handles real-world transformations:

```bash
python cli.py test-robustness \
  --input output/stego \
  --model checkpoints/best_model.pth
```

This tests:
- **JPEG compression** (quality 90, 75, 50)
- **Resizing** (scale 0.9, 0.8, 0.7)
- **Color shifting** (shift 0.05, 0.1, 0.15)

Lower MSE = better robustness!

## Step 4: Test Against Steg-Detector

Check if your steganographic images can be detected:

```bash
python cli.py test-detection \
  --input output/stego \
  --model checkpoints/best_model.pth
```

**Note**: The detector needs to be trained separately for meaningful results. This is for testing purposes.

## Step 5: Use in Real Scenarios

### Hide Any File Type

```bash
# Text file
python cli.py hide --input document.txt --cover-dir images/ --output stego/

# Image file
python cli.py hide --input photo.jpg --cover-dir images/ --output stego/

# Binary file
python cli.py hide --input program.exe --cover-dir images/ --output stego/

# Any file works!
```

### Secure Communication

```bash
# Server (receiver)
python cli.py tunnel --mode server --port 8080

# Client (sender) - in another terminal
python cli.py tunnel \
  --mode client \
  --host localhost \
  --port 8080 \
  --send output/stego
```

## Step 6: Evaluate Model Quality

### Check Training Metrics

The model saves the best PSNR achieved. Check it:

```python
import torch

checkpoint = torch.load('checkpoints/best_model.pth')
print(f"Best PSNR: {checkpoint['psnr']:.2f} dB")
print(f"Trained for {checkpoint['epoch']} epochs")

# PSNR > 40dB = Good (invisible hiding)
# PSNR > 35dB = Acceptable
# PSNR < 35dB = Visible artifacts
```

### Visual Inspection

1. Compare cover image vs stego image:
   ```bash
   # They should look identical!
   # Use any image viewer to compare
   ```

2. Check file recovery:
   ```bash
   # Original file size
   ls -lh secret.txt
   
   # Recovered file size (should match)
   ls -lh recovered.txt
   ```

## Step 7: Improve the Model (Optional)

If results aren't good enough:

### Retrain with More Epochs

```bash
python train.py \
  --data-dir data/images \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.0005
```

### Use More Training Data

```bash
# Download more images
python scripts/download_coco.py --split train2017 --prepare-training

# Retrain with larger dataset
python train.py --data-dir data/images --epochs 50
```

### Adjust Learning Rate

```bash
# Lower learning rate for fine-tuning
python train.py --data-dir data/images --lr 0.0001
```

## Common Issues & Solutions

### Issue: Low PSNR (< 35dB)
**Solution**: Train longer or use more data
```bash
python train.py --epochs 100 --data-dir data/images
```

### Issue: Data Recovery Fails
**Solution**: Check if model loaded correctly
```bash
python cli.py extract --input output/stego --output test.txt --model checkpoints/best_model.pth
```

### Issue: Images Look Different
**Solution**: Model may need more training
- Check PSNR (should be > 40dB)
- Retrain with more epochs

## Next Steps Checklist

- [ ] Verify model checkpoint exists
- [ ] Test hide/extract with a simple text file
- [ ] Test with encryption
- [ ] Run robustness tests
- [ ] Test with different file types
- [ ] Try secure tunneling
- [ ] Evaluate PSNR and visual quality

## Example Workflow

```bash
# 1. Training (already done)
python train.py --data-dir data/images --epochs 50

# 2. Create secret file
echo "Top secret message!" > secret.txt

# 3. Hide with encryption
python cli.py hide \
  --input secret.txt \
  --cover-dir data/images \
  --output stego_images \
  --password "secure123"

# 4. Test robustness
python cli.py test-robustness --input stego_images

# 5. Extract (on another machine or after transformations)
python cli.py extract \
  --input stego_images \
  --output recovered.txt \
  --password "secure123"

# 6. Verify
cat recovered.txt
# Should output: "Top secret message!"
```

## Tips for Production Use

1. **Always use encryption** for sensitive data
2. **Test robustness** before sending images
3. **Use multiple images** for larger files (split across images)
4. **Verify extraction** works before relying on it
5. **Keep model checkpoints** backed up

Your model is ready to use! 🎉

