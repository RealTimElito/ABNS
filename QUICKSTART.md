# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Basic Usage

### 1. Prepare Training Data

**Option A: Download Sample Images (Quick Start)**
```bash
# Download 100 placeholder images for testing
python scripts/download_sample_data.py --source placeholder --num-images 100

# Or download from Unsplash (slower, but real images)
python scripts/download_sample_data.py --source unsplash --num-images 100
```

**Option B: Use Your Own Images**
```bash
mkdir -p data/images
# Copy your JPG/PNG images to data/images/
cp /path/to/your/images/*.jpg data/images/
```

**Option C: Use Standard Datasets**
- **COCO**: Large, diverse dataset (recommended)
- **ImageNet**: High-quality benchmark dataset
- **DIV2K**: High-resolution images
- See `DATASET_GUIDE.md` for detailed recommendations

**Minimum**: 1,000 images for basic training  
**Recommended**: 10,000+ images for good results

### 2. Train the Model

```bash
python train.py --epochs 50 --batch-size 32 --data-dir data/images
```

This will:
- Train the encoder/decoder networks
- Save the best model to `checkpoints/best_model.pth`
- Target PSNR > 40dB for minimal visual distortion

### 3. Hide Data in Images

```bash
# Without encryption
python cli.py hide --input secret.txt --cover-dir data/images --output output/stego

# With AES-256 encryption
python cli.py hide --input secret.txt --cover-dir data/images --output output/stego --password "your_password"
```

### 4. Extract Data from Images

```bash
# Without decryption
python cli.py extract --input output/stego --output recovered.txt

# With AES-256 decryption
python cli.py extract --input output/stego --output recovered.txt --password "your_password"
```

### 5. Test Robustness

Test how well the hidden data survives attacks:

```bash
python cli.py test-robustness --input output/stego
```

This tests:
- JPEG compression (quality 90, 75, 50)
- Resizing (scale 0.9, 0.8, 0.7)
- Color shifting (shift 0.05, 0.1, 0.15)

### 6. Test Against Steg-Detector

Test if your steganographic images can be detected:

```bash
python cli.py test-detection --input output/stego
```

### 7. Secure Tunneling

**Server (Receiver):**
```bash
python cli.py tunnel --mode server --port 8080
```

**Client (Sender):**
```bash
python cli.py tunnel --mode client --host localhost --port 8080 --send output/stego
```

Note: Requires `socat` to be installed:
- Ubuntu/Debian: `sudo apt-get install socat`
- macOS: `brew install socat`

## Architecture Overview

### Phase 1: Neural Hider ✅
- **Encoder**: CNN that embeds 1 bit per pixel into 256x256 images
- **Decoder**: U-Net-like CNN that extracts hidden data
- **Target**: PSNR > 40dB (minimal visual distortion)

### Phase 2: Robustness Test ✅
- JPEG compression resistance
- Resizing resistance
- Color shifting resistance

### Phase 3: Protocol ✅
- CLI tool for hiding/extracting files
- AES-256 pre-encryption
- File-to-image conversion (1 bit per pixel)

### Phase 4: Adversarial Defense ✅
- Steg-detector model for testing
- Binary classification (contains hidden data or not)

## Project Structure

```
ghost_protocol/
├── models/
│   ├── encoder.py      # CNN encoder for embedding
│   ├── decoder.py      # CNN decoder for extraction
│   └── steg_detector.py # Adversarial detector
├── utils/
│   ├── crypto.py       # AES-256 encryption
│   ├── image_processing.py # Image I/O utilities
│   └── robustness.py   # Attack simulation
└── network/
    └── tunnel.py       # Secure tunneling wrapper

train.py                # Training script
cli.py                  # Command-line interface
```

## Tips

1. **Training**: Use a diverse set of images for better generalization
2. **Capacity**: Current implementation supports ~8KB per 256x256 image
3. **Security**: Always use AES-256 encryption for sensitive data
4. **Robustness**: The model learns to survive common image transformations
5. **Detection**: The steg-detector can be trained separately for adversarial testing

## Troubleshooting

- **CUDA out of memory**: Reduce `--batch-size` in training
- **No images found**: Ensure images are in the specified directory
- **socat not found**: Install socat or use Python-only networking
- **Low PSNR**: Train for more epochs or adjust learning rate

