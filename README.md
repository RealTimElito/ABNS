# Ghost-Protocol: AI-Based Neural Steganography

A secure, end-to-end communication channel where data is hidden within the weights or high-frequency noise of standard media files using Deep Learning.

## Features

- **Neural Steganography**: Hide data in images using CNN-based encoder/decoder
- **AES-256 Encryption**: Pre-encrypt payloads before embedding
- **Robustness**: Survives JPEG compression, resizing, and color shifting
- **Secure Tunneling**: Stream steganographic files over secure channels
- **Adversarial Defense**: Test against steg-detection models

## Installation

```bash
pip install -r requirements.txt
```

## Quick Setup

### Download Training Data

**Option 1: COCO Dataset (Recommended)**
```bash
# Download validation set (~5K images, ~1GB)
python scripts/download_coco.py --split val2017 --prepare-training

# Or download full training set (~330K images, ~20GB)
python scripts/download_coco.py --split train2017 --prepare-training
```

**Option 2: Sample Images (Quick Test)**
```bash
# Download 100 placeholder images for testing
python scripts/download_sample_data.py --source placeholder --num-images 100
```

## Usage

### Training the Model

```bash
python train.py --epochs 50 --batch-size 32
```

### Hiding Data in Images

```bash
python cli.py hide --input secret.txt --cover-dir images/ --output hidden_images/
```

### Extracting Data from Images

```bash
python cli.py extract --input hidden_images/ --output recovered.txt
```

### Secure Tunneling

```bash
# Server (receiver)
python cli.py tunnel --mode server --port 8080

# Client (sender)
python cli.py tunnel --mode client --host localhost --port 8080 --send hidden_images/
```

## Architecture

- **Encoder**: CNN that embeds 1 bit per pixel into 256x256 images
- **Decoder**: CNN that extracts hidden data from steganographic images
- **Target PSNR**: > 40dB (minimal visual distortion)

## Development Phases

1. ✅ Neural Hider - CNN encoder/decoder
2. ✅ Robustness Test - JPEG, resizing, color shifting
3. ✅ Protocol - CLI tool with AES-256 encryption
4. ✅ Adversarial Defense - Steg-detector integration

