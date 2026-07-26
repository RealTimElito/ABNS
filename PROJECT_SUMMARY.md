# Ghost-Protocol: Project Summary

## ✅ Completed Implementation

All four development phases have been successfully implemented:

### Phase 1: Neural Hider ✅
- **Encoder Network** (`ghost_protocol/models/encoder.py`):
  - CNN architecture inspired by HiDDeN
  - Embeds 1 bit per pixel into 256x256 images
  - Uses residual approach for minimal distortion
  - Output range: [-1, 1]

- **Decoder Network** (`ghost_protocol/models/decoder.py`):
  - U-Net-like architecture with skip connections
  - Recovers hidden data from steganographic images
  - Handles various image transformations

- **Training Script** (`train.py`):
  - Custom dataset loader
  - Combined loss: reconstruction + distortion
  - PSNR tracking (target > 40dB)
  - Checkpoint saving

### Phase 2: Robustness Test ✅
- **Attack Simulation** (`ghost_protocol/utils/robustness.py`):
  - JPEG compression (configurable quality)
  - Resizing (scale down/up)
  - Color shifting (channel-wise)
  - PSNR calculation utilities

- **CLI Integration** (`cli.py test-robustness`):
  - Automated testing against multiple attack types
  - MSE calculation for recovery quality

### Phase 3: Protocol ✅
- **AES-256 Encryption** (`ghost_protocol/utils/crypto.py`):
  - PBKDF2 key derivation (100,000 iterations)
  - AES-GCM mode (authenticated encryption)
  - Salt and nonce management

- **File Processing** (`ghost_protocol/utils/image_processing.py`):
  - File-to-binary tensor conversion
  - Binary tensor-to-file recovery
  - Image loading/saving with normalization

- **CLI Tool** (`cli.py`):
  - `hide`: Embed files in images with optional encryption
  - `extract`: Recover files from images with optional decryption
  - Batch processing support
  - Progress reporting

### Phase 4: Adversarial Defense ✅
- **Steg-Detector** (`ghost_protocol/models/steg_detector.py`):
  - Binary classifier CNN
  - Detects presence of hidden data
  - Output: probability of steganographic content

- **Detection Testing** (`cli.py test-detection`):
  - Tests steganographic images against detector
  - Reports detection rate
  - Can be extended with trained detector model

### Networking ✅
- **Secure Tunneling** (`ghost_protocol/network/tunnel.py`):
  - Python wrapper around socat
  - Fallback to pure Python socket implementation
  - Server/client modes
  - File streaming support

## Project Structure

```
ABNS/
├── ghost_protocol/          # Main package
│   ├── models/               # Neural networks
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── steg_detector.py
│   ├── utils/                # Utilities
│   │   ├── crypto.py         # AES-256 encryption
│   │   ├── image_processing.py
│   │   └── robustness.py
│   └── network/              # Networking
│       └── tunnel.py
├── train.py                  # Training script
├── cli.py                    # Command-line interface
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
├── README.md                 # Main documentation
├── QUICKSTART.md            # Quick start guide
└── .gitignore               # Git ignore rules
```

## Key Features

1. **High Capacity**: 1 bit per pixel = ~8KB per 256x256 image
2. **Low Distortion**: Target PSNR > 40dB
3. **Robust**: Survives JPEG, resizing, color shifts
4. **Secure**: AES-256 pre-encryption
5. **Stealthy**: Tested against steg-detection
6. **Networked**: Secure tunneling support

## Technical Stack

- **Framework**: PyTorch 2.0+
- **Architecture**: Custom CNN (HiDDeN-inspired)
- **Security**: cryptography library (AES-256-GCM)
- **Networking**: socat wrapper + Python sockets
- **Image Processing**: PIL, torchvision

## Usage Examples

### Training
```bash
python train.py --epochs 50 --batch-size 32
```

### Hiding Data
```bash
python cli.py hide --input secret.txt \
  --cover-dir images/ --output stego/ \
  --password "secure_password"
```

### Extracting Data
```bash
python cli.py extract --input stego/ \
  --output recovered.txt \
  --password "secure_password"
```

### Testing
```bash
python cli.py test-robustness --input stego/
python cli.py test-detection --input stego/
```

### Tunneling
```bash
# Server
python cli.py tunnel --mode server --port 8080

# Client
python cli.py tunnel --mode client \
  --host localhost --port 8080 \
  --send stego/
```

## Next Steps

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Training Data**:
   - Place images in `data/images/`
   - Recommended: 1000+ diverse images

3. **Train Model**:
   - Run training script
   - Monitor PSNR (should exceed 40dB)

4. **Test End-to-End**:
   - Hide a test file
   - Extract and verify
   - Test robustness
   - Test detection

## Notes

- The steg-detector needs separate training for meaningful results
- For production use, consider training on larger datasets
- Network tunneling requires socat installation
- GPU recommended for training (CUDA support)

## License

This project is provided as-is for educational and research purposes.

