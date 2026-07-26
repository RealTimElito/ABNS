# How Training Works in Ghost-Protocol

## Overview

The training process teaches the neural networks to hide and recover **binary data** in images. It doesn't use text directly during training, but the trained model can hide **any file** (text, images, documents, etc.) later.

## Training Process

### Step 1: Generate Random Binary Data

During training, the system generates **random binary patterns** (not text):

```python
# From train.py, line 79-82
secret_shape = (1, 256, 256)  # 1 channel, 256x256 pixels
secret_data = torch.randint(0, 2, secret_shape).float()  # Random 0s and 1s
secret_data = secret_data * 2.0 - 1.0  # Convert to [-1, 1] range
```

This creates a random binary image (256x256 pixels) where each pixel is either 0 or 1.

### Step 2: Embed Data into Cover Image

The encoder takes:
- **Cover image**: A real photo (e.g., from COCO dataset)
- **Secret data**: The random binary pattern

And produces:
- **Stego image**: The cover image with hidden data embedded (looks almost identical)

```python
stego_image = encoder(cover_image, secret_data)
```

### Step 3: Extract Data from Stego Image

The decoder tries to recover the original secret data:

```python
recovered_data = decoder(stego_image)
```

### Step 4: Calculate Loss

The training optimizes two things:

1. **Reconstruction Loss**: How well the decoder recovers the secret data
   ```python
   recon_loss = MSE(recovered_data, secret_data)
   ```

2. **Distortion Loss**: How much the stego image differs from the cover image
   ```python
   distortion_loss = MSE(stego_image, cover_image)
   ```

3. **Combined Loss**: Balance between hiding data and keeping image quality
   ```python
   total_loss = recon_loss + 0.1 * distortion_loss
   ```

The goal: Hide data perfectly (recon_loss → 0) while keeping the image unchanged (distortion_loss → 0).

## Why Random Binary Data?

During training, we use **random binary patterns** instead of real files because:

1. **Diversity**: Random patterns cover all possible bit combinations
2. **Efficiency**: No need to load/process actual files during training
3. **Generalization**: If it can hide random data, it can hide any file (since files are just binary data)

## How Files Are Hidden (After Training)

Once the model is trained, you can hide **any file** (text, images, documents, etc.):

### Process:

1. **File → Binary**: Convert file to binary bits
   ```python
   # From image_processing.py
   file_bytes = read_file("secret.txt")
   bits = [bit for byte in file_bytes for bit in byte_to_bits(byte)]
   ```

2. **Binary → Tensor**: Reshape bits into 256x256 image format
   ```python
   bit_array = reshape(bits, (256, 256))
   secret_tensor = convert_to_tensor(bit_array)
   ```

3. **Embed**: Use trained encoder to hide in cover image
   ```python
   stego_image = encoder(cover_image, secret_tensor)
   ```

4. **Extract**: Use trained decoder to recover
   ```python
   recovered_tensor = decoder(stego_image)
   ```

5. **Tensor → File**: Convert back to file
   ```python
   bits = tensor_to_bits(recovered_tensor)
   file_bytes = bits_to_bytes(bits)
   write_file("recovered.txt", file_bytes)
   ```

## Training Flow Diagram

```
┌─────────────┐
│ Cover Image │ (Real photo from dataset)
└──────┬──────┘
       │
       ▼
┌─────────────────┐      ┌──────────────────┐
│ Random Binary   │      │  Cover Image     │
│ Pattern (256x256)│      │  (256x256x3)     │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
            ┌───────────────┐
            │   Encoder     │ (CNN)
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │  Stego Image  │ (Looks like cover)
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │   Decoder      │ (CNN)
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ Recovered Data │ (Should match random binary)
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  Calculate     │
            │  Loss & Update │
            └────────────────┘
```

## Key Points

1. **Training**: Uses random binary patterns (not text)
2. **Inference**: Can hide any file (text, images, etc.)
3. **Capacity**: 1 bit per pixel = ~8KB per 256x256 image
4. **Goal**: Perfect recovery + invisible hiding

## Example: Hiding a Text File

```bash
# 1. Train the model (uses random binary data)
python train.py --data-dir data/images --epochs 50

# 2. Hide a text file (converts text → binary → hides in image)
python cli.py hide --input secret.txt --cover-dir images/ --output stego/

# 3. Extract the text file (extracts binary → converts back to text)
python cli.py extract --input stego/ --output recovered.txt
```

The model doesn't "know" it's text - it just sees binary data. That's why it works for any file type!

