"""Example usage of Ghost-Protocol"""

import os
from pathlib import Path

# Create example directories
Path("data/images").mkdir(parents=True, exist_ok=True)
Path("output").mkdir(parents=True, exist_ok=True)

# Create a sample secret file
secret_content = b"This is a secret message hidden in images!"
with open("secret.txt", "wb") as f:
    f.write(secret_content)

print("Ghost-Protocol Example Usage")
print("=" * 50)
print("\n1. First, you need training images in data/images/")
print("   Place some JPG/PNG images there for training.")
print("\n2. Train the model:")
print("   python train.py --epochs 50 --batch-size 32")
print("\n3. Hide data in images:")
print("   python cli.py hide --input secret.txt --cover-dir data/images --output output/stego")
print("\n4. Extract data from images:")
print("   python cli.py extract --input output/stego --output recovered.txt")
print("\n5. Test robustness:")
print("   python cli.py test-robustness --input output/stego")
print("\n6. Test against steg-detector:")
print("   python cli.py test-detection --input output/stego")
print("\n7. Secure tunneling (requires socat):")
print("   Server: python cli.py tunnel --mode server --port 8080")
print("   Client: python cli.py tunnel --mode client --host localhost --port 8080 --send output/stego")
print("\nNote: For encryption, add --password flag to hide/extract commands")

