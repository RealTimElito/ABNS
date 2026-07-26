"""AES-256 encryption utilities"""

import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend


def encrypt_file(input_path, output_path, password):
    """Encrypt a file using AES-256-GCM.
    
    Args:
        input_path: Path to file to encrypt
        output_path: Path to save encrypted file
        password: Password string (will be hashed to 32 bytes for AES-256)
    """
    # Generate key from password using PBKDF2
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())
    
    # Read plaintext
    with open(input_path, 'rb') as f:
        plaintext = f.read()
    
    # Encrypt
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce for GCM
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    
    # Write: salt (16) + nonce (12) + ciphertext
    with open(output_path, 'wb') as f:
        f.write(salt)
        f.write(nonce)
        f.write(ciphertext)


def decrypt_file(input_path, output_path, password):
    """Decrypt a file using AES-256-GCM.
    
    Args:
        input_path: Path to encrypted file
        output_path: Path to save decrypted file
        password: Password string used for encryption
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    
    # Read encrypted file
    with open(input_path, 'rb') as f:
        salt = f.read(16)
        nonce = f.read(12)
        ciphertext = f.read()
    
    # Derive key from password
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())
    
    # Decrypt
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    
    # Write plaintext
    with open(output_path, 'wb') as f:
        f.write(plaintext)

