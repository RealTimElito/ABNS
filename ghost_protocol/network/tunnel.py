"""Secure tunneling using socat wrapper"""

import subprocess
import os
import socket
import threading
import time
from pathlib import Path


def create_tunnel(mode='server', host='localhost', port=8080):
    """Create a secure tunnel using socat.
    
    Args:
        mode: 'server' or 'client'
        host: Host address (for client mode)
        port: Port number
        
    Returns:
        process: Subprocess object for the tunnel
    """
    if mode == 'server':
        # Server: listen on port
        cmd = ['socat', '-d', '-d', f'TCP-LISTEN:{port},fork,reuseaddr', 'STDIO']
    else:
        # Client: connect to host:port
        cmd = ['socat', '-d', '-d', f'TCP:{host}:{port}', 'STDIO']
    
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return process
    except FileNotFoundError:
        raise RuntimeError(
            "socat not found. Please install socat:\n"
            "  Ubuntu/Debian: sudo apt-get install socat\n"
            "  macOS: brew install socat\n"
            "  Or use the Python fallback mode"
        )


def send_files(file_paths, host='localhost', port=8080):
    """Send files over a socket connection.
    
    Args:
        file_paths: List of file paths to send
        host: Host address
        port: Port number
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        sock.connect((host, port))
        
        # Send number of files
        num_files = len(file_paths)
        sock.sendall(num_files.to_bytes(4, 'big'))
        
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"Warning: {file_path} does not exist, skipping")
                continue
            
            # Send filename and size
            filename = path.name.encode('utf-8')
            file_size = path.stat().st_size
            
            sock.sendall(len(filename).to_bytes(4, 'big'))
            sock.sendall(filename)
            sock.sendall(file_size.to_bytes(8, 'big'))
            
            # Send file content
            with open(path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    sock.sendall(chunk)
            
            print(f"Sent: {path.name} ({file_size} bytes)")
        
        print("All files sent successfully")
        
    finally:
        sock.close()


def receive_files(output_dir, port=8080):
    """Receive files over a socket connection.
    
    Args:
        output_dir: Directory to save received files
        port: Port number to listen on
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', port))
    sock.listen(1)
    
    print(f"Listening on port {port}...")
    
    try:
        conn, addr = sock.accept()
        print(f"Connection from {addr}")
        
        # Receive number of files
        num_files_bytes = conn.recv(4)
        if len(num_files_bytes) < 4:
            raise RuntimeError("Failed to receive file count")
        num_files = int.from_bytes(num_files_bytes, 'big')
        
        print(f"Receiving {num_files} files...")
        
        for i in range(num_files):
            # Receive filename
            filename_len_bytes = conn.recv(4)
            if len(filename_len_bytes) < 4:
                raise RuntimeError("Failed to receive filename length")
            filename_len = int.from_bytes(filename_len_bytes, 'big')
            
            filename_bytes = b''
            while len(filename_bytes) < filename_len:
                chunk = conn.recv(filename_len - len(filename_bytes))
                if not chunk:
                    raise RuntimeError("Connection closed while receiving filename")
                filename_bytes += chunk
            
            filename = filename_bytes.decode('utf-8')
            
            # Receive file size
            file_size_bytes = conn.recv(8)
            if len(file_size_bytes) < 8:
                raise RuntimeError("Failed to receive file size")
            file_size = int.from_bytes(file_size_bytes, 'big')
            
            # Receive file content
            output_file = output_path / filename
            received = 0
            with open(output_file, 'wb') as f:
                while received < file_size:
                    chunk = conn.recv(min(8192, file_size - received))
                    if not chunk:
                        raise RuntimeError("Connection closed while receiving file")
                    f.write(chunk)
                    received += len(chunk)
            
            print(f"Received: {filename} ({file_size} bytes)")
        
        print("All files received successfully")
        
    finally:
        conn.close()
        sock.close()

