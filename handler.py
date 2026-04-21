"""Ultra-minimal handler — NO torch/onnx imports. Just test Python + runpod work."""
import runpod
import os
import sys
import subprocess

def handler(job):
    info = {
        "python": sys.version,
        "platform": sys.platform,
        "cwd": os.getcwd(),
    }
    
    # Check disk
    try:
        import shutil
        t, u, f = shutil.disk_usage("/")
        info["disk"] = f"{f/1e9:.1f}GB free / {t/1e9:.1f}GB total"
    except: pass
    
    # Check if CUDA libs exist
    try:
        r = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=5)
        cuda_libs = [l.strip() for l in r.stdout.split("\n") if "cuda" in l.lower()][:5]
        info["cuda_libs"] = cuda_libs
    except: pass
    
    # Check nvidia-smi
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                          capture_output=True, text=True, timeout=5)
        info["gpu"] = r.stdout.strip()
    except Exception as e:
        info["gpu"] = str(e)
    
    # Try importing torch (just version, don't init CUDA)
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = str(torch.cuda.is_available())
    except Exception as e:
        info["torch_error"] = str(e)
    
    return info

print("Ultra-minimal handler starting...", flush=True)
runpod.serverless.start({"handler": handler})
