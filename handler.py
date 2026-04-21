"""
FASHN VTON RunPod Serverless Handler
Downloads weights on cold start, runs inference on requests.
"""
import runpod
import logging
import sys
import os
import base64
import io
import time
import traceback
import shutil

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s")
log = logging.getLogger("vton")

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
pipeline = None
init_error = None


def init():
    """Download weights and initialize the TryOnPipeline."""
    global pipeline, init_error
    try:
        log.info("=== INIT START ===")
        
        # Disk info
        total, used, free = shutil.disk_usage("/")
        log.info(f"Disk: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")
        
        # GPU info
        import torch
        log.info(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")
        
        # Download weights
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        from huggingface_hub import hf_hub_download
        
        log.info("Downloading TryOnModel weights...")
        hf_hub_download(repo_id="fashn-ai/fashn-vton-1.5",
                       filename="model.safetensors",
                       local_dir=WEIGHTS_DIR)
        
        dwpose_dir = os.path.join(WEIGHTS_DIR, "dwpose")
        os.makedirs(dwpose_dir, exist_ok=True)
        for fn in ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]:
            log.info(f"Downloading DWPose/{fn}...")
            hf_hub_download(repo_id="fashn-ai/DWPose", filename=fn,
                          local_dir=dwpose_dir)
        
        # Init human parser (downloads its own weights)
        log.info("Initializing FashnHumanParser...")
        from fashn_human_parser import FashnHumanParser
        _ = FashnHumanParser(device="cpu")
        
        # Init pipeline
        log.info("Loading TryOnPipeline...")
        from fashn_vton import TryOnPipeline
        pipeline = TryOnPipeline(weights_dir=WEIGHTS_DIR)
        
        total2, used2, free2 = shutil.disk_usage("/")
        log.info(f"Disk after init: {free2/1e9:.1f}GB free")
        log.info("=== INIT DONE ===")
        
    except Exception as e:
        init_error = traceback.format_exc()
        log.error(f"INIT FAILED:\n{init_error}")


def handler(job):
    """Process a virtual try-on request."""
    global pipeline, init_error
    
    if pipeline is None:
        return {"error": f"Pipeline not loaded. Init error:\n{init_error or 'unknown'}"}
    
    job_input = job.get("input", {})
    
    person_b64 = job_input.get("person_image")
    garment_b64 = job_input.get("garment_image")
    
    if not person_b64 or not garment_b64:
        return {"error": "Missing person_image or garment_image (base64)"}
    
    try:
        from PIL import Image
        
        person_img = Image.open(io.BytesIO(base64.b64decode(person_b64))).convert("RGB")
        garment_img = Image.open(io.BytesIO(base64.b64decode(garment_b64))).convert("RGB")
        
        log.info(f"Input: person={person_img.size}, garment={garment_img.size}")
        t0 = time.time()
        
        result = pipeline(
            person_image=person_img,
            garment_image=garment_img,
            num_inference_steps=job_input.get("steps", 30),
            guidance_scale=job_input.get("guidance_scale", 2.5),
            seed=job_input.get("seed", 42),
        )
        
        elapsed = time.time() - t0
        log.info(f"Inference done in {elapsed:.1f}s")
        
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode()
        
        return {"image": result_b64, "time": round(elapsed, 1)}
        
    except Exception as e:
        log.error(f"Inference error: {traceback.format_exc()}")
        return {"error": str(e)}


log.info("Starting VTON worker...")
runpod.serverless.start({"handler": handler, "init": init})
