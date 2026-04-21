"""
RunPod Serverless handler for FASHN VTON 1.5 (virtual try-on).
"""
import runpod
import logging
import sys
import os
import base64
import io
import time
import traceback

# Prevent HF double-caching (symlinks + original)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HOME"] = "/app/hf_cache"

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("vton")

WEIGHTS_DIR = "/app/weights"
pipeline = None
init_error = None


def download_weights():
    """Download model weights from HuggingFace."""
    from huggingface_hub import hf_hub_download

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dwpose_dir = os.path.join(WEIGHTS_DIR, "dwpose")
    os.makedirs(dwpose_dir, exist_ok=True)

    tryon_path = os.path.join(WEIGHTS_DIR, "model.safetensors")
    if not os.path.exists(tryon_path):
        log.info("Downloading TryOnModel weights (~2.4 GB)...")
        hf_hub_download(
            repo_id="fashn-ai/fashn-vton-1.5",
            filename="model.safetensors",
            local_dir=WEIGHTS_DIR,
            local_dir_use_symlinks=False,
        )
        log.info(f"TryOnModel saved: {os.path.getsize(tryon_path) / 1e9:.2f} GB")
    else:
        log.info("TryOnModel cached")

    for fname in ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]:
        fpath = os.path.join(dwpose_dir, fname)
        if not os.path.exists(fpath):
            log.info(f"Downloading DWPose/{fname}...")
            hf_hub_download(
                repo_id="fashn-ai/DWPose",
                filename=fname,
                local_dir=dwpose_dir,
                local_dir_use_symlinks=False,
            )
        else:
            log.info(f"DWPose/{fname} cached")

    log.info("Downloading FashnHumanParser weights...")
    from fashn_human_parser import FashnHumanParser
    _ = FashnHumanParser(device="cpu")

    # Log disk usage
    import shutil
    total, used, free = shutil.disk_usage("/")
    log.info(f"Disk: {used/1e9:.1f}GB used / {total/1e9:.1f}GB total / {free/1e9:.1f}GB free")
    log.info("All weights ready")


def init():
    """Cold start: download weights and initialize pipeline."""
    global pipeline, init_error
    try:
        import torch
        log.info(f"=== VTON Worker Init ===")
        log.info(f"Python {sys.version}")
        log.info(f"PyTorch {torch.__version__}")
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            log.info(f"VRAM: {props.total_memory / 1e9:.1f} GB")

        t0 = time.time()
        download_weights()
        log.info(f"Weights downloaded in {time.time() - t0:.1f}s")

        t1 = time.time()
        from fashn_vton import TryOnPipeline
        pipeline = TryOnPipeline(weights_dir=WEIGHTS_DIR)
        log.info(f"Pipeline loaded in {time.time() - t1:.1f}s")
        log.info("=== Init complete ===")
    except Exception as e:
        init_error = traceback.format_exc()
        log.error(f"=== INIT FAILED ===\n{init_error}")


def handler(job):
    """Process a virtual try-on request."""
    if pipeline is None:
        return {"error": f"Pipeline not initialized", "init_error": init_error}

    from PIL import Image
    inp = job.get("input", {})

    person_b64 = inp.get("person_image")
    garment_b64 = inp.get("garment_image")
    category = inp.get("category", "tops")

    if not person_b64 or not garment_b64:
        return {"error": "person_image and garment_image (base64) required"}
    if category not in ("tops", "bottoms", "one-pieces"):
        return {"error": f"Invalid category '{category}'"}

    try:
        person_img = Image.open(io.BytesIO(base64.b64decode(person_b64))).convert("RGB")
        garment_img = Image.open(io.BytesIO(base64.b64decode(garment_b64))).convert("RGB")
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    t0 = time.time()
    result = pipeline(
        person_image=person_img,
        garment_image=garment_img,
        category=category,
        garment_photo_type=inp.get("garment_photo_type", "model"),
        num_samples=min(inp.get("num_samples", 1), 4),
        num_timesteps=min(inp.get("num_timesteps", 20), 50),
        guidance_scale=inp.get("guidance_scale", 1.5),
        seed=inp.get("seed", 42),
        segmentation_free=inp.get("segmentation_free", True),
    )
    elapsed = time.time() - t0
    log.info(f"Inference: {elapsed:.2f}s")

    images_b64 = []
    for img in result.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    return {"images": images_b64, "count": len(images_b64), "inference_time_s": round(elapsed, 2)}


log.info("Starting VTON worker...")
runpod.serverless.start({"handler": handler, "init": init})
