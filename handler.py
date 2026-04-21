"""
RunPod Serverless Worker — FLUX Klein 9B Virtual Try-On LoRA
Pay per second. Auto-scales to zero when idle.
"""
import os
import io
import time
import base64
import logging
import runpod
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("flux-vton")

pipe = None

def load_pipeline():
    global pipe
    from diffusers import FluxKontextPipeline
    
    log.info("Loading FLUX Klein 9B...")
    t0 = time.time()
    
    # Load from cache on network volume or download
    cache_dir = os.environ.get("HF_HOME", "/workspace/hf_cache")
    
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-Klein-9B",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    ).to("cuda")
    
    log.info(f"Base loaded in {time.time()-t0:.0f}s")
    
    # Load VTON LoRA
    pipe.load_lora_weights(
        "fal/flux-klein-9b-virtual-tryon-lora",
        weight_name="flux-klein-tryon.safetensors"
    )
    log.info(f"Pipeline ready | GPU: {torch.cuda.get_device_name(0)}")


def decode_img(data):
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def encode_img(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def handler(job):
    global pipe
    if pipe is None:
        load_pipeline()
    
    inp = job["input"]
    person = decode_img(inp["person_image"])
    garment = decode_img(inp["garment_image"])
    category = inp.get("category", "tops")
    steps = inp.get("num_steps", 4)
    
    prompt = "TRYON person standing naturally. Replace the outfit with the clothing shown in the reference image. The final image is a full body shot."
    
    log.info(f"VTON: {category}, steps={steps}")
    t0 = time.time()
    
    result = pipe(
        prompt=prompt,
        image=person,
        image_2=garment,
        num_inference_steps=steps,
        guidance_scale=2.5,
        height=1024,
        width=768,
    )
    
    img = result.images[0]
    elapsed = round(time.time() - t0, 1)
    log.info(f"Done in {elapsed}s")
    
    return {
        "image_base64": encode_img(img),
        "elapsed_sec": elapsed,
        "resolution": f"{img.size[0]}x{img.size[1]}",
    }


runpod.serverless.start({"handler": handler})
