"""
RunPod Serverless Handler — FLUX Klein 9B Virtual Try-On LoRA
Model: black-forest-labs/FLUX.2-Klein-9B + fal/flux-klein-9b-virtual-tryon-lora
Ref: https://huggingface.co/fal/flux-klein-9b-virtual-tryon-lora
"""
import os
os.environ.pop("SSL_CERT_FILE", None)

import io
import sys
import time
import base64
import logging
import traceback
import requests as http_requests

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s")
log = logging.getLogger("flux-vton")

import runpod

# ─── Model config ───────────────────────────────────────────
LORA_REPO = "fal/flux-klein-9b-virtual-tryon-lora"
LORA_FILE = "flux-klein-tryon.safetensors"
BASE_MODEL = "black-forest-labs/FLUX.2-Klein-9B"

# ─── Module-level initialization (RunPod best practice) ─────
pipe = None
init_error = None

try:
    import torch
    log.info(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")

    # Try pipeline classes in priority order
    pipeline_cls = None
    try:
        from diffusers import Flux2KleinPipeline
        pipeline_cls = Flux2KleinPipeline
        log.info("Using Flux2KleinPipeline")
    except ImportError:
        log.warning("Flux2KleinPipeline not found, trying FluxPipeline")
        try:
            from diffusers import FluxPipeline
            pipeline_cls = FluxPipeline
            log.info("Using FluxPipeline as fallback")
        except ImportError:
            log.warning("FluxPipeline not found either")
            from diffusers import AutoPipelineForImage2Image
            pipeline_cls = AutoPipelineForImage2Image
            log.info("Using AutoPipelineForImage2Image as fallback")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        log.info("HF_TOKEN found, using for gated model access")

    log.info(f"Loading base model: {BASE_MODEL}")
    t0 = time.time()
    pipe = pipeline_cls.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    ).to("cuda")
    log.info(f"Base model loaded in {time.time()-t0:.1f}s")

    log.info(f"Loading LoRA: {LORA_REPO}/{LORA_FILE}")
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE)
    log.info("LoRA loaded OK")

    try:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
        log.info("torch.compile OK")
    except Exception as e:
        log.warning(f"torch.compile skipped: {e}")

    log.info("=== INIT DONE ===")

except Exception:
    init_error = traceback.format_exc()
    log.error(f"INIT FAILED:\n{init_error}")


# ─── Helper functions ───────────────────────────────────────

def load_image(data: str):
    """Load image from base64 string or URL."""
    from PIL import Image

    if not data:
        return None

    # URL
    if data.startswith("http://") or data.startswith("https://"):
        resp = http_requests.get(data, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    # Base64 (with or without data URI prefix)
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_prompt(inp):
    """Build TRYON prompt from input parameters or use custom prompt."""
    custom = inp.get("prompt")
    if custom:
        if not custom.strip().upper().startswith("TRYON"):
            custom = "TRYON " + custom
        return custom

    category = inp.get("category", "full")
    person_desc = inp.get("person_description", "person standing casually")
    top_desc = inp.get("top_description", "the top garment")
    bottom_desc = inp.get("bottom_description", "the bottom garment")

    if category in ("tops", "top"):
        return f"TRYON {person_desc}. Replace the outfit with {top_desc} as shown in the reference images. The final image is a full body shot."
    elif category in ("bottoms", "bottom"):
        return f"TRYON {person_desc}. Replace the outfit with {bottom_desc} as shown in the reference images. The final image is a full body shot."
    else:
        return f"TRYON {person_desc}. Replace the outfit with {top_desc} and {bottom_desc} as shown in the reference images. The final image is a full body shot."


# ─── Handler ────────────────────────────────────────────────

def handler(job):
    if pipe is None:
        return {"error": f"Pipeline not loaded.\n{init_error or 'unknown'}"}

    inp = job.get("input", job)

    # Support both base64 and URL inputs
    person_data = inp.get("person_image")
    garment_data = inp.get("garment_image")  # backwards compat
    top_data = inp.get("top_image") or garment_data
    bottom_data = inp.get("bottom_image")

    if not person_data:
        return {"error": "Missing person_image (base64 or URL)"}
    if not top_data:
        return {"error": "Missing top_image or garment_image (base64 or URL)"}

    try:
        num_steps = int(inp.get("num_steps", 28))
        guidance_scale = float(inp.get("guidance_scale", 2.5))
        lora_scale = float(inp.get("lora_scale", 1.0))
        width = int(inp.get("width", 768))
        height = int(inp.get("height", 1024))

        person = load_image(person_data)
        top = load_image(top_data)
        bottom = load_image(bottom_data) if bottom_data else None

        prompt = build_prompt(inp)

        images = [person, top]
        if bottom:
            images.append(bottom)

        log.info(f"VTON: steps={num_steps} gs={guidance_scale} lora={lora_scale} "
                 f"images={len(images)} person={person.size} "
                 f"top={top.size} bottom={bottom.size if bottom else 'N/A'}")
        log.info(f"Prompt: {prompt[:120]}...")

        t0 = time.time()

        try:
            result = pipe(
                prompt=prompt,
                image=images[0],
                image_2=images[1],
                image_3=images[2] if len(images) > 2 else None,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )
        except TypeError:
            log.warning("Trying alternative pipeline API...")
            result = pipe(
                prompt=prompt,
                image=images,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )

        output_img = result.images[0]
        elapsed = round(time.time() - t0, 1)
        log.info(f"Done in {elapsed}s | {output_img.size}")

        return {
            "image_base64": encode_image(output_img),
            "elapsed_sec": elapsed,
            "resolution": f"{output_img.size[0]}x{output_img.size[1]}",
        }

    except Exception as e:
        log.error(f"Inference error:\n{traceback.format_exc()}")
        return {"error": str(e)}


# ─── Start RunPod worker ────────────────────────────────────
log.info("Starting FLUX Klein VTON worker...")
runpod.serverless.start({"handler": handler})
