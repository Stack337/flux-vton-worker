"""
RunPod Serverless Handler — FLUX Klein 9B Virtual Try-On LoRA
Model: black-forest-labs/FLUX.2-Klein-9B + fal/flux-klein-9b-virtual-tryon-lora
"""
import os
os.environ.pop("SSL_CERT_FILE", None)

import io
import sys
import time
import base64
import logging
import traceback

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s")
log = logging.getLogger("flux-vton")

import runpod

pipe = None
init_error = None

LORA_REPO = "fal/flux-klein-9b-virtual-tryon-lora"
LORA_FILE = "flux-klein-tryon.safetensors"
BASE_MODEL = "black-forest-labs/FLUX.2-Klein-9B"


def init():
    """Load FLUX Klein 9B + VTON LoRA at container startup."""
    global pipe, init_error
    try:
        import torch
        log.info(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")

        from diffusers import FluxKontextPipeline

        log.info(f"Loading base model: {BASE_MODEL}")
        t0 = time.time()
        pipe = FluxKontextPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        log.info(f"Base model loaded in {time.time()-t0:.1f}s")

        log.info(f"Loading LoRA: {LORA_REPO}/{LORA_FILE}")
        pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE)

        try:
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
            log.info("torch.compile OK")
        except Exception as e:
            log.warning(f"torch.compile skipped: {e}")

        log.info("=== INIT DONE ===")

    except Exception:
        init_error = traceback.format_exc()
        log.error(f"INIT FAILED:\n{init_error}")


def decode_image(data: str):
    from PIL import Image
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_prompt(category="tops"):
    if category in ("tops", "top"):
        return "TRYON person standing. Replace the outfit with the top garment shown in the reference image. The final image is a full body shot."
    elif category in ("bottoms", "bottom"):
        return "TRYON person standing. Replace the outfit with the bottom garment shown in the reference image. The final image is a full body shot."
    else:
        return "TRYON person standing. Replace the outfit with the clothing shown in the reference images. The final image is a full body shot."


def handler(job):
    global pipe, init_error

    if pipe is None:
        return {"error": f"Pipeline not loaded.\n{init_error or 'unknown'}"}

    inp = job.get("input", job)

    person_b64 = inp.get("person_image")
    garment_b64 = inp.get("garment_image")
    if not person_b64 or not garment_b64:
        return {"error": "Missing person_image or garment_image (base64)"}

    try:
        category = inp.get("category", "tops")
        num_steps = inp.get("num_steps", 4)
        guidance_scale = inp.get("guidance_scale", 2.5)

        person = decode_image(person_b64)
        garment = decode_image(garment_b64)
        prompt = build_prompt(category)

        log.info(f"VTON: cat={category} steps={num_steps} person={person.size} garment={garment.size}")
        t0 = time.time()

        result = pipe(
            prompt=prompt,
            image=person,
            image_2=garment,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=1024,
            width=768,
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


log.info("Starting FLUX Klein VTON worker...")
runpod.serverless.start({"handler": handler, "init": init})
