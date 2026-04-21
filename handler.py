"""
Diagnostic handler — tests each import separately to find what crashes.
"""
import runpod
import logging
import sys
import traceback

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("diag")

results = {}

# Test 1: torch
try:
    import torch
    results["torch"] = f"OK v{torch.__version__}, CUDA={torch.cuda.is_available()}"
    if torch.cuda.is_available():
        results["gpu"] = torch.cuda.get_device_name(0)
except Exception as e:
    results["torch"] = f"FAIL: {e}"

# Test 2: onnxruntime
try:
    import onnxruntime as ort
    results["onnxruntime"] = f"OK v{ort.__version__}, providers={ort.get_available_providers()}"
except Exception as e:
    results["onnxruntime"] = f"FAIL: {e}"

# Test 3: fashn_vton import
try:
    from fashn_vton import TryOnPipeline
    results["fashn_vton"] = "OK"
except Exception as e:
    results["fashn_vton"] = f"FAIL: {traceback.format_exc()}"

# Test 4: fashn_human_parser
try:
    from fashn_human_parser import FashnHumanParser
    results["fashn_human_parser"] = "OK"
except Exception as e:
    results["fashn_human_parser"] = f"FAIL: {e}"

# Test 5: disk
try:
    import shutil
    total, used, free = shutil.disk_usage("/")
    results["disk"] = f"{free/1e9:.1f}GB free / {total/1e9:.1f}GB total"
except Exception as e:
    results["disk"] = f"FAIL: {e}"

for k, v in results.items():
    log.info(f"{k}: {v}")


def handler(job):
    return results


log.info("Diagnostic handler ready")
runpod.serverless.start({"handler": handler})
