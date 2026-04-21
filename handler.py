"""
Minimal RunPod handler for diagnostics.
If this works, the infra is fine and we add model loading next.
"""
import runpod
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("test")

log.info("=== Handler starting ===")

def handler(job):
    log.info(f"Got job: {job['id']}")
    inp = job.get("input", {})
    return {"status": "ok", "message": "Worker is alive!", "input_keys": list(inp.keys())}

log.info("=== Registering handler ===")
runpod.serverless.start({"handler": handler})
