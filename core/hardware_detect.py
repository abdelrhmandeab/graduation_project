"""Hardware detection and model recommendation for Jarvis LLM tier selection.

Detects system RAM and GPU availability via Ollama API, then recommends
the best model tier that fits the user's hardware.
"""

import httpx
import psutil
import subprocess

from core.logger import logger

# Model tiers ordered by quality (best first within each hardware bracket).
# Format: (min_ram_gb, gpu_required, tier_label, model_name, num_ctx, lightweight_num_ctx)
_MODEL_TIERS = [
    # 16GB+ RAM, GPU — best quality
    (16, True, "high", "qwen3:8b", 8192, 4096),
    # 12GB+ RAM (GPU or not) — comfortable headroom for qwen3:4b
    (12, False, "medium", "qwen3:4b", 4096, 2048),
    # 8GB RAM, GPU — qwen3:4b runs fast on GPU
    (8, True, "medium", "qwen3:4b", 4096, 2048),
    # 8GB RAM, no GPU — qwen3:1.7b fits comfortably
    (8, False, "low", "qwen3:1.7b", 2048, 1024),
    # <8GB RAM — fall back to the smallest production-grade Qwen3 model
    (0, False, "minimal", "qwen3:0.6b", 1024, 512),
]

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_NUM_CTX = 4096
DEFAULT_LIGHTWEIGHT_NUM_CTX = 2048
DEFAULT_TIER = "medium"


def detect_total_ram_gb():
    """Return total system RAM in GB."""
    try:
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception as exc:
        logger.warning("Failed to detect RAM: %s", exc)
        return 8.0  # conservative fallback


def detect_gpu_available(ollama_base_url="http://localhost:11434"):
    """Check if Ollama has a GPU available by querying running models."""
    try:
        r = httpx.get(f"{ollama_base_url}/api/ps", timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            models = data.get("models", [])
            for m in models:
                # If any model is using a GPU layer, GPU is available
                size_vram = m.get("size_vram", 0)
                if size_vram and size_vram > 0:
                    return True
        # Fallback: check /api/tags for GPU hints isn't reliable,
        # so default to no GPU if we can't confirm.
        return False
    except Exception:
        return False


def _detect_cuda_vram_mb():
    """Best-effort CUDA VRAM detection. Returns 0 when CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return int(getattr(props, "total_memory", 0) / (1024 * 1024))
    except Exception:
        pass

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if completed.returncode == 0:
            first = (completed.stdout or "").strip().splitlines()[0]
            return max(0, int(float(first.strip())))
    except Exception:
        pass
    return 0


def detect_free_ram_mb():
    """Return currently available RAM in MB."""
    try:
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception as exc:
        logger.warning("Failed to detect free RAM: %s", exc)
        return 4096


def recommend_whisper_runtime():
    """Recommend faster-whisper model/device/compute for the current machine."""
    vram_mb = _detect_cuda_vram_mb()
    free_ram_mb = detect_free_ram_mb()
    if vram_mb >= 6000:
        model, device, compute_type = "medium", "cuda", "float16"
    elif vram_mb >= 3000:
        model, device, compute_type = "small", "cuda", "int8_float16"
    elif free_ram_mb >= 8000:
        model, device, compute_type = "small", "cpu", "int8"
    else:
        model, device, compute_type = "base", "cpu", "int8"

    return {
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "vram_mb": vram_mb,
        "free_ram_mb": free_ram_mb,
    }


def recommend_model_tier(ollama_base_url="http://localhost:11434"):
    """Detect hardware and return the recommended model configuration.

    Returns a dict with keys:
        tier: str           - One of "high", "medium", "low", "minimal"
        model: str          - Ollama model name (e.g. "qwen3:4b")
        num_ctx: int        - Recommended context window
        lightweight_num_ctx: int - Lightweight prompt context window
        ram_gb: float       - Detected RAM
        gpu: bool           - Whether GPU was detected
    """
    ram_gb = detect_total_ram_gb()
    gpu = detect_gpu_available(ollama_base_url)

    logger.info(
        "Hardware detection: %.1f GB RAM, GPU=%s",
        ram_gb, "yes" if gpu else "no",
    )

    for min_ram, needs_gpu, tier_label, model, num_ctx, lw_ctx in _MODEL_TIERS:
        if ram_gb >= min_ram and (not needs_gpu or gpu):
            logger.info(
                "Recommended tier=%s model=%s (num_ctx=%d, lightweight=%d)",
                tier_label, model, num_ctx, lw_ctx,
            )
            return {
                "tier": tier_label,
                "model": model,
                "num_ctx": num_ctx,
                "lightweight_num_ctx": lw_ctx,
                "ram_gb": ram_gb,
                "gpu": gpu,
            }

    # Ultimate fallback (should never hit — last tier matches min_ram=0)
    return {
        "tier": DEFAULT_TIER,
        "model": DEFAULT_MODEL,
        "num_ctx": DEFAULT_NUM_CTX,
        "lightweight_num_ctx": DEFAULT_LIGHTWEIGHT_NUM_CTX,
        "ram_gb": ram_gb,
        "gpu": gpu,
    }
