"""
Lightweight CUDA runtime tuning for data-center GPUs (e.g. GCE T4/L4).

Safe no-ops on CPU/MPS or when ``SPOTBALLER_CUDA_TUNING=0``.
"""

from __future__ import annotations

import os


def _env_truthy(name: str, default: str = "1") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v in ("1", "true", "yes", "on")


def apply_cuda_runtime_tuning() -> None:
    """Enable cuDNN autotune and faster matmul on CUDA when available."""
    if not _env_truthy("SPOTBALLER_CUDA_TUNING", "1"):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_inference_device() -> str:
    """Prefer CUDA, then MPS, then CPU (matches PretrainedStack._pick_device)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
