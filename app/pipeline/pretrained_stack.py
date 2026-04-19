"""
Optional Hugging Face–backed components aligned with `app/ml/model_shortlist.py`:
- SigLIP: team clustering via image embeddings; zero-shot action hints vs text prompts.
- TrOCR: jersey digit reading (PARSeq-style STR substitute via a strong printed-text OCR).
- VideoMAE: optional short-clip logits (Kinetics head) for auxiliary motion signal.

Requires: transformers, torch (already in requirements). Falls back cleanly if unavailable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import (
        SiglipModel,
        SiglipProcessor,
        TrOCRProcessor,
        VideoMAEForVideoClassification,
        VideoMAEImageProcessor,
        VisionEncoderDecoderModel,
    )
except Exception:  # pragma: no cover
    SiglipModel = None  # type: ignore
    SiglipProcessor = None  # type: ignore
    VisionEncoderDecoderModel = None  # type: ignore
    TrOCRProcessor = None  # type: ignore
    VideoMAEForVideoClassification = None  # type: ignore
    VideoMAEImageProcessor = None  # type: ignore

SIGLIP_ID = "google/siglip-base-patch16-224"
TROCR_ID = "microsoft/trocr-base-printed"
VIDEOMAE_ID = "MCG-NJU/videomae-base"

ACTION_PROMPTS = [
    "a basketball player shooting the ball toward the hoop",
    "a basketball player passing the ball to a teammate",
    "a basketball player dribbling the basketball",
    "a basketball player defending another player",
]


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _ensure_hf_no_proxy_for_hub() -> None:
    """
    If HTTP(S) proxy env vars are set, route huggingface.co around the proxy.

    Many environments return ``403 Forbidden`` on CONNECT to the Hub through
    a corporate proxy; direct TLS to ``huggingface.co`` works. Opt out with
    ``SPOTBALLER_HF_RESPECT_PROXY=1`` if you must send Hub traffic via proxy.
    """
    if _env_truthy("SPOTBALLER_HF_RESPECT_PROXY"):
        return
    proxy_set = any(
        os.environ.get(k)
        for k in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        )
    )
    if not proxy_set:
        return
    extra_hosts = "huggingface.co,.hf.co,hf.co"
    for key in ("NO_PROXY", "no_proxy"):
        cur = (os.environ.get(key) or "").strip()
        if "huggingface.co" in cur:
            continue
        os.environ[key] = f"{cur},{extra_hosts}" if cur else extra_hosts


def _apply_hf_hub_env_defaults() -> None:
    """Honor SPOTBALLER_HF_OFFLINE so huggingface_hub does not contact the API."""
    if _env_truthy("SPOTBALLER_HF_OFFLINE"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _hf_config_cached_locally(repo_id: str) -> bool:
    """True if ``config.json`` for ``repo_id`` is already in the HF cache (no Hub HEAD/GET)."""
    if _env_truthy("SPOTBALLER_HF_DISABLE_AUTO_OFFLINE"):
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    try:
        p = try_to_load_from_cache(repo_id, "config.json")
        return p is not None and os.path.isfile(p)
    except Exception:
        return False


def hf_from_pretrained_kwargs(repo_id: str | None = None) -> dict[str, Any]:
    """
    Extra arguments for ``*.from_pretrained(...)``.

    Even when weights are already in ``~/.cache/huggingface/hub``, the default
    ``from_pretrained(repo_id)`` path can still **call the Hub** (version checks,
    metadata), which triggers rate-limit warnings without ``HF_TOKEN``. Set
    ``HF_HUB_OFFLINE=1`` or ``SPOTBALLER_HF_LOCAL_ONLY=1`` after the cache is
    populated on the machine, or set ``HF_TOKEN`` / ``huggingface-cli login``.

    Pass ``repo_id`` so when ``config.json`` is already cached locally, we set
    ``local_files_only=True`` and avoid Hub requests (helps broken proxies).
    """
    kw: dict[str, Any] = {}
    if (
        _env_truthy("HF_HUB_OFFLINE")
        or _env_truthy("TRANSFORMERS_OFFLINE")
        or _env_truthy("SPOTBALLER_HF_LOCAL_ONLY")
    ):
        kw["local_files_only"] = True
    elif repo_id and _hf_config_cached_locally(repo_id):
        kw["local_files_only"] = True
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        kw["token"] = token
    return kw


_ensure_hf_no_proxy_for_hub()
_apply_hf_hub_env_defaults()


@dataclass
class PipelineConfig:
    """Toggle each pretrained branch; all default on except VideoMAE (VRAM / CPU heavy)."""

    use_siglip_teams: bool = True
    use_trocr_jersey: bool = True
    use_siglip_action_hints: bool = True
    use_videomae_clips: bool = False
    device: str | None = None


@dataclass
class PretrainedStack:
    """Lazily loaded models; safe to construct even when transformers is missing."""

    config: PipelineConfig = field(default_factory=PipelineConfig)
    _siglip_processor: Any = None
    _siglip_model: Any = None
    _trocr_processor: Any = None
    _trocr_model: Any = None
    _videomae_processor: Any = None
    _videomae_model: Any = None
    load_errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if torch is None:
            self.load_errors.append("torch unavailable")
            return
        if SiglipModel is None:
            self.load_errors.append("transformers unavailable")
            return
        dev = self._pick_device()
        if self.config.use_siglip_teams or self.config.use_siglip_action_hints:
            self._init_siglip(dev)
        if self.config.use_trocr_jersey:
            self._init_trocr(dev)
        if self.config.use_videomae_clips:
            self._init_videomae(dev)

    def _pick_device(self) -> str:
        if self.config.device:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _init_siglip(self, device: str) -> None:
        try:
            _kw = hf_from_pretrained_kwargs(SIGLIP_ID)
            self._siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_ID, **_kw)
            self._siglip_model = SiglipModel.from_pretrained(SIGLIP_ID, **_kw).to(device)
            self._siglip_model.eval()
            self._siglip_device = device
        except Exception as exc:  # pragma: no cover
            self.load_errors.append(f"siglip: {exc}")
            self._siglip_processor = None
            self._siglip_model = None

    def _init_trocr(self, device: str) -> None:
        try:
            _kw = hf_from_pretrained_kwargs(TROCR_ID)
            self._trocr_processor = TrOCRProcessor.from_pretrained(TROCR_ID, **_kw)
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_ID, **_kw).to(device)
            self._trocr_model.eval()
            self._trocr_device = device
        except Exception as exc:  # pragma: no cover
            self.load_errors.append(f"trocr: {exc}")
            self._trocr_processor = None
            self._trocr_model = None

    def _init_videomae(self, device: str) -> None:
        try:
            _kw = hf_from_pretrained_kwargs(VIDEOMAE_ID)
            self._videomae_processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_ID, **_kw)
            self._videomae_model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_ID, **_kw).to(device)
            self._videomae_model.eval()
            self._videomae_device = device
        except Exception as exc:  # pragma: no cover
            self.load_errors.append(f"videomae: {exc}")
            self._videomae_processor = None
            self._videomae_model = None

    @property
    def has_siglip(self) -> bool:
        return self._siglip_model is not None and self._siglip_processor is not None

    @property
    def has_trocr(self) -> bool:
        return self._trocr_model is not None and self._trocr_processor is not None

    @property
    def has_videomae(self) -> bool:
        return self._videomae_model is not None and self._videomae_processor is not None

    def embed_jersey_region(self, bgr_crop: np.ndarray) -> np.ndarray | None:
        """L2-normalized SigLIP image embedding for team clustering."""
        if not self.has_siglip or not self.config.use_siglip_teams:
            return None
        pil = self._bgr_to_pil(bgr_crop)
        if pil is None:
            return None
        device = getattr(self, "_siglip_device", "cpu")
        with torch.inference_mode():
            inputs = self._siglip_processor(images=pil, return_tensors="pt").to(device)
            image_feats = self._siglip_model.get_image_features(**inputs)
            vec = image_feats[0].detach().float().cpu().numpy()
            n = np.linalg.norm(vec) + 1e-8
            return (vec / n).astype(np.float32)

    def recognize_jersey_digits(self, bgr_crop: np.ndarray) -> tuple[str | None, float]:
        """TrOCR digit read; returns (digits, confidence heuristic)."""
        if not self.has_trocr or not self.config.use_trocr_jersey:
            return None, 0.0
        pil = self._bgr_to_pil(bgr_crop)
        if pil is None:
            return None, 0.0
        device = getattr(self, "_trocr_device", "cpu")
        try:
            with torch.inference_mode():
                pixel_values = self._trocr_processor(images=pil, return_tensors="pt").pixel_values.to(device)
                # Avoid HF UserWarning about legacy max_length default; jersey text is short.
                generated_ids = self._trocr_model.generate(
                    pixel_values,
                    max_new_tokens=8,
                )
                text = self._trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            return None, 0.0
        match = re.search(r"\d{1,2}", text or "")
        if not match:
            return None, 0.0
        return match.group(0), 0.75

    def action_hint_scores(self, bgr_frame_region: np.ndarray) -> dict[str, float] | None:
        """SigLIP image–text similarity scores for action-style prompts (VideoMAE alternative)."""
        if not self.has_siglip or not self.config.use_siglip_action_hints:
            return None
        pil = self._bgr_to_pil(bgr_frame_region)
        if pil is None:
            return None
        device = getattr(self, "_siglip_device", "cpu")
        try:
            with torch.inference_mode():
                inputs = self._siglip_processor(
                    text=ACTION_PROMPTS,
                    images=pil,
                    padding="max_length",
                    return_tensors="pt",
                ).to(device)
                outputs = self._siglip_model(**inputs)
                logits = outputs.logits_per_image[0].detach().float().cpu().numpy()
                ex = np.exp(logits - np.max(logits))
                probs = ex / (np.sum(ex) + 1e-8)
            return {ACTION_PROMPTS[i]: float(probs[i]) for i in range(len(ACTION_PROMPTS))}
        except Exception:
            return None

    def videomae_topk(self, bgr_frames: list[np.ndarray], k: int = 5) -> list[tuple[str, float]] | None:
        """Kinetics-class logits on a short clip; auxiliary only."""
        if not self.has_videomae or not self.config.use_videomae_clips or len(bgr_frames) < 16:
            return None
        device = getattr(self, "_videomae_device", "cpu")
        frames = bgr_frames[-16:]
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        try:
            inputs = self._videomae_processor(rgb, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = self._videomae_model(**inputs).logits[0].float().cpu().numpy()
        except Exception:
            return None
        id2label = getattr(self._videomae_model.config, "id2label", {}) or {}
        top_idx = np.argsort(logits)[-k:][::-1]
        out: list[tuple[str, float]] = []
        for idx in top_idx:
            label = id2label.get(int(idx), f"class_{int(idx)}")
            out.append((label, float(logits[idx])))
        return out

    @staticmethod
    def _bgr_to_pil(bgr: np.ndarray):
        if bgr is None or bgr.size == 0:
            return None
        try:
            from PIL import Image

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except Exception:
            return None


def pipeline_config_from_flags(
    use_pretrained_stack: str = "true",
    use_videomae: str = "false",
) -> PipelineConfig:
    """Map API/CLI string flags to `PipelineConfig` (E-BARD path is separate `weights`)."""
    on = str(use_pretrained_stack).lower() in ("1", "true", "yes", "on")
    vm = str(use_videomae).lower() in ("1", "true", "yes", "on")
    return PipelineConfig(
        use_siglip_teams=on,
        use_trocr_jersey=on,
        use_siglip_action_hints=on,
        use_videomae_clips=vm,
    )


def build_stack(config: PipelineConfig | None = None) -> PretrainedStack | None:
    """Returns None if nothing requested; otherwise a stack (possibly with load_errors)."""
    cfg = config or PipelineConfig()
    if not any(
        (
            cfg.use_siglip_teams,
            cfg.use_trocr_jersey,
            cfg.use_siglip_action_hints,
            cfg.use_videomae_clips,
        )
    ):
        return None
    if torch is None or SiglipModel is None:
        return PretrainedStack(config=cfg)

    return PretrainedStack(config=cfg)
