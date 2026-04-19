"""
Run ``app.run_local`` on a GCE VM via ``gcloud compute scp`` + ``ssh``.

Requires ``gcloud`` on the API host PATH and authenticated project (``gcloud auth login``).

Environment (all optional except VM/zone/project for a successful run):

- ``SPOTBALLER_GCLOUD_VM`` — instance name (default ``spotballer-vm1``)
- ``SPOTBALLER_GCLOUD_ZONE`` — e.g. ``asia-southeast1-b``
- ``SPOTBALLER_GCLOUD_PROJECT`` — GCP project id
- ``SPOTBALLER_VM_REMOTE_HOME`` — remote SpotBaller root (default ``~/SpotBaller``)
- ``SPOTBALLER_VM_UPLOAD_DIR`` — where to place uploaded videos on VM (default ``~/spotballer_vm_uploads``)
- ``CLOUDSDK_PYTHON`` — if set, prefixed when invoking ``gcloud`` (SDK 565+ needs Python 3.10+)
- ``SPOTBALLER_VM_PREFETCH_FRAMES`` — default ``4``; passed as ``--prefetch-frames`` (0 disables)
- ``SPOTBALLER_VM_ASYNC_WRITER`` — default ``1`` (true); adds ``--async-writer``
- ``SPOTBALLER_VM_IDENTITY_STRIDE`` — default ``1`` (omit flag; set ``2``+ after deploying code that supports ``--identity-stride``)
- ``SPOTBALLER_CUDA_TUNING`` — default ``1`` on remote (cuDNN benchmark + matmul precision)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VmGcloudConfig:
    vm: str
    zone: str
    project: str
    remote_spotballer: str
    remote_upload_dir: str


def vm_config_from_env() -> VmGcloudConfig | None:
    vm = os.environ.get("SPOTBALLER_GCLOUD_VM", "").strip()
    zone = os.environ.get("SPOTBALLER_GCLOUD_ZONE", "").strip()
    project = os.environ.get("SPOTBALLER_GCLOUD_PROJECT", "").strip()
    if not (vm and zone and project):
        return None
    return VmGcloudConfig(
        vm=vm,
        zone=zone,
        project=project,
        remote_spotballer=os.environ.get("SPOTBALLER_VM_REMOTE_HOME", "~/SpotBaller").strip() or "~/SpotBaller",
        remote_upload_dir=os.environ.get("SPOTBALLER_VM_UPLOAD_DIR", "~/spotballer_vm_uploads").strip()
        or "~/spotballer_vm_uploads",
    )


def _gcloud_base(cfg: VmGcloudConfig) -> list[str]:
    return [
        "gcloud",
        "compute",
        "--project",
        cfg.project,
        "--quiet",
    ]


def _gcloud_env() -> dict[str, str]:
    env = os.environ.copy()
    py = os.environ.get("CLOUDSDK_PYTHON", "").strip()
    if py:
        env["CLOUDSDK_PYTHON"] = py
    return env


def _vm_int_env(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "").strip()
        if not v:
            return default
        return int(v)
    except ValueError:
        return default


def _vm_bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=True,
        text=True,
        env=_gcloud_env(),
    )


def run_pipeline_on_vm(
    cfg: VmGcloudConfig,
    job_id: str,
    local_video: Path,
    local_out_dir: Path,
    *,
    weights: str = "auto",
    use_pretrained_stack: bool = True,
    use_videomae: bool = False,
    max_frames: int | None = None,
) -> dict:
    """
    Upload ``local_video``, run ``python -m app.run_local`` on the VM, copy ``performance.json``
    (and a few small JSON artifacts) into ``local_out_dir``.

    Returns a dict suitable for storing under job ``result`` (includes ``performance`` key).
    """
    if not local_video.is_file():
        raise FileNotFoundError(f"Video not found: {local_video}")

    gcloud = shutil.which("gcloud")
    if not gcloud:
        raise RuntimeError("gcloud CLI not found on PATH. Install Google Cloud SDK.")

    base = _gcloud_base(cfg)
    vm_ref = f"{cfg.vm}"
    ext = local_video.suffix or ".mp4"
    remote_name = f"{job_id}{ext}"
    # Expand ~ in remote paths on the VM via bash
    remote_upload = f"{cfg.remote_upload_dir.rstrip('/')}/{remote_name}"
    remote_out = f"~/runs/spotballer_ui_{job_id}"
    remote_out_dir = remote_out.rstrip("/")

    # Ensure dirs on VM
    mkdir_cmd = f"mkdir -p {cfg.remote_upload_dir} {remote_out_dir}"
    ssh_mk = base + ["ssh", vm_ref, f"--zone={cfg.zone}", "--command", mkdir_cmd]
    r0 = _run(ssh_mk, check=False)
    if r0.returncode != 0:
        raise RuntimeError(f"gcloud ssh (mkdir): {r0.stderr or r0.stdout}")

    # Upload video
    scp_up = base + [
        "scp",
        str(local_video.resolve()),
        f"{vm_ref}:{remote_upload}",
        f"--zone={cfg.zone}",
    ]
    r1 = _run(scp_up)
    if r1.returncode != 0:
        raise RuntimeError(f"gcloud scp upload failed: {r1.stderr or r1.stdout}")

    flags = [f"--video {remote_upload}", f"--out {remote_out_dir}", f"--weights {weights}"]
    if not use_pretrained_stack:
        flags.append("--no-pretrained-stack")
    if use_videomae:
        flags.append("--videomae")
    if max_frames is not None:
        flags.append(f"--max-frames {int(max_frames)}")

    prefetch = _vm_int_env("SPOTBALLER_VM_PREFETCH_FRAMES", 4)
    if prefetch > 0:
        flags.append(f"--prefetch-frames {prefetch}")
    if _vm_bool_env("SPOTBALLER_VM_ASYNC_WRITER", True):
        flags.append("--async-writer")
    id_stride = _vm_int_env("SPOTBALLER_VM_IDENTITY_STRIDE", 1)
    if id_stride > 1:
        flags.append(f"--identity-stride {id_stride}")

    remote_py = (
        f"set -euo pipefail; "
        f"export SPOTBALLER_HF_OFFLINE=${{SPOTBALLER_HF_OFFLINE:-1}}; "
        f"export SPOTBALLER_CUDA_TUNING=${{SPOTBALLER_CUDA_TUNING:-1}}; "
        f"cd {cfg.remote_spotballer} && "
        f"source .venv/bin/activate && "
        f"export PYTHONPATH={cfg.remote_spotballer} && "
        f"python3 -m app.run_local {' '.join(flags)}"
    )
    ssh_run = base + ["ssh", vm_ref, f"--zone={cfg.zone}", "--command", remote_py]
    r2 = _run(ssh_run)
    if r2.returncode != 0:
        raise RuntimeError(f"Remote run_local failed (exit {r2.returncode}): {r2.stderr or r2.stdout}")

    # Pull key artifacts
    local_out_dir.mkdir(parents=True, exist_ok=True)
    pull_names = (
        "performance.json",
        "pipeline.json",
        "stats.json",
        "progress.json",
    )
    for name in pull_names:
        scp_down = base + [
            "scp",
            f"{vm_ref}:{remote_out_dir}/{name}",
            str(local_out_dir / name),
            f"--zone={cfg.zone}",
        ]
        r3 = _run(scp_down, check=False)
        if r3.returncode != 0 and name == "performance.json":
            raise RuntimeError(f"Could not copy performance.json: {r3.stderr or r3.stdout}")

    perf: dict | None = None
    perf_path = local_out_dir / "performance.json"
    if perf_path.is_file():
        try:
            perf = json.loads(perf_path.read_text())
        except Exception:
            perf = None

    return {
        "performance": perf,
        "remote_out": remote_out,
        "remote_vm": cfg.vm,
        "remote_zone": cfg.zone,
        "stdout_tail": (r2.stdout or "")[-4000:],
    }
