"""
Run ``app.run_local`` on a GCE VM via ``gcloud compute scp`` + ``ssh``.

Requires ``gcloud`` on the API host PATH and authenticated project (``gcloud auth login``).

Environment (all optional except VM/zone/project for a successful run):

- ``SPOTBALLER_GCLOUD_VM`` — instance name (example ``spotballer-vm-2``)
- ``SPOTBALLER_GCLOUD_ZONE`` — e.g. ``asia-southeast1-a``
- ``SPOTBALLER_GCLOUD_PROJECT`` — GCP project id
- ``SPOTBALLER_VM_REMOTE_HOME`` — remote SpotBaller root (default ``~/SpotBaller``)
- ``SPOTBALLER_VM_UPLOAD_DIR`` — where to place uploaded videos on VM (default ``~/spotballer_vm_uploads``)
- ``CLOUDSDK_PYTHON`` — if set, prefixed when invoking ``gcloud`` (SDK 565+ needs Python 3.10+)
- ``SPOTBALLER_VM_PREFETCH_FRAMES`` — default ``4``; passed as ``--prefetch-frames`` (0 disables)
- ``SPOTBALLER_VM_ASYNC_WRITER`` — default ``1`` (true); adds ``--async-writer``
- ``SPOTBALLER_VM_IDENTITY_STRIDE`` — default ``1`` (omit flag; set ``2``+ after deploying code that supports ``--identity-stride``)
- ``SPOTBALLER_CUDA_TUNING`` — default ``1`` on remote (cuDNN benchmark + matmul precision)
- ``SPOTBALLER_VM_PROGRESS_POLL_SEC`` — default ``5``; while the remote ``run_local`` runs, ``gcloud compute scp`` pulls ``progress.json`` this often so the API/dashboard see live frames
- ``CLOUDSDK_CONFIG`` or ``SPOTBALLER_GCLOUD_CONFIG`` — override gcloud state directory (optional)
- **Writable config fallback:** if ``~/.config/gcloud`` is not writable (e.g. sandboxed API), subprocesses use ``runtime/gcloud_config`` under the repo and copy credentials from home when readable
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Cached result of :func:`_effective_gcloud_config_dir` — ``tuple`` wrapper so ``None`` is a valid cached value.
_gcloud_cfg_cache: tuple[Path | None] | None = None


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


def _repo_root() -> Path:
    """SpotBaller repo root (``app/gcp`` → parents ×3)."""
    return Path(__file__).resolve().parent.parent.parent


def _home_gcloud_dir() -> Path:
    return Path.home() / ".config" / "gcloud"


def _seed_gcloud_config_from_home(dest: Path, home_gc: Path) -> None:
    """Copy auth state so ``gcloud`` works when only home config is readable (not writable)."""
    try:
        src_cred = home_gc / "credentials.db"
        if not src_cred.is_file():
            return
        if (dest / "credentials.db").is_file():
            return
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_cred, dest / "credentials.db")
        for name in ("active_config", "config_sentinel", "default_configs.db", "access_tokens.db"):
            sp = home_gc / name
            if sp.is_file():
                shutil.copy2(sp, dest / name)
        src_cfg = home_gc / "configurations"
        if src_cfg.is_dir():
            shutil.copytree(src_cfg, dest / "configurations", dirs_exist_ok=True)
        src_legacy = home_gc / "legacy_credentials"
        if src_legacy.is_dir():
            shutil.copytree(src_legacy, dest / "legacy_credentials", dirs_exist_ok=True)
    except OSError:
        pass


def _effective_gcloud_config_dir() -> Path | None:
    """
    Directory for ``CLOUDSDK_CONFIG`` passed to every ``gcloud`` invocation.

    Returns ``None`` to leave gcloud's default (``~/.config/gcloud``) when that path is writable.

    If home config is not writable, uses ``runtime/gcloud_config`` under the repo (always writable
    under normal checkouts) and seeds it from home when possible.
    """
    global _gcloud_cfg_cache
    if _gcloud_cfg_cache is not None:
        return _gcloud_cfg_cache[0]

    explicit = (os.environ.get("CLOUDSDK_CONFIG") or os.environ.get("SPOTBALLER_GCLOUD_CONFIG") or "").strip()
    if explicit:
        p = Path(os.path.expanduser(explicit)).resolve()
        p.mkdir(parents=True, exist_ok=True)
        (p / "logs").mkdir(parents=True, exist_ok=True)
        _gcloud_cfg_cache = (p,)
        return p

    home_gc = _home_gcloud_dir()
    try:
        if home_gc.is_dir() and os.access(home_gc, os.W_OK):
            _gcloud_cfg_cache = (None,)
            return None
    except OSError:
        pass

    local_root = _repo_root() / "runtime" / "gcloud_config"
    local_root.mkdir(parents=True, exist_ok=True)
    (local_root / "logs").mkdir(parents=True, exist_ok=True)
    _seed_gcloud_config_from_home(local_root, home_gc)
    _gcloud_cfg_cache = (local_root,)
    return local_root


def reset_gcloud_config_cache_for_tests() -> None:
    """Clear cached gcloud config path (unit tests only)."""
    global _gcloud_cfg_cache
    _gcloud_cfg_cache = None


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
    cfg_dir = _effective_gcloud_config_dir()
    if cfg_dir is not None:
        env["CLOUDSDK_CONFIG"] = str(cfg_dir)
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


def _scp_remote_file(
    base: list[str],
    vm_ref: str,
    zone: str,
    remote_path: str,
    local_path: Path,
) -> bool:
    """Copy one file from VM; return True if exit code 0."""
    cmd = base + [
        "scp",
        f"{vm_ref}:{remote_path}",
        str(local_path),
        f"--zone={zone}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=_gcloud_env())
    return r.returncode == 0


def vm_progress_stub_payload(
    cfg: VmGcloudConfig,
    planned_total_frames: int | None,
    *,
    detail: str = "Remote job starting (gcloud ssh)",
) -> dict:
    """Initial ``progress.json`` for API/UI: matches local frames/total/percent shape when length is known."""
    pct: float | None = None
    if planned_total_frames is not None and planned_total_frames > 0:
        pct = 0.0
    return {
        "status": "processing",
        "frames_processed": 0,
        "total_frames": planned_total_frames,
        "percent_complete": pct,
        "vm_remote": True,
        "gcp_vm": cfg.vm,
        "gcp_zone": cfg.zone,
        "gcp_project": cfg.project,
        "detail": detail,
    }


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
    planned_total_frames: int | None = None,
) -> dict:
    """
    Upload ``local_video``, run ``python -m app.run_local`` on the VM, copy ``performance.json``
    (and a few small JSON artifacts) into ``local_out_dir``.

    ``planned_total_frames`` should match the local OpenCV frame count (and ``max_frames`` cap)
    so ``progress.json`` mirrors local jobs (frames / total / percent) before the remote writer starts.

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
    local_out_dir.mkdir(parents=True, exist_ok=True)
    poll_sec = max(2, _vm_int_env("SPOTBALLER_VM_PROGRESS_POLL_SEC", 5))
    remote_progress = f"{remote_out_dir}/progress.json"
    local_progress = local_out_dir / "progress.json"
    # So GET /jobs/{id}/progress shows frames/total like local runs (API may also write earlier in create_job).
    try:
        stub = vm_progress_stub_payload(cfg, planned_total_frames)
        local_progress.write_text(json.dumps(stub, indent=2))
    except OSError:
        pass

    proc = subprocess.Popen(
        ssh_run,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_gcloud_env(),
    )

    def _poll_vm_progress() -> None:
        """Mirror remote progress.json into local_out_dir while SSH runs."""
        while True:
            _scp_remote_file(base, vm_ref, cfg.zone, remote_progress, local_progress)
            if proc.poll() is not None:
                break
            time.sleep(poll_sec)

    poll_thread = threading.Thread(target=_poll_vm_progress, name="vm-progress-scp", daemon=True)
    poll_thread.start()

    stdout, stderr = proc.communicate()
    poll_thread.join(timeout=poll_sec + 15.0)

    r2 = subprocess.CompletedProcess(
        ssh_run,
        proc.returncode if proc.returncode is not None else -1,
        stdout,
        stderr,
    )
    if r2.returncode != 0:
        raise RuntimeError(f"Remote run_local failed (exit {r2.returncode}): {r2.stderr or r2.stdout}")

    # Pull the same artifacts as a local ``run_local`` run into ``runtime/jobs/<id>/`` so
    # ``GET /results`` and ``/results/{id}`` match local jobs (video, stats, exports).
    pull_names = (
        "performance.json",
        "pipeline.json",
        "stats.json",
        "progress.json",
        "events.json",
        "tracks.json",
        "player_identity_map.json",
        "team_box_score.json",
        "stats_by_track.json",
        "action_hints.json",
        "videomae_aux.json",
        "stats.csv",
        "team_box_score_players.csv",
        "action_hints_long.csv",
        "pipeline_performance.csv",
        "annotated.mp4",
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
