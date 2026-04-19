# GCP GPU worker pool (SpotBaller)

This folder supports **GCS + Pub/Sub** workers that run the same `run_video_analysis` pipeline as local jobs.

## Single GPU VM (personal / Cursor / SSH)

Use a **Compute Engine VM + T4/L4** without GKE: run the app **on the VM**; your Mac only **SSH’s** or uses **Cursor Remote SSH**.

1. **SSH:** `gcloud compute ssh YOUR_VM --zone=YOUR_ZONE`
2. **Put code on the VM:** `git clone <your-fork-or-repo> ~/SpotBaller` (or `gcloud compute scp --recurse ./SpotBaller YOUR_VM:~/SpotBaller --zone=ZONE` from your Mac).
3. **Bootstrap** (installs deps + CUDA PyTorch when `nvidia-smi` exists):

   ```bash
   cd ~/SpotBaller
   bash infra/gcp/setup_gcp_dev_vm.sh
   ```

   Optional: `export SPOTBALLER_REPO='https://github.com/ORG/SpotBaller.git'` before first run to clone into `~/SpotBaller`.

4. **Run a job:**

   ```bash
   source ~/SpotBaller/.venv/bin/activate
   export PYTHONPATH=$HOME/SpotBaller
   cd ~/SpotBaller
   python3 -m app.run_local --video ~/videos/game.mp4 --out ~/runs/job1
   ```

5. **API + Mac browser:** start `uvicorn app.api.main:app --host 0.0.0.0 --port 8000` in **tmux** on the VM; from the Mac:

   `gcloud compute ssh YOUR_VM --zone=ZONE -- -L 8000:localhost:8000`  
   then open `http://localhost:8000` and use **`mode=local`**.

6. **Cursor:** Remote-SSH to the same VM, open **`~/SpotBaller`** — terminal and Python run on the **GPU VM**.

### VM hardware (example: `spotballer-vm-2`)

Inspect shape and GPU:

```bash
gcloud compute instances describe spotballer-vm-2 --zone=asia-southeast1-a --format='yaml(machineType,guestAccelerators)'
nvidia-smi
python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Example profile: **custom-8-16384** (8 vCPU, 16 GiB RAM) + **NVIDIA T4** (16 GiB). The pipeline uses **CUDA** for YOLO and for SigLIP/TrOCR/VideoMAE when `torch.cuda.is_available()`.

**Tuning (dashboard “run on VM” / `app.gcp.vm_runner`):** the API host can set:

| Variable | Default | Effect |
|----------|---------|--------|
| `SPOTBALLER_VM_PREFETCH_FRAMES` | `4` | Decode ahead while inference runs |
| `SPOTBALLER_VM_ASYNC_WRITER` | `1` | Encode annotated video on a background thread |
| `SPOTBALLER_VM_IDENTITY_STRIDE` | `1` | Set to `2` or `3` after `git pull` on the VM for faster runs (fewer SigLIP/TrOCR steps) |
| `SPOTBALLER_VM_PROGRESS_POLL_SEC` | `5` | API host: interval to copy `progress.json` from the VM while the job runs (live dashboard/API) |
| `SPOTBALLER_CUDA_TUNING` | `1` on remote | cuDNN benchmark + faster matmul on CUDA |

On the VM directly, you can also run:

```bash
python3 -m app.run_local --video … --out … --prefetch-frames 4 --async-writer --identity-stride 2
```

### Hugging Face (token warning / “still calling HF”)

Transformers loads SigLIP / TrOCR / VideoMAE with ``from_pretrained("org/model")``. **Downloaded weights** live under ``~/.cache/huggingface/hub``, but the library can still **contact huggingface.co** (metadata / version checks), which shows **unauthenticated** warnings without a token.

**Option A — authenticated Hub (recommended if you ever pull new models):**

```bash
export HF_TOKEN="hf_..."   # from https://huggingface.co/settings/tokens
# or: huggingface-cli login
```

**Option B — fully offline after the cache is on the VM** (no Hub calls; requires snapshots present under ``~/.cache/huggingface/hub``):

```bash
export SPOTBALLER_HF_OFFLINE=1
# equivalent: export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
```

**Option C — local files only** (same as B for ``from_pretrained`` kwargs):

```bash
export SPOTBALLER_HF_LOCAL_ONLY=1
```

Populate the cache on the VM once (same repos as local):  
``cd ~/SpotBaller && PYTHONPATH=. python3 scripts/predownload_models.py``  
—or copy ``~/.cache/huggingface`` from your Mac:  
``rsync -av ~/.cache/huggingface/ user@vm:~/.cache/huggingface/``

### One-command deploy from your laptop (pack + upload + bootstrap)

From the **SpotBaller repo root** on your Mac (after `gcloud auth login` and `gcloud config set project …`):

```bash
bash infra/gcp/deploy_to_vm.sh
```

Defaults: `VM=spotballer-vm-2`, `ZONE=asia-southeast1-a`, `PROJECT=datacloudpoc`. Override if needed:

```bash
VM=spotballer-vm-2 ZONE=asia-southeast1-a PROJECT=datacloudpoc bash infra/gcp/deploy_to_vm.sh
```

### Automatic VM ↔ repo sync

Tarball deploy (`deploy_to_vm.sh`) **does not** include `.git`, so use a **git clone** on the VM if you want pulls/cron/CI sync.

**One command from your laptop** (uploads the bootstrap script and runs it over SSH; defaults match `deploy_to_vm.sh`):

```bash
cd /path/to/SpotBaller
bash infra/gcp/laptop_vm_clone_and_sync.sh
# Optional: SPOTBALLER_REPO=git@github.com:tungtang/spotBaller.git VM_SYNC_USE_IAP=0 bash infra/gcp/laptop_vm_clone_and_sync.sh
```

That runs `vm_bootstrap_git_sync.sh` on the VM: **clone** (or move aside an old tarball tree), **`setup_gcp_dev_vm.sh`**, then **cron** every 5 minutes for `vm_pull_latest.sh`. If `~/SpotBaller` already exists from a tarball, it is renamed to `~/SpotBaller.pre-git.<timestamp>` before cloning.

**Alternatively**, skip git on the VM and keep pushing updates with **`bash infra/gcp/deploy_to_vm.sh`** from your laptop whenever you want the instance refreshed. That path works without GitHub credentials on the VM, but **cron / `vm_pull_latest` / the GitHub workflow** need a real clone (or you must change those automations to run `deploy_to_vm.sh` from CI instead of `git pull` on the VM).

**Option A — poll from the VM (cron only, if you already cloned):**

```bash
chmod +x ~/SpotBaller/infra/gcp/vm_pull_latest.sh
bash ~/SpotBaller/infra/gcp/install_vm_auto_sync.sh
```

Default: every **5** minutes (`SPOTBALLER_VM_CRON_MINS`), branch **`main`**. Logs: `~/SpotBaller/runtime/vm_sync.log`.  
Optional: `export SPOTBALLER_VM_POST_PULL='sudo systemctl restart myapi'` before `install_vm_auto_sync.sh` if you need a restart hook (use a real unit name).

**Option B — push from GitHub (CI):** enable `.github/workflows/vm-sync-on-push.yml` by adding secrets `GCP_SA_JSON`, `GCP_PROJECT`, `GCP_ZONE`, `GCP_VM_NAME` (see comments in the workflow). Each push to `main` / `master` runs `vm_pull_latest.sh` over SSH (IAP tunnel by default). If IAP is not set up, delete `--tunnel-through-iap` from the workflow and ensure SSH is reachable from GitHub (not recommended; prefer IAP or a self-hosted runner in GCP).

**Option C — ad hoc from your laptop:**

```bash
bash infra/gcp/trigger_vm_pull.sh
```

**Private GitHub repo:** on the VM, use HTTPS + [fine-grained token](https://github.com/settings/tokens) or SSH deploy key (`git@github.com:...`) with `~/.ssh` configured.

**gcloud + Python:** SDK 565+ needs **Python 3.10+** for `gcloud compute ssh` / `scp`. If you see `unsupported operand type(s) for |` or similar, install Python 3.12 (e.g. from [python.org](https://www.python.org/downloads/)) and:

```bash
export CLOUDSDK_PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
bash infra/gcp/deploy_to_vm.sh
```

(Adjust the path to your `python3.10+` binary.)

If `torch.cuda.is_available()` is **False** after setup, check NVIDIA drivers on the image and reinstall PyTorch with a [matching CUDA wheel](https://pytorch.org/get-started/locally/) (override `PYTORCH_CUDA_URL`, e.g. `cu121`).

## One-time GCP setup

1. Create a bucket (e.g. `spotballer-prod`) for uploads and job outputs.
2. Create Pub/Sub topic `video-jobs` and subscription `video-workers` (pull).
   - Increase **ack deadline** toward the max your jobs need (up to **600s**). Longer jobs require a DB lease / Cloud Tasks pattern (see `architecture/scale-up.md`).
3. Grant the worker service account:
   - `roles/storage.objectAdmin` on the bucket (or narrower prefix IAM).
   - `roles/pubsub.subscriber` on the subscription.
4. Grant the API (or operator) service account `roles/pubsub.publisher` on the topic.

## One image: API + worker

Build **`infra/gcp/Dockerfile.gpu-worker`** once; push a single tag (e.g. `…/spotballer/gcp:latest`). Use **two Deployments** (or API elsewhere) with different env:

| Role | `SPOTBALLER_CONTAINER_ROLE` | Notes |
|------|-----------------------------|--------|
| Pub/Sub consumer | `worker` (default if unset) | GPU node pool; see `k8s/gpu-worker-deployment.example.yaml` |
| FastAPI | `api` | CPU node pool recommended; see `k8s/api-deployment.example.yaml` |

Entry command (Docker `CMD`): `python -m app.gcp.container_main`.

## API env (enqueue)

Set on the FastAPI host / `api` pod:

- `SPOTBALLER_CONTAINER_ROLE=api`
- `SPOTBALLER_GCP_PROJECT` — GCP project id  
- `SPOTBALLER_GCS_BUCKET` — bucket name (no `gs://`)  
- `SPOTBALLER_PUBSUB_TOPIC` — topic id (`video-jobs`)
- Optional: `SPOTBALLER_API_HOST`, `SPOTBALLER_API_PORT` (defaults `0.0.0.0` / `8000`)

Install deps: `pip install -r requirements.txt -r requirements-gcp.txt`

Create a job with **`mode=gcp`** (form field). `video_path` may be a **local path** (API uploads to `gs://…/uploads/`) or a **`gs://…`** URI.

## Worker env (GKE pod)

- `SPOTBALLER_CONTAINER_ROLE=worker` (optional; default)
- `GOOGLE_CLOUD_PROJECT` or `SPOTBALLER_GCP_PROJECT`  
- `SPOTBALLER_GCS_BUCKET`  
- `SPOTBALLER_PUBSUB_SUBSCRIPTION` — subscription id (`video-workers`)  
- Optional: `SPOTBALLER_PUBSUB_MAX_MESSAGES` (default `1`)

Deploy with GPU limits (see `k8s/gpu-worker-deployment.example.yaml`). Use **Workload Identity**: API service account needs **Pub/Sub publisher** + **GCS**; worker SA needs **subscriber** + **GCS**.

## Process command (local / override)

```text
python -m app.gcp.container_main
```

Worker-only (same as before): `python -m app.gcp.worker_main`

## Job layout in GCS

- Input: `gs://BUCKET/uploads/{job_id}_{filename}` (or your own `gs://` passed in)  
- Output: `gs://BUCKET/jobs/{job_id}/` — full artifact set + `job.json`  
- The API refreshes local `runtime/jobs/{id}/job.json` from GCS on `GET /jobs/{id}` when status is terminal.
