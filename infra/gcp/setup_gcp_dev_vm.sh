#!/usr/bin/env bash
# Bootstrap SpotBaller on a GCE GPU VM (Ubuntu/Debian).
# Run ON THE VM after SSH:  bash infra/gcp/setup_gcp_dev_vm.sh
#
# Optional env:
#   SPOTBALLER_HOME   — clone/install directory (default: $HOME/SpotBaller)
#   SPOTBALLER_REPO   — git URL to clone if SPOTBALLER_HOME is missing (optional)
#   PYTORCH_CUDA_URL  — PyTorch wheel index (default: cu124)
set -euo pipefail

SPOTBALLER_HOME="${SPOTBALLER_HOME:-$HOME/SpotBaller}"
PYTORCH_CUDA_URL="${PYTORCH_CUDA_URL:-https://download.pytorch.org/whl/cu124}"

echo "==> SpotBaller VM setup: SPOTBALLER_HOME=$SPOTBALLER_HOME"

if [[ -n "${SPOTBALLER_REPO:-}" ]] && [[ ! -d "$SPOTBALLER_HOME/.git" ]]; then
  echo "==> Cloning $SPOTBALLER_REPO -> $SPOTBALLER_HOME"
  mkdir -p "$(dirname "$SPOTBALLER_HOME")"
  git clone "$SPOTBALLER_REPO" "$SPOTBALLER_HOME"
fi

if [[ ! -f "$SPOTBALLER_HOME/requirements.txt" ]]; then
  echo "No requirements.txt under $SPOTBALLER_HOME"
  echo "Clone the repo first, e.g.:"
  echo "  export SPOTBALLER_REPO='https://github.com/YOU/SpotBaller.git'"
  echo "  export SPOTBALLER_HOME='$SPOTBALLER_HOME'"
  echo "  bash infra/gcp/setup_gcp_dev_vm.sh"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing OS packages (git, Python, ffmpeg, Tesseract, OpenCV GL)..."
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-venv \
    python3-pip \
    ffmpeg \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0
fi

cd "$SPOTBALLER_HOME"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "==> pip install -r requirements.txt (base deps)..."
pip install -r requirements.txt

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU detected; installing PyTorch CUDA wheels from $PYTORCH_CUDA_URL ..."
  pip install --upgrade torch torchvision --index-url "$PYTORCH_CUDA_URL"
else
  echo "==> No nvidia-smi; keeping CPU PyTorch from requirements.txt"
fi

echo "==> Verifying torch + CUDA..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo ""
echo "Hugging Face (optional):"
echo "  - Predownload: cd $SPOTBALLER_HOME && PYTHONPATH=. python3 scripts/predownload_models.py"
echo "  - Then avoid Hub API calls: export SPOTBALLER_HF_OFFLINE=1"
echo "  - Or set HF_TOKEN=hf_... from https://huggingface.co/settings/tokens"
echo ""
echo "==> Done. Next:"
echo "    source $SPOTBALLER_HOME/.venv/bin/activate"
echo "    export PYTHONPATH=$SPOTBALLER_HOME"
echo "    cd $SPOTBALLER_HOME"
echo "    python3 -m app.run_local --video /path/to/video.mp4 --out \$HOME/runs/test1"
echo ""
echo "API (tmux recommended):"
echo "    uvicorn app.api.main:app --host 0.0.0.0 --port 8000"
echo "From Mac, SSH tunnel: gcloud compute ssh VM --zone ZONE -- -L 8000:localhost:8000"
