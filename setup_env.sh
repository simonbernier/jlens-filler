#!/usr/bin/env bash
# One-time environment setup for the filler-token / J-lens project.
# Works on a fresh cloud GPU box (Runpod/Lambda) or a CPU-only laptop for the small model.
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. (Recommended) a fresh virtual environment
# ---------------------------------------------------------------------------
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 1. Clone + install the Jacobian-lens reference implementation (package: jlens)
#    Repo: https://github.com/anthropics/jacobian-lens  (Apache-2.0)
# ---------------------------------------------------------------------------
if [ ! -d jacobian-lens ]; then
  git clone https://github.com/anthropics/jacobian-lens.git
fi
pip install -e ./jacobian-lens

# ---------------------------------------------------------------------------
# 2. Project dependencies
# ---------------------------------------------------------------------------
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 3. Hugging Face auth (DeepSeek V4 Flash weights + the lens repo download).
#    Set HF_TOKEN in your shell, or run `huggingface-cli login`.
# ---------------------------------------------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
  python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
else
  echo "NOTE: HF_TOKEN not set. Run 'huggingface-cli login' before downloading gated weights."
fi

echo
echo "Done. Quick check:"
python -c "import torch, transformers, jlens; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| jlens OK')"
echo "Next: python 01_smoke_test.py         # runs the 4B dev model (CPU ok, GPU faster)"
