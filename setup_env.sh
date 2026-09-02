#!/usr/bin/env bash
# =============================================================================
# One-time environment setup for the filler-token / J-lens project.
#
# Works on:
#   * a fresh rented GPU box (Runpod / Lambda / any Linux)   -> venv or conda
#   * a local Windows machine via Git Bash with miniconda    -> conda
#   * macOS / CPU-only laptops                               -> conda or venv
#
# It auto-detects everything. Just run:
#     bash setup_env.sh
#
# Optional overrides (export before running):
#   ENV_BACKEND=conda|venv|system   force the environment backend (default: auto)
#   ENV_NAME=jlens-filler           conda env name (default below)
#   PY_VERSION=3.11                 python for a freshly created conda env
#   TORCH_VARIANT=auto|cuda|cpu|skip  torch build (default: auto = CUDA if a GPU
#                                     is present, else the CPU wheel)
#   CUDA_TAG=cu128                  force a specific PyTorch CUDA index
#   SKIP_BNB=1                      skip bitsandbytes (CUDA-only; DeepSeek does not need it)
#   HF_TOKEN=hf_...                 Hugging Face auth for gated weights
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-jlens-filler}"
PY_VERSION="${PY_VERSION:-3.11}"
ENV_BACKEND="${ENV_BACKEND:-auto}"
TORCH_VARIANT="${TORCH_VARIANT:-auto}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Platform detection
# ---------------------------------------------------------------------------
case "$(uname -s)" in
  Linux*)                      OS=linux   ;;
  Darwin*)                     OS=macos   ;;
  MINGW*|MSYS*|CYGWIN*)        OS=windows ;;
  *)                           OS=unknown ;;
esac
say "Platform: $OS  ($(uname -s -m))"

command -v git >/dev/null 2>&1 || die "git is not installed / not on PATH."

# ---------------------------------------------------------------------------
# 1. Find conda (miniconda / anaconda / miniforge), if it is installed
# ---------------------------------------------------------------------------
CONDA_BASE=""

find_conda_base() {
  # (a) conda already on PATH and functional
  if command -v conda >/dev/null 2>&1; then
    local base
    if base="$(conda info --base 2>/dev/null)"; then
      [ -n "$base" ] && { printf '%s\n' "$base"; return 0; }
    fi
  fi
  # (b) $CONDA_EXE set by an activated shell
  if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    printf '%s\n' "$(dirname "$(dirname "$CONDA_EXE")")"; return 0
  fi
  # (c) common install locations, incl. the Windows ones as Git Bash sees them
  local win_home=""
  if [ -n "${USERPROFILE:-}" ] && command -v cygpath >/dev/null 2>&1; then
    win_home="$(cygpath -u "$USERPROFILE" 2>/dev/null || true)"
  fi
  local c
  for c in \
      "${CONDA_ROOT:-}" \
      "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
      "$HOME/Miniconda3" "$HOME/Anaconda3" \
      "${win_home:+$win_home/miniconda3}" "${win_home:+$win_home/anaconda3}" \
      "${win_home:+$win_home/Miniconda3}" "${win_home:+$win_home/Anaconda3}" \
      "${win_home:+$win_home/AppData/Local/miniconda3}" \
      "${win_home:+$win_home/AppData/Local/Continuum/miniconda3}" \
      /c/ProgramData/miniconda3 /c/ProgramData/Miniconda3 \
      /c/ProgramData/anaconda3 /c/ProgramData/Anaconda3 \
      /opt/conda /opt/miniconda3 /opt/anaconda3 /usr/local/miniconda3 ; do
    [ -n "$c" ] || continue
    if [ -f "$c/etc/profile.d/conda.sh" ]; then printf '%s\n' "$c"; return 0; fi
  done
  return 1
}

if [ "$ENV_BACKEND" != "venv" ] && [ "$ENV_BACKEND" != "system" ]; then
  CONDA_BASE="$(find_conda_base || true)"
fi

# ---------------------------------------------------------------------------
# 2. Create + activate the environment
#    conda if available (your local box), plain venv otherwise (rented GPU pod)
# ---------------------------------------------------------------------------
if [ "$ENV_BACKEND" = "auto" ]; then
  if [ -n "$CONDA_BASE" ]; then ENV_BACKEND=conda; else ENV_BACKEND=venv; fi
fi

case "$ENV_BACKEND" in

  conda)
    [ -n "$CONDA_BASE" ] || die "ENV_BACKEND=conda but no conda installation was found."
    say "Using conda at: $CONDA_BASE"
    # conda's own scripts are not written for 'set -u'
    set +u
    # shellcheck disable=SC1091
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    # (awk consumes all input -> no SIGPIPE surprise under 'set -o pipefail')
    ENV_EXISTS="$(conda env list | awk -v n="$ENV_NAME" '$1==n{f=1} END{print f+0}')"
    if [ "$ENV_EXISTS" = "1" ] || [ -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
      echo "conda env '$ENV_NAME' already exists - reusing it."
    else
      say "Creating conda env '$ENV_NAME' (python $PY_VERSION)"
      conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip
    fi
    conda activate "$ENV_NAME"
    set -u
    ;;

  venv)
    say "No conda found - using a plain virtualenv (.venv)"
    PYBOOT=""
    for cand in python3 python py; do
      if command -v "$cand" >/dev/null 2>&1; then PYBOOT="$cand"; break; fi
    done
    [ -n "$PYBOOT" ] || die "No python interpreter found on PATH."
    [ -d .venv ] || "$PYBOOT" -m venv .venv
    set +u
    if [ -f .venv/bin/activate ]; then
      # shellcheck disable=SC1091
      . .venv/bin/activate            # Linux / macOS
    else
      # shellcheck disable=SC1091
      . .venv/Scripts/activate        # Windows
    fi
    set -u
    ;;

  system)
    say "ENV_BACKEND=system - installing into the current interpreter, no new env"
    ;;

  *) die "Unknown ENV_BACKEND='$ENV_BACKEND' (use conda | venv | system)" ;;
esac

PY="python"
command -v "$PY" >/dev/null 2>&1 || PY="python3"
say "Interpreter: $($PY -c 'import sys; print(sys.executable)')  ($($PY -V 2>&1))"

"$PY" -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 3. PyTorch  (CUDA wheel where there is a GPU, CPU wheel otherwise)
# ---------------------------------------------------------------------------
detect_cuda_tag() {
  # Read "CUDA Version: 12.8" out of nvidia-smi and map it to a PyTorch index.
  local ver major minor
  ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9]*\).*/\1/p' | head -1)"
  [ -n "$ver" ] || { echo cu126; return; }
  major="${ver%%.*}"; minor="${ver#*.}"
  if   [ "$major" -ge 13 ];                              then echo cu128
  elif [ "$major" -eq 12 ] && [ "${minor:-0}" -ge 8 ];   then echo cu128
  elif [ "$major" -eq 12 ] && [ "${minor:-0}" -ge 6 ];   then echo cu126
  elif [ "$major" -eq 12 ];                              then echo cu121
  else                                                        echo cu118
  fi
}

HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_GPU=1
fi

if [ "$TORCH_VARIANT" = "auto" ]; then
  if [ "$HAS_GPU" -eq 1 ]; then TORCH_VARIANT=cuda; else TORCH_VARIANT=cpu; fi
fi

if "$PY" -c 'import torch' >/dev/null 2>&1 && [ "${FORCE_TORCH:-0}" != "1" ]; then
  say "torch already installed ($("$PY" -c 'import torch;print(torch.__version__)')) - leaving it alone (FORCE_TORCH=1 to reinstall)"
elif [ "$TORCH_VARIANT" = "skip" ]; then
  say "TORCH_VARIANT=skip - not installing torch"
elif [ "$TORCH_VARIANT" = "cuda" ]; then
  TAG="${CUDA_TAG:-$(detect_cuda_tag)}"
  say "NVIDIA GPU detected - installing CUDA torch ($TAG)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
  "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$TAG" || {
    warn "CUDA wheel index $TAG failed; falling back to the default PyPI torch build."
    "$PY" -m pip install torch
  }
else
  if [ "$OS" = "macos" ]; then
    say "No CUDA GPU - installing the default torch build (MPS on Apple Silicon)"
    "$PY" -m pip install torch
  else
    say "No CUDA GPU detected - installing the CPU-only torch wheel"
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

# ---------------------------------------------------------------------------
# 4. Project dependencies
#    bitsandbytes is CUDA-only (4-bit loading of the bf16 side models such as
#    gemma-27b; DeepSeek V4 Flash ships pre-quantized and does not use it). On a
#    CPU/macOS box it is skipped so the rest of the install still succeeds.
# ---------------------------------------------------------------------------
say "Installing project requirements"
if [ "$HAS_GPU" -eq 0 ] || [ "${SKIP_BNB:-0}" = "1" ]; then
  grep -v -i '^[[:space:]]*bitsandbytes' requirements.txt > /tmp/req_nobnb.$$.txt
  "$PY" -m pip install -r /tmp/req_nobnb.$$.txt
  rm -f /tmp/req_nobnb.$$.txt
  echo "(skipped bitsandbytes - no CUDA GPU here; only the bf16 side models use it)"
else
  "$PY" -m pip install -r requirements.txt
fi

# openai client: used by 10/11 for the OpenRouter accuracy sweeps
"$PY" -m pip install "openai>=1.40"

# Notebook tooling: 10/11 are '# %%' cell scripts, which VS Code runs through
# ipykernel. Registering the kernel also makes the env selectable from .ipynb files.
say "Installing notebook tooling (ipykernel, ipython)"
"$PY" -m pip install ipykernel ipython
"$PY" -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)" \
  || warn "could not register the Jupyter kernel; VS Code will still run '# %%' cells via the selected interpreter."

# ---------------------------------------------------------------------------
# 5. Jacobian-lens reference implementation (package: jlens)
#    Repo: https://github.com/anthropics/jacobian-lens  (Apache-2.0)
# ---------------------------------------------------------------------------
say "Installing the Jacobian lens (jlens)"
if [ ! -d jacobian-lens ]; then
  git clone https://github.com/anthropics/jacobian-lens.git
else
  echo "jacobian-lens/ already present - pulling latest"
  git -C jacobian-lens pull --ff-only || warn "could not fast-forward jacobian-lens; using the local checkout"
fi
"$PY" -m pip install -e ./jacobian-lens

# ---------------------------------------------------------------------------
# 6. Hugging Face auth (DeepSeek V4 Flash weights + the lens .pt download)
#    On a GPU box also point the cache at the big volume, e.g.
#       export HF_HOME=/workspace/hf
# ---------------------------------------------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
  say "Logging in to Hugging Face with \$HF_TOKEN"
  "$PY" -c "import os; from huggingface_hub import login; login(os.environ['HF_TOKEN'])"
else
  warn "HF_TOKEN not set. Run 'huggingface-cli login' before downloading gated weights."
fi
[ -n "${HF_HOME:-}" ] || [ "$HAS_GPU" -eq 0 ] || \
  warn "HF_HOME is not set. On a rented pod, 'export HF_HOME=/workspace/hf' keeps the 300GB+ of weights off the small root disk."

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------
say "Verifying the install"
"$PY" - <<'PYCHECK'
import importlib, sys
import torch, transformers
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__)
print("cuda avail  ", torch.cuda.is_available(),
      f"({torch.cuda.device_count()} GPU(s))" if torch.cuda.is_available() else "")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}      {p.name}  {p.total_memory/1e9:.0f} GB")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("mps avail    True")
print("transformers", transformers.__version__)
for mod in ("jlens", "accelerate", "huggingface_hub", "pandas", "matplotlib",
            "openai", "dotenv", "ipykernel"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:<12} OK {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"{mod:<12} MISSING ({e})")
try:
    import bitsandbytes  # noqa: F401
    print("bitsandbytes OK")
except Exception:
    print("bitsandbytes not installed (CUDA-only; for bf16 side models, not DeepSeek)")
PYCHECK

# ---------------------------------------------------------------------------
# 8. What to do next
# ---------------------------------------------------------------------------
say "Done."
# The interpreter path, printed literally — conda does not always put an env under
# <base>/envs (if the base install is not writable it lands in ~/.conda/envs instead),
# and this is the exact string VS Code wants under "Enter interpreter path...".
INTERP="$("$PY" -c 'import sys; print(sys.executable)')"
echo "Interpreter path (paste this into VS Code if it does not autodetect):"
echo "  $INTERP"
echo
if [ "$ENV_BACKEND" = "conda" ]; then
  echo "Activate this env in new shells with:   conda activate $ENV_NAME"
  echo "In VS Code: Ctrl+Shift+P -> Python: Select Interpreter."
  echo "  A newly created env will not appear until you hit the refresh icon in that"
  echo "  dropdown (or run 'Developer: Reload Window'); otherwise use 'Enter interpreter"
  echo "  path...' with the path above."
elif [ "$ENV_BACKEND" = "venv" ]; then
  if [ -f .venv/bin/activate ]; then
    echo "Activate this env in new shells with:   source .venv/bin/activate"
  else
    echo "Activate this env in new shells with:   source .venv/Scripts/activate"
  fi
  echo "VS Code autodetects a .venv in the workspace root."
fi
echo
echo "Credentials (per machine, never committed) - export them, or put them in a .env"
echo "file in the repo root, which python-dotenv picks up:"
echo "  HF_TOKEN=hf_...                 # gated weights + lens repo"
echo "  OPENROUTER_API_KEY=sk-or-...    # 10/11 (OpenRouter)"
echo
echo "Next:"
echo "  python 00_smoke_test.py --api   # API-path check (stage 1, no GPU)"
echo "  python 00_smoke_test.py --model deepseek --tokenizer-only"
echo "                                  # DeepSeek prompt preflight on the tokenizer alone:"
echo "                                  # reasoning off, tail tokens, decode mode (no weights)"
if [ "$HAS_GPU" -eq 1 ]; then
  echo "  python 00_smoke_test.py                    # dev model on this GPU (incl. one greedy answer)"
  echo "  python run_qwen_pipeline.py                # stages 0-3 on the dev model, logged"
  echo "  python 00_smoke_test.py --model deepseek   # the real target (see run_deepseek.md)"
else
  echo "  python 00_smoke_test.py         # dev model, CPU (slow; set dtype='float32' in config.py)"
  echo "  python 11_run_fig2_sweep.py     # API-only, no GPU needed"
fi
