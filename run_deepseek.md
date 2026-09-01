# Scaling to DeepSeek V4 Flash (284B MoE) on rented GPUs

The dev model is for developing the pipeline. Once `00` and `20` are clean on `dev`,
flip to the real target. **Nothing in the code changes except the `--model deepseek`
flag** — `config.py` already sets `device_map="auto"` and loads the checkpoint as
published.

## 0. What the checkpoint actually is

`deepseek-ai/DeepSeek-V4-Flash` is **one mixed-precision release**, not a choice of
precisions: MoE expert weights are stored in **FP4**, the attention / norm / router
weights in **FP8** (`quant_method: "fp8"`, e4m3, 128×128 blocks), 284B total params
with 13B active, 43 layers, `d_model` 4096, 256 experts. On disk that is **~160 GB
over 46 shards**.

Three consequences, and they drive everything below:

- **Never add bitsandbytes.** `load_in_4bit=True` would make bnb dequantize the whole
  thing to bf16 (284B × 2 B ≈ **568 GB**) before re-quantizing it to nf4. That OOMs
  any box you would plausibly rent, and it does so *after* you have paid to download
  160 GB. `common.describe_checkpoint` now refuses this combination up front.
- **Hopper or newer.** Ampere has no FP8 units, so an A100 box would upcast the
  weights and blow the budget in a different way. H100/H200 run FP8 natively; neither
  has FP4 units (that is Blackwell), so on both, expert weights are upconverted
  inside the kernel — identical behaviour on either card.
- **The third-party re-quants are the wrong model.** `RedHatAI/…-BF16`,
  `nvidia/…-NVFP4` and the community W4A16 repos are not what the lens was fit on.
  `check_provenance()` will say so, and it is right: the `J_l` matrices only mean
  something in the residual basis of the checkpoint they were fitted against.

## 1. Free preflight — do this before renting anything

The one real risk is that HF transformers has no native path for the FP4 expert
weights and upcasts them at load. You can find that out on your laptop, with no
download and no GPU:

```python
from accelerate import init_empty_weights
import transformers

cfg = transformers.AutoConfig.from_pretrained(
    "deepseek-ai/DeepSeek-V4-Flash", trust_remote_code=True)
print(cfg.quantization_config)                 # what the checkpoint declares
with init_empty_weights():
    m = transformers.AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
from collections import Counter
print(Counter(p.dtype for p in m.parameters()))  # what transformers would allocate
```

Read the dtype histogram, not the parameter count:

| what you see | footprint | box |
|---|---|---|
| experts stay FP4/FP8 | ~160 GB | **4×80GB** — 2× headroom |
| experts upcast to bf16 | ~542 GB | 8×80GB (4×H200's 564 GB leaves nothing for activations) |
| everything upcast to bf16 | ~568 GB | 8×80GB |

Note what that table says about **4×H200**: it does not cover either upcast case, so
it is not insurance — if you want insurance for roughly the same hourly price,
8×H100 is the box that actually covers them.

vLLM has purpose-built kernels for this checkpoint and would sidestep the question
entirely, but it is not an option here: jlens reads the residual stream through
forward hooks, and vLLM does not expose it. This has to run on HF transformers.

## 2. Pick a box

**4×H100 80GB (320 GB)** is the default: 2× the weights, ~$11/hr on Vast at the time
of writing. Go to 8×H100 only if the preflight says the experts upcast.

- **Skip spot**, despite the price. A preemption costs you the 160 GB re-download,
  which is most of the bill on a run this short.
- Prefer on-demand / Secure Cloud for a stable multi-hour session.
- The readout itself is short — n=300 at k=10 is roughly 30–60 minutes of compute —
  so download time and run time are comparable and the whole session is 2–3 hours.
  Optimising the hourly rate matters less than not having to do the session twice.

## 3. Storage

- Model weights: budget **200 GB** of pod disk for the 160 GB snapshot plus slack.
  (Earlier versions of this note said 300–600 GB, which assumed a bf16 download that
  does not exist.)
- The lens repo `camilablank/workspace-lenses` is ~47 GB total, but you only need
  `deepseek-v4-flash/j-lens/lens.pt` — **1.41 GB** —, and
  `JacobianLens.from_pretrained(..., filename=...)` downloads just that file. It
  covers ~42 of the 43 layers and expands to ~2.8 GB of fp32 in host RAM.
- Point HF caches at the big volume: `export HF_HOME=/workspace/hf`.

## 4. Environment

```bash
git clone https://github.com/simonbernier/jlens-filler.git && cd jlens-filler
export HF_HOME=/workspace/hf      # do this BEFORE setup_env.sh — keeps the weights off root
export HF_TOKEN=hf_...            # needed for deepseek weights
bash setup_env.sh
source .venv/bin/activate         # in new shells; `conda activate jlens-filler` if the pod has conda
python 00_smoke_test.py --model deepseek     # FIRST: does the big lens load + apply?
python 20_lens_readout.py --model deepseek --n 40 --k 10   # then a short readout run
```

Run the smoke test before anything else — it is what tells you the load worked, the
provenance matches, and the two lenses actually differ, and it is cheap enough to
fail fast on. If it OOMs, kill the instance inside the first hour and re-rent 8×.

`setup_env.sh` is the same script you run locally — it adapts to the pod on its own:

- **Environment:** a bare pod has no conda, so the script creates `.venv`. If the
  image *does* ship conda (many GPU images do), it creates/reuses a `jlens-filler`
  conda env instead. Force either with `ENV_BACKEND=venv` / `ENV_BACKEND=conda`.
- **torch:** if the image already has a tuned torch, the script leaves it alone —
  usually what you want on a rented GPU image. Otherwise it reads the CUDA version
  out of `nvidia-smi` and installs the matching wheel (`cu128` for a 12.8 driver, and
  so on). Override with `FORCE_TORCH=1`, or `CUDA_TAG=cu126` to pin the index.
- **bitsandbytes** installs wherever there is a GPU. DeepSeek does **not** use it (see
  §0); it is there for the bf16 side models such as `gemma-27b`. Its absence on a
  CPU-only box is not a problem for this run.
- The closing verification block prints each GPU's name and VRAM plus a `jlens` import
  check. Read it before kicking off a 160 GB download — it is the cheapest place to
  catch a pod that came up with fewer GPUs than you paid for.
- If the pod is a bare container, `git` may be missing:
  `apt-get update && apt-get install -y git` first (the script checks for git up front
  and stops with a clear message).

`accelerate` (installed) handles the multi-GPU sharding via `device_map="auto"`.

## 5. Gotchas

- **`trust_remote_code=True`** is already set — DeepSeek ships custom modeling code.
- **Layer coverage:** the scripts iterate `lens.source_layers`, so they adapt to
  whatever the lens covers without being told.
- **Verify the snapshot:** if HF resolves `deepseek-ai/DeepSeek-V4-Flash` to a revision
  the lens wasn't fit on, `check_provenance()` warns — pin `hf_id` to the provenance's
  exact `model_id`/revision.
- **Context window:** every lens call goes through `common.apply_lens`, which sizes the
  window to the prompt. Don't pass a fixed `--max-seq-len` unless you have checked it
  against the longest prompt at your `k`; too small now raises rather than silently
  truncating, but it still stops the run.
- **Cost control:** start with `--n 2 --k 10` to confirm it runs end to end before a
  full sweep. Both CSVs checkpoint every 10 examples, so an interrupted sweep keeps
  what it had.
- **Don't expect the API and the local weights to agree exactly.** Same weights do not
  mean same outputs across serving stacks: MoE routing is batch-composition sensitive,
  and providers differ in kernels, tensor-parallel degree and attention backend. The
  useful move is to *measure* the agreement — `answers_<tag>.csv` from `20` and
  `results/fig2_raw.jsonl` from stage 1 share the seeded test set, so they join on
  `idx` for an agreement rate and a McNemar test.
- **Don't reach for an API** to save money on the lens work: APIs don't expose hidden
  states. The API is only useful for *behavioral* uplift checks (stage 1).
