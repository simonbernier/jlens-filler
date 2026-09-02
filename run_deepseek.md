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

The second free preflight is the tokenizer (a few MB, no GPU):

```bash
python 00_smoke_test.py --model deepseek --tokenizer-only
```

It renders the real task prompt through the real chat template and reports the
three things that bit the dev model before any weights were involved:

- **there is no Jinja chat template.** V4 ships `tokenizer.chat_template = None`
  and `apply_chat_template` raises; the prompt format is the Python module
  `encoding/encoding_dsv4.py` in the weights repo. `paper_tasks.render_chat`
  detects the missing template, downloads that one file and calls its
  `encode_messages(msgs, thinking_mode="chat")`. Verified 2026-09-01 on the
  real tokenizer: the prompt is
  `<｜begin▁of▁sentence｜>{system}<｜User｜>…<｜Assistant｜></think>`, 270 tokens
  at k=10;
- **reasoning is off** — V4 Flash is a hybrid reasoning model; `thinking_mode=
  "chat"` closes reasoning with a bare `</think>` right after the assistant
  marker, and `render_chat` still fails loudly if a `<think>` is left open. On
  the dev model (Qwen, whose Jinja template defaults to thinking on) this bug
  produced a 0%-accuracy run whose every reply was `"Thinking Process: 1."`;
- **the post-filler tail** — the exact tokens between the last dot and the
  generation point, all of which 20 reads (the last one is where the answer is
  predicted). On V4 Flash it is `Answer : <｜Assistant｜> </think>`; on Qwen it
  is 12 tokens including the `<think></think>` pair that turns reasoning off;
- **the numeric decode mode** — `exact` on V4 Flash (301 single-token numerals
  in 0..300), which is what makes this the model for headline numbers; the dev
  model only gets `prefix`;
- **BOS handling** — the encoder renders `<｜begin▁of▁sentence｜>` into the
  prompt text, and jlens's own `encode` would tokenize with special tokens (and
  forces `add_bos_token` on), risking a second BOS. `common.load_model`
  overrides the lens's `encode` to tokenize the rendered text verbatim, exactly
  as generation does, and `apply_lens` checks the token count.

## 2. Pick a box

**4×H100 80GB (320 GB)** is the default: 2× the weights, ~$11/hr on Vast at the time
of writing. Go to 8×H100 only if the preflight says the experts upcast.

**2×H200 141GB (282 GB)** is the cheapest box that fits when H100s are scarce: same
Hopper chip as the H100 (native FP8, FP4 experts upconverted in-kernel), and 282 GB
leaves ~120 GB for activations over the 160 GB checkpoint. Only worth going to 4×H200
if the preflight shows an upcast, and even 564 GB is tight in that case (see the table).
**A100s are not an option** whatever the price — Ampere has no FP8 units, transformers'
FP8 loader refuses compute capability < 8.9, and a bf16 fallback is ~568 GB.

- **Rent on-demand, not "interruptible" / "spot".** Vast (and most clouds) offer
  the same machine two ways: *on-demand*, where you pay the listed rate and keep
  the machine until you stop it, or *interruptible* (elsewhere called *spot* or
  *preemptible*), where you bid a lower price — often half — but the host can
  take the machine back at any moment, without warning. The discount is not
  worth it here: the weights live on the pod's local disk, so an interruption
  throws away the 160 GB download and the next instance starts it over, which
  costs more than the discount saved on a run this short.
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
python 00_smoke_test.py --model deepseek --tokenizer-only  # template + tail + decode mode, no weights
python 00_smoke_test.py --model deepseek     # FIRST: does the big lens load + apply? does the reply parse?
python 20_lens_readout.py --model deepseek --n 40 --k 10 --lens both   # then a short readout run
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

- **`trust_remote_code=True`** is already set — DeepSeek ships custom modeling code
  (transformers 5.16 also has a native DeepSeek-V4 class and uses that).
- **transformers 5.16.x crashes at load** with `'NoneType' object has no attribute
  'get'` from `quantizer_finegrained_fp8.update_tp_plan` — a regression in the FP8
  MoE loader, fixed on `main` but not released as of 2026-09-02.
  `common.patch_fp8_tp_plan_bug` (called by `load_model`) works around it; if a
  later transformers release fixes it the shim does nothing.
- **V4's residual is four parallel streams, not one.** The blocks use
  manifold-constrained hyper-connections: block outputs are `[B, S, hc_mult=4, 4096]`,
  and the model only collapses them to one vector where it reads them (inside each
  block, and via `model.hc_head` before the final norm). jlens assumes `[B, S, D]` and
  unembeds each stream separately, which surfaced as a `topk`/`int()` TypeError in the
  smoke test. `common.collapse_hyper_connection_streams` (called by `load_model`)
  collapses every recorded layer with `hc_head` before the lens sees it — the lens is
  d_model=4096, so it was fit on collapsed vectors. The README of the lens repo says
  "mHC residual, mHC coefficients detached" but not which collapse; `check_provenance`
  now prints every extra field in the lens file so a recorded choice can be matched.
  If it names a different one, change the single marked line in that function.
- **`kernels` must be installed** (`pip install "kernels>=0.16,<0.17"`, now in
  `requirements.txt`). transformers does not bundle the FP8 matmul: the first forward
  pass fetches the Triton kernel `kernels-community/finegrained-fp8` from the Hub
  through the `kernels` package, and without it the load succeeds but the smoke test
  dies at `finegrained_fp8_linear` with "finegrained-fp8 kernel unavailable". The
  fetch is a few MB into `HF_HOME` and happens once.
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
  what it had. Use `--lens both`: one model load, both lenses, and 30 prefers the
  single `_both.csv` over stale single-lens files of the same condition.
- **Read the probe lines before walking away.** 20 prints the prompt length, the
  tail tokens it will read and the model's first greedy reply, and stops if that
  reply has no integer in it or if a cached `answers_<tag>.csv` disagrees with the
  model (a leftover from a broken run would otherwise be reused silently).
- **Stale answer caches:** `answers_<tag>.csv` is reused across lens passes by
  design. After any prompt or template change, pass `--regen-answers` once.
- **Interpreting the heatmaps:** every readout row carries a shuffled-quantity
  control (`ctrl_*`, the same test against another example's A1/A2/sum); 21 and
  30 show it next to the real fractions. A quantity is only "decoded" where it
  beats that control — the any-layer-any-position aggregates saturate on noise
  even in exact mode (43 layers × k positions of argmax draws).
- **Don't expect the API and the local weights to agree exactly.** Same weights do not
  mean same outputs across serving stacks: MoE routing is batch-composition sensitive,
  and providers differ in kernels, tensor-parallel degree and attention backend. The
  useful move is to *measure* the agreement — `answers_<tag>.csv` from `20` and
  `results/fig2_raw.jsonl` from stage 1 share the seeded test set, so they join on
  `idx` for an agreement rate and a McNemar test.
- **Don't reach for an API** to save money on the lens work: APIs don't expose hidden
  states. The API is only useful for *behavioral* uplift checks (stage 1).
