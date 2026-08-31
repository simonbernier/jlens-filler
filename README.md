# Filler tokens × Jacobian lens — project scaffold

Setup for investigating **why filler tokens improve LLM math performance**, using the
**Jacobian lens (J-lens)** and comparing it to the standard techniques from
*Reading Between the Dots* (Brauer, Verdun & Marks, arXiv:2607.03502).

The design lets you develop cheaply on a small model, then flip one config key to run
the real target, **DeepSeek V4 Flash** (284B MoE), on rented GPUs.

## Why this is a promising angle
The paper established the phenomenon and localized it using **attention analysis, the
logit lens, KV-cache transplants, and an unsupervised decoding pipeline** — but it
predates the J-lens and *does not use it*. The workspace-lenses repo now ships a J-lens
for DeepSeek V4 Flash and several smaller models. So the simple, cool first result is:
**take one paper task (2-fact addition), and show the J-lens reads the hidden
computation across the dots more cleanly / earlier than the logit lens.** The logit lens
is exactly the paper's main readout tool, and `jlens` gives it to you for free
(`use_jacobian=False`), so it's an apples-to-apples comparison in one codebase.

## Get the code on a new machine
```bash
git clone https://github.com/simonbernier/jlens-filler.git
cd jlens-filler
bash setup_env.sh
```
On a rented GPU box that is the whole bootstrap. Nothing in `results/` is tracked, so
each machine regenerates its own outputs; `03_accuracy_sweep.py` is resumable, so you
can copy a `results/accuracy_raw.jsonl` over by hand if you want to continue a sweep
started elsewhere.

Credentials are never committed — export them per machine:
```bash
export HF_TOKEN=...            # gated weights + lens repo (or: huggingface-cli login)
export DEEPSEEK_API_KEY=...    # only for 03_accuracy_sweep.py
```

## Install
```bash
bash setup_env.sh          # clones anthropics/jacobian-lens, pip installs, checks torch/cuda
pip install openai         # only for 03 (API accuracy sweep)
```

## Run order (the replication plan)
```bash
python tests/test_mock.py           # 0. no-GPU sanity: prompts, spans, ranks, aggregation
python 01_smoke_test.py             # 1. proves the lens pipeline works on Qwen3.5-4B
python 02_filler_experiment.py      # 2. quick bare-prompt mini-experiment (dev model)

# --- the paper replication on DeepSeek V4 Flash ---
export DEEPSEEK_API_KEY=sk-...
python 03_accuracy_sweep.py --n 300                       # Fig. 2: accuracy vs k (API)
python 04_lens_readout.py --model deepseek --n 300 --k 10 # Fig. 3 data: both lenses (GPU box)
python 05_analyze_lens.py --readout results/lens_readout_deepseek_dots-10.csv
```
`03` needs no GPU (behavioral only; the paper found uplift robust to API vs local
4-bit). `04` needs the rented GPU box from `run_deepseek.md` — develop with
`--model dev --n 40` first. `05` is pure pandas/matplotlib and runs anywhere.
`ANALYSIS.md` explains how to read the outputs and what would count as the J-lens
teaching us something new.

## Files
| file | what it is |
|---|---|
| `config.py` | model + lens registry (`dev` = Qwen3.5-4B, `deepseek` = DeepSeek V4 Flash, plus alternates). Device/4-bit settings live here. |
| `common.py` | loaders + the J-lens / logit-lens helpers. Reads each lens's `provenance` to guard against model↔lens mismatch. |
| `filler_tasks.py` | bare-prompt builders for the quick mini-experiment (`02`). |
| `paper_tasks.py` | paper-faithful task library (torch-free): chat prompts with few-shot filler, fixed test set, fillers incl. counting/scrambled, filler-span location, numeric-token utils, McNemar. |
| `01_smoke_test.py` | end-to-end sanity check on the small model. |
| `02_filler_experiment.py` | quick J-lens vs logit-lens mini-experiment (bare prompts). |
| `03_accuracy_sweep.py` | **Fig. 2 replication**: accuracy vs filler length k via the DeepSeek API; resumable; binomial SEs + McNemar. |
| `04_lens_readout.py` | **Fig. 3 data**: greedy answers + both lenses at every (layer, filler/post position); compact CSV. |
| `05_analyze_lens.py` | Fig. 3 heatmaps (correct/wrong × A1/A2/sum), J-lens−logit-lens difference maps, crystallization + parallelism stats. |
| `06_build_fig2_dataset.py` | **notebook-style (`# %%`)**: builds the Fig. 2 datasets from compose_facts — knowledge check (≥3/4 standalone trials), filtering, few-shot holdout, k-dots prompts. |
| `07_fig2_deepseek_api.py` | **notebook-style (`# %%`)**: runs 06's datasets against the API, saves raw/summary results, plots the two-panel Fig. 2. |
| `data/compose_facts/` | vendored fact files (age/atomic/static) from rgreenblatt/compose_facts. |
| `tests/test_mock.py` | fake-tokenizer/fake-lens tests of all plumbing; runs with numpy/pandas/matplotlib only. |
| `ANALYSIS.md` | what algorithm to expect, and what a J-lens advantage (or null) would mean. |
| `run_deepseek.md` | how to scale to DeepSeek V4 Flash on Runpod/Lambda. |

## Fig. 2 with compose_facts (dots only) — `06` + `07`
The paper-faithful Fig. 2 path using Ryan Greenblatt's compose_facts fact files
(paper Appendix A), replacing the hardcoded facts in `paper_tasks.py` for this
purpose. Both scripts are `# %%` notebooks for VS Code; run 06 top-to-bottom,
then 07. Both are resumable (caches in `results/*.jsonl`) and have a
`USE_MOCK = True` switch for a free end-to-end dry run.
```bash
export DEEPSEEK_API_KEY=sk-...
# 06: knowledge check (~2.5k calls) -> filter -> few-shot holdout -> data/fig2_{1,2}fact.jsonl
# 07: sweep (2 tasks x 6 k x 300 examples = 3600 calls) -> results/fig2_accuracy_vs_k.png
```
The built datasets are `.jsonl`, hence gitignored by design: they depend on the
knowledge check of whichever model/API you ran, so each machine rebuilds its own.
1-fact holds out 5 facts for few-shot; 2-fact holds out 10 elements → 5 pairs
(paper Appendix A). k ∈ {0, 5, 10, 25, 50, 100} dots; same fixed test set at
every k so McNemar applies.

## The one thing to get right: model ↔ lens matching
A Jacobian lens is a set of matrices in **one model's** residual-stream basis. Applying it
to a different model (or even base vs. instruct) gives nonsense. Every lens `.pt` stores a
`provenance.model_id`; `check_provenance()` warns if it disagrees with `config.py`. If it
warns, trust the provenance and edit the `hf_id`.

Second thing to get right: `03` and `04` must use the **same test-set seed**
(`--seed 0` default) so behavioral and mechanistic results describe the same examples.

## Compute
- **`dev` (Qwen3.5-4B):** runs on a small GPU, or slowly on CPU (set `dtype="float32"`).
- **`deepseek`:** 284B MoE — rented multi-GPU + 4-bit only. See `run_deepseek.md`.
- **`03_accuracy_sweep.py`:** API only — no GPU. ~300 examples × 13 conditions ≈ 4k calls.
- An **inference API cannot be used** for the lens work: the lens needs the residual
  stream, which APIs don't expose. White-box weights are required.

## Suggested next steps (beyond this scaffold)
1. Confirm the Fig. 2 shape on V4 Flash, then check the J-lens picture tracks the
   behavioral uplift across k (run `04` at k=10/25/50).
2. Add the **`sum` crystallization** readout comparison — `05` already reports it:
   does J-lens surface the sum earlier (in depth) or in-filler (in position)?
3. Bring in the paper's other tools for triangulation: attention to filler positions, and a
   KV-cache transplant at filler positions (causal check).
4. If V4 Flash shows weak 2-fact uplift (the paper saw only ~3 points on V3), switch
   `paper_tasks.py` to 1-fact addition (54%→72% in the paper) — the prompt format
   generalizes directly.
