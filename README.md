# Filler tokens × Jacobian lens

Why do filler tokens improve LLM math performance? Replication of
*Reading Between the Dots* (Brauer, Verdun & Marks, arXiv:2607.03502) on
**DeepSeek V4 Flash**, extended with the **Jacobian lens (J-lens)** — a readout
tool the paper predates and does not use.

The code is numbered by **study stage**, in the order the work happens.
Stage 1 is API-only (no GPU); stages 2–4 need local weights on a rented GPU box.

| stage | scripts | what | where |
|---|---|---|---|
| 0 | `00_smoke_test.py` | prove the machine works (API path or GPU/lens path) | anywhere |
| 1 | `10_build_fig2_dataset.py` → `11_run_fig2_sweep.py` | **Fig. 2**: accuracy vs filler length k, paper-scale n | API (no GPU) |
| 2 | `20_lens_readout.py` (`LENS="logit"`) → `21_analyze_readout.py` | **Fig. 3**: the paper's logit-lens picture on V4 Flash | GPU box |
| 3 | `20_lens_readout.py` (`LENS="jlens"`) → `30_compare_lenses.py` | **the new result**: does the J-lens see more than the logit lens? | GPU box |
| 4 | `40_attention_study.py` | attention study as in the paper (optional, code ready) | GPU box |

Shared modules (no numbers = not run directly):
`config.py` (model + lens registry), `common.py` (model/lens loading, provenance
guard, lens application, and the two guards below), `paper_tasks.py` (torch-free
task library: prompts, fixed test set, numeric-token utils, McNemar), `api_common.py` (OpenRouter
client, provider pin, reasoning-off), `lens_analysis.py` (readout loading +
"what algorithm?" aggregation for 21/30). `ANALYSIS.md` explains how to read
every output and what a J-lens advantage (or null) would mean;
`run_deepseek.md` covers renting the GPU box.

## Get the code on a new machine
```bash
git clone https://github.com/simonbernier/jlens-filler.git
cd jlens-filler
bash setup_env.sh
```
That one command is the whole bootstrap on every machine — a rented GPU box, a
Windows laptop under Git Bash, macOS. Nothing in `results/` is tracked, so each
machine regenerates its own outputs; the stage-1 scripts are resumable, so you
can copy `results/knowledge_check.jsonl` and `results/fig2_raw.jsonl` over by
hand to continue a sweep started elsewhere.

Credentials are never committed. Either export them per machine, or drop a
`.env` in the repo root (gitignored) — `python-dotenv` loads it, and VS Code
reads it too:
```bash
HF_TOKEN=...                   # gated weights + lens repo (or: huggingface-cli login)
OPENROUTER_API_KEY=sk-or-...   # stage 1 (10/11 go through OpenRouter)
```

## Install
`setup_env.sh` detects the machine it is on and needs no flags:

| it checks | GPU box (no conda) | local box with miniconda |
|---|---|---|
| **environment** | `python -m venv .venv` | `conda create -n jlens-filler python=3.11` |
| **torch** | CUDA wheel matched to the driver (`cu118`/`cu121`/`cu126`/`cu128` from `nvidia-smi`) | CUDA wheel if you have an NVIDIA GPU, else the CPU-only wheel |
| **bitsandbytes** | installed (needed for the 4-bit DeepSeek load) | skipped when there is no CUDA, so the rest of the install still succeeds |

It also clones + `pip install -e`s `anthropics/jacobian-lens`, installs `openai`
(stage 1) and `ipykernel`/`ipython` (every numbered script except `00` and `40`
is a `# %%` cell notebook; the kernel is registered as *Python (jlens-filler)*),
logs in to HF if `HF_TOKEN` is set, and finishes with a verification block printing the torch
version, each GPU's name and VRAM, and an import check for every package
including `jlens`. The last thing it prints is the **interpreter path** — paste
that into VS Code if it doesn't autodetect the env.

On Windows it finds a miniconda that is not on Git Bash's `PATH` (it looks in
`%USERPROFILE%`, `AppData\Local` and `C:\ProgramData`) and sources conda's own
`profile.d/conda.sh`, so `conda activate` works inside the script.

Re-running it is safe: an existing env is reused, an existing `jacobian-lens/`
checkout is fast-forwarded, and an already-importable torch is left alone —
which is what you want on a rented image that ships a tuned build.

Overrides, if a box needs one:
```bash
ENV_BACKEND=conda|venv|system   # force the environment backend (default: auto)
ENV_NAME=jlens-filler           # conda env name
PY_VERSION=3.11                 # python for a freshly created conda env
TORCH_VARIANT=auto|cuda|cpu|skip
CUDA_TAG=cu128                  # pin the PyTorch CUDA index
FORCE_TORCH=1                   # reinstall torch even if it imports
SKIP_BNB=1                      # skip bitsandbytes
```

After setup, activate the env in new shells with `conda activate jlens-filler`
(or `source .venv/bin/activate` — `.venv/Scripts/activate` on Windows).

### VS Code
`.vscode/settings.json` is tracked and pins the interpreter
(`${userHome}/.conda/envs/jlens-filler/python.exe`), turns on terminal
auto-activation, loads `.env` so the API key reaches stage 1, and sets the
notebook cwd to the repo root so the `# %%` scripts resolve `data/` and
`results/` the same way a plain `python 20_...` does. If VS Code doesn't list
the env, refresh *Python: Select Interpreter* or use *Enter interpreter
path...* with the path the setup script printed (on Windows the env lands in
`%USERPROFILE%\.conda\envs` and the interpreter is `...\jlens-filler\python.exe`
in the env root — no `bin/`).

## Stage 0 — smoke test
```bash
python 00_smoke_test.py --api               # API path: key, endpoints, provider pin, parsing
python 00_smoke_test.py                     # GPU path on the dev model (Qwen3.5-4B)
python 00_smoke_test.py --model deepseek    # GPU path on the real target (GPU box)
```
The GPU path also validates prompt construction + filler-span location on the
real tokenizer, then asserts the J-lens and logit-lens actually differ
somewhere (a no-op transport step would silently fake a null result).

## Stage 1 — Figure 2 (accuracy vs k, API only)
Paper-faithful Fig. 2 using Ryan Greenblatt's compose_facts fact files (paper
Appendix A), dots filler, k ∈ {0, 5, 10, 25, 50, 100}, paper-scale
n (800 1-fact / 1500 2-fact). Both scripts are `# %%` notebooks for VS Code;
run 10 top-to-bottom, then 11. Both are resumable (caches in
`results/*.jsonl`).
```bash
export OPENROUTER_API_KEY=sk-or-...
# 10: knowledge check (~1.5k calls, cached) -> filter -> few-shot holdout
#     -> data/fig2_{1,2}fact.jsonl + fig2_meta.json
# 11: sweep (6 k x (800+1500) examples = 13.8k calls, cached) ->
#     results/fig2_summary.csv + results/fig2_accuracy_vs_k.png
```
Everything both scripts must agree on (model id, provider pin, reasoning OFF —
V4 Flash is a hybrid reasoning model) lives in `api_common.py`; 11 asserts the
datasets were built with the same pin it is about to sweep with. The built
datasets are `.jsonl`, hence gitignored by design: they depend on the knowledge
check of whichever model/API you ran, so each machine rebuilds its own.
Growing n later is cheap: test-set sampling is sequential in a seeded rng, so
earlier examples (and their cached results) stay valid and only new examples
cost calls. 11's figure also overlays the paper's own dot-filler curves for
DeepSeek V3 and Kimi K2 (transcribed from `plotting/plot_filler_accuracy.py`
in github.com/kaleybrauer/filler-token-reasoning) — context for how V4 Flash's
uplift compares, with the different-pipeline caveats noted in the script. 1-fact holds out 5 facts for few-shot; 2-fact holds out 10
elements → 5 pairs (paper Appendix A); same fixed test set at every k so
McNemar applies.

## Stage 2 — the paper's logit-lens picture (GPU box)
20, 21 and 30 are `# %%` notebooks like 10/11: open one in VS Code, edit the
**Config** cell, run top-to-bottom. In 20 the model load is its own cell, so you
can re-run the readout loop without paying for it again; 21 and 30 default to
`TAG = ""`, which picks up whatever 20 wrote last, so the usual loop is *run 20,
run 21, look at the figure*. Every one of them still runs headless on a rented
box, where the Config cell's defaults become CLI flags:
```bash
python 20_lens_readout.py --model dev --n 40 --k 10           # pipe-clean first
python 20_lens_readout.py --model deepseek --n 300 --k 10 --lens logit
python 21_analyze_readout.py                                  # newest condition
python 21_analyze_readout.py --tag deepseek_dots-10           # or name one
```
**The dev model runs on a 12 GB card.** Applying a J-lens needs no backward pass
— the `J_l` matrices are pre-fitted and `lens.apply` is under `torch.no_grad()` —
so VRAM is just weights, exactly like a plain logit lens. Qwen3.5-4B is 9.3 GB of
bf16 on HF, of which ~1.3 GB is a vision encoder the lens never touches;
`offload_vision` (default on, see `common.offload_vision_tower`) parks it in host
RAM, leaving ~8.0 GB on the GPU and ~2 GB of headroom on an RTX 4070 Super.
`load_model` prints VRAM after loading — read that before starting a long run.
Don't quantize the dev model to buy headroom: the `J_l` were fit in the bf16
residual basis, so 4-bit weights change what the lens reads.

**Context window.** jlens's `encode` truncates at 512 tokens *from the right* by
default, and our readout positions are negative indices into the untruncated
text — so a long prompt would not error, it would silently shift every position
and produce a plausible, wrong heatmap. Five few-shot examples plus k=25 dots is
already past 512. Every lens call in this repo goes through `common.apply_lens`,
which sizes the window to the prompt (`MAX_SEQ_LEN = None`) and raises rather
than truncate if you pass an explicit value that is too small.

20 greedy-generates each answer (correct/wrong split) and records, per
(layer, position), the top numeric token, whether each of A1/A2/sum is decoded
there, and their ranks. It adapts the paper's numeric-decode criterion to the
model's tokenizer — exact match where digits are grouped into single tokens
(DeepSeek), first-token match where they are split (Qwen, Llama 3) — prints
which mode it is in, and records it in the CSV and every figure title. Headline
numbers should come from an exact-mode run; `00_smoke_test.py` tells you which
mode a model gives you before you spend GPU hours. 21 turns
that into Figure-3-style heatmaps + the "what algorithm?" summary. The logit
lens is run through `jlens` with `use_jacobian=False`, so stage 3 is an exact
apples-to-apples upgrade. Use the **same `SEED` as stage 1** (default 0) so
behavioral and mechanistic results describe the same examples.

## Stage 3 — J-lens vs logit-lens (the new result)
```bash
python 20_lens_readout.py --model deepseek --n 300 --k 10 --lens jlens
python 30_compare_lenses.py                                   # newest condition
```
(Or run 20 once with `--lens both`.) 30 needs both lenses for one condition and
finds them by tag, so it picks up a single `both` file or a `logit` + `jlens`
pair without being told which; pass `--readout a.csv b.csv` to name them
explicitly. 30 makes the J-lens − logit-lens
difference maps and per-layer decode curves; `ANALYSIS.md` says what each
outcome means. Greedy answers are cached per condition
(`results/answers_<tag>.csv`), so the jlens pass reuses the logit pass's
generations.

## Stage 4 (optional) — attention study
```bash
python 40_attention_study.py --model dev --n 40 --k 10        # dev model first
python 40_attention_study.py --model deepseek --n 100 --k 10
```
Per (layer, head, query position): attention mass from filler/answer positions
onto the fact entities, the rest of the question, and the filler itself —
the paper's "what do filler tokens attend to?" analysis. Memory-hungry
(forces eager attention); see the script docstring.

## The one thing to get right: model ↔ lens matching
A Jacobian lens is a set of matrices in **one model's** residual-stream basis.
Applying it to a different model (or even base vs. instruct) gives nonsense.
Every lens `.pt` stores a `provenance.model_id`; `check_provenance()` warns if
it disagrees with `config.py`. If it warns, trust the provenance and edit the
`hf_id`.

## Compute
- **`dev` (Qwen3.5-4B):** runs on a small GPU, or slowly on CPU (set `dtype="float32"`).
- **`deepseek`:** 284B MoE — rented multi-GPU + 4-bit only. See `run_deepseek.md`.
- **Stage 1:** API only — no GPU. The paper found uplift robust to API vs local 4-bit.
- An **inference API cannot be used** for the lens work: the lens needs the
  residual stream, which APIs don't expose. White-box weights are required.

## Suggested next steps (beyond this scaffold)
1. Confirm the Fig. 2 shape on V4 Flash, then check the lens picture tracks the
   behavioral uplift across k (run 20 at k=10/25/50).
2. Add the **`sum` crystallization** readout comparison — 21/30 already report
   it: does the J-lens surface the sum earlier (in depth) or in-filler (in
   position)?
3. Stage 4's attention analysis, plus a KV-cache transplant at filler positions
   (causal check), for triangulation.
4. If V4 Flash shows weak 2-fact uplift (the paper saw only ~3 points on V3),
   the 1-fact task (54%→72% in the paper) is already in the stage-1 data.
