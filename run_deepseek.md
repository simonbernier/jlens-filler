# Scaling to DeepSeek V4 Flash (284B MoE) on rented GPUs

The small model is for developing the pipeline. Once `01`/`02` are clean on `dev`, flip
to the real target. **Nothing in the code changes except the `--model deepseek` flag** —
`config.py` already sets `device_map="auto"` + 4-bit for it.

## 1. Pick a box
DeepSeek V4 Flash is ~284B total params (MoE, ~a few dozen B active/token). Rough VRAM:
- **bf16:** ~570 GB → needs ~8×80GB (H100/A100 80GB) or more.
- **4-bit (nf4, as configured):** ~150–170 GB of weights → fits on **4×80GB**, comfortably
  on **8×80GB** (leave headroom: KV cache + the residual-stream capture the lens does).

On **Runpod**: a 4×H100 or 8×A100-80GB pod. On **Lambda**: an 8×A100/H100 instance.
Prefer a Secure Cloud / on-demand pod for a stable multi-hour session; spot is fine for
short sweeps if you checkpoint results to disk between prompts.

## 2. Storage
- Model weights: budget **300–600 GB** of pod disk (4-bit download is smaller but HF may
  pull bf16 shards first — check the snapshot).
- The lens repo `camilablank/workspace-lenses` is **~47 GB total**, but you only need the
  `deepseek-v4-flash/j-lens/lens.pt` file — `JacobianLens.from_pretrained(..., filename=...)`
  downloads just that one. Still allow **10–20 GB** for it (a 284B lens is large).
- Point HF caches at the big volume: `export HF_HOME=/workspace/hf`.

## 3. Environment
```bash
export HF_HOME=/workspace/hf
export HF_TOKEN=hf_...            # needed for deepseek weights
bash setup_env.sh
python 01_smoke_test.py --model deepseek     # first: does the big lens even load + apply?
python 02_filler_experiment.py --model deepseek --k 25
```
`accelerate` (installed) handles the multi-GPU sharding via `device_map="auto"`.

## 4. Gotchas
- **`trust_remote_code=True`** is already set — DeepSeek ships custom modeling code.
- **Layer coverage:** the lens only has `source_layers` in (typically) the model's second
  half. The scripts iterate `lens.source_layers`, so they adapt automatically.
- **Verify the snapshot:** if HF resolves `deepseek-ai/DeepSeek-V4-Flash` to a revision the
  lens wasn't fit on, `check_provenance()` will warn — pin `hf_id` to the provenance's exact
  `model_id`/revision.
- **Cost control:** start with 1–2 prompts and `k=10` to confirm it runs before a full sweep;
  the residual capture across all filler positions is the memory-heavy part.
- **Don't reach for an API** to save money here — APIs don't expose hidden states, so the
  lens can't run against them. The API is only useful for *behavioral* uplift checks.
