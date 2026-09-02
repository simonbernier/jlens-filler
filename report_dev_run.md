# Dev-model run report — Qwen3.5-4B, stages 2–3 (2026-09-01)

What was run, on the desktop's RTX 4070 Super, through `run_qwen_pipeline.bat`
(log: `results/pipeline_log.txt`): `00_smoke_test.py` → `20_lens_readout.py
--model dev --n 300 --k 10 --lens both --regen-answers` → `21_analyze_readout.py`
→ `30_compare_lenses.py` → `00_smoke_test.py --model deepseek --tokenizer-only`.
Outputs are in `results/`: `lens_readout_dev_dots-10_both.csv`,
`answers_dev_dots-10.csv`, `fig3_dev_dots-10_{logit,jlens}.png`,
`jlens_vs_logit_dev_dots-10.png`, the two summary JSONs, and
`deepseek_preflight_log.txt`. The whole readout takes about 25 minutes.

## 1. Bugs found and fixed

**Reasoning mode was on.** The first run of 20 (its `answers_dev_dots-10.csv`
was still on disk) scored 0%: all 300 replies were `"Thinking Process:\n\n1."`.
Qwen3.5's chat template opens a `<think>` block in the generation prompt by
default, and `answer()` decoded with `skip_special_tokens=True`, which stripped
the `<think>` tag so `parse_answer`'s reasoning guard never fired and the list
marker `1` was scored as the prediction. Every heatmap from that run was
therefore a readout of a model about to reason, split into 300 "wrong"
examples. Fix: one shared `paper_tasks.render_chat` that turns thinking off
(`enable_thinking=False` for Qwen; `thinking_mode="chat"` for DeepSeek V4, see
§3) and raises if a `<think>` is left open; `answer()` keeps special tokens;
the probe cell in 20 prints the first real reply and refuses to start if it
does not parse, or if a cached answers file disagrees with the model. The
smoke test now also generates one answer on the task prompt and checks it.

**The post-filler tail was cut short.** `POST_TAIL_MAX = 8` read the first
eight tokens after the filler. With reasoning off, Qwen's tail is 12 tokens
(`\n Answer : <|im_end|> \n <|im_start|> assistant \n <think> \n\n </think>
\n\n`), so the last four — including position −1, the token the answer is
actually predicted from — were never read. 20 now reads the whole tail; the
probe prints it.

**Lens and generation tokenized differently (DeepSeek only).** jlens's
`encode` tokenizes with `add_special_tokens=True`; the rendered DeepSeek prompt
already starts with `<｜begin▁of▁sentence｜>`, so the lens would have run on a
sequence with a second BOS while `answer()` generated from the single-BOS one.
`common.load_model` now makes the lens tokenize the rendered text verbatim and
`apply_lens` checks the token count against the prompt's. No effect on Qwen.

**Stale readouts were silently merged.** `find_readouts` merged every CSV of a
condition; a `_both.csv` next to an older `_logit.csv` would have had the old
logit rows win. It now prefers the `both` file and says so.

**No chance level.** "Decoded" is an argmax over ten digit tokens (prefix
mode) or ~300 numeric tokens (exact mode), and the summary's
decoded-anywhere-in-filler statistic is an `any()` over 31 × 10 cells — it
reads 1.0 for every quantity on a model that gets 1% right (see §2). Each row
now carries `ctrl_*`: the same test against the previous example's A1/A2/sum.
21 draws it as a third heatmap row, 30 dots it under the per-layer curves, and
the printed summary shows it next to every fraction. Related, 30 now compares
over all examples (and says so in the title) when fewer than 20 are correct,
instead of drawing maps from three.

## 2. Results on Qwen3.5-4B (dots, k=10, n=300, prefix decode mode)

**The dev model does not do the task.** Greedy accuracy with reasoning off is
1.0% (3/300), against 45% for DeepSeek V4 Flash through the API at the same k
(`results/fig2_summary.csv`). It replies with the first operand's atomic
number 27% of the time and the second's 15%. So on this model there is no
correct/wrong split to study and no mechanism to read out; it is a pipe-cleaner
only, as `config.py` already says. (`22_agreement_check.py` would confirm the
same thing against `fig2_raw.jsonl`; a local k=0 run is not worth the GPU time
at this accuracy.)

**What the lenses see, per cell (fraction of examples whose top numeric token
is the quantity's leading digit; control in parentheses):**

| region | lens | A1 | A2 | sum |
|---|---|---|---|---|
| filler | logit | 0.113 (0.086) | 0.075 (0.065) | 0.046 (0.050) |
| filler | J-lens | 0.112 (0.088) | 0.079 (0.070) | 0.041 (0.042) |
| post | logit | 0.120 (0.092) | 0.116 (0.101) | 0.173 (0.179) |
| post | J-lens | 0.125 (0.094) | 0.116 (0.099) | 0.135 (0.140) |

A1 sits a few points above chance in both regions (the model's habit of
answering A1), A2 barely, the sum nowhere. The per-layer curves in
`jlens_vs_logit_dev_dots-10.png` put the A1 and A2 signal entirely in layers
25–30, identically for both lenses; the sum curve is flat at 0.57 for both
lenses *and both controls* — that is simply the fraction of 2-fact sums that
start with `1`, which the model's standing preference for the digit `1`
matches at almost every cell. Without the control that curve would read as
"the sum is decoded at every layer".

**J-lens vs logit lens.** On the decode-fraction metric they are the same on
this model. Where they differ is coherence, not content: at position −1 the
J-lens brings the median full-vocab rank of the A1/A2/sum digits down to ~25 by
layer 20, while the logit lens has them at rank 10⁵ until layer 28 — the
transported mid-stack residual already "knows a digit is coming", but no
better which digit than chance. Two logit-lens artefacts are worth knowing
about before the DeepSeek run: at layers 1 and 3 the raw residual of the final
`\n\n` token puts `1` in the top 10 for 56% of examples (= the `1`-first sums),
and the J-lens smoke-test table shows the usual early-layer garbage for the
logit lens where the J-lens is already sensible. Neither is a bug; both are
why the control row exists.

**Verdict on the "very different" worry:** the two lenses agree wherever the
model has a signal (layers 25–30, A1/A2) and disagree only in the noise floor,
where the logit lens is garbage and the J-lens is generically numeric. That is
the expected shape, not a transport bug (the smoke test's disagreement check
also passed: 28/31 layers differ).

## 3. DeepSeek V4 Flash: what changes and what was verified

Verified on the real tokenizer with `00_smoke_test.py --model deepseek
--tokenizer-only` (`results/deepseek_preflight_log.txt`), no weights:

- **V4 ships no Jinja chat template** — `tokenizer.chat_template` is `None`,
  and `apply_chat_template` raises. The old `render_chat` would have crashed
  at the first prompt on the rented box. The prompt format lives in
  `encoding/encoding_dsv4.py` in the weights repo (standard library only);
  `paper_tasks.deepseek_v4_encoder` downloads that file and `render_chat` calls
  its `encode_messages(msgs, thinking_mode="chat")` whenever the tokenizer has
  no template. Rendered prompt: `<｜begin▁of▁sentence｜>{system}<｜User｜>…
  <｜Assistant｜></think>` — reasoning off, 270 tokens at k=10.
- **Post-filler tail** read by 20: `Answer : <｜Assistant｜> </think>` — four
  tokens, position −1 is the `</think>` the model answers after.
- **Decode mode is `exact`**: 301 single-token numerals in 0..300, so the
  paper's criterion applies as written and the headline numbers should come
  from here.
- **BOS**: the encoder renders it; the lens tokenizes verbatim (see §1).
- The reasoning bug is absent on DeepSeek by construction (`thinking_mode="chat"`
  is the only path), and `render_chat` still checks for an open `<think>`.

Not verifiable without weights: model load (§0–1 of `run_deepseek.md`
unchanged), and whether `jlens.from_hf` finds the text decoder inside
DeepSeek's remote modeling code — the smoke test on the box is the first thing
to run for that reason.

## 4. Files touched

`paper_tasks.py` (`render_chat`, `deepseek_v4_encoder`), `common.py` (encode
override, token-count check), `20_lens_readout.py` (probe, full tail, controls,
`POS_CHUNK=32`), `21_analyze_readout.py` and `30_compare_lenses.py` (control
rows/curves, small-n fallback), `lens_analysis.py` (`has_control`, control
statistics, `both`-file preference), `00_smoke_test.py` (greedy-answer check,
`--tokenizer-only`), `40_attention_study.py` (uses `render_chat`),
`run_qwen_pipeline.{bat,py}` and `run_deepseek_preflight.bat` (new),
`README.md`, `run_deepseek.md`, `setup_env.sh` (next-steps text).

Loose ends: `results/lens_readout_dev_dots-10_logit.csv` is the 0%-accuracy
readout from the reasoning-mode run and can be deleted; the README's stage-1
table still names `10_build_fig2_dataset.py` / `11_run_fig2_sweep.py` while
the folder has `build_filler_uplift_dataset.py` / `run_filler_uplift_sweep.py`.
