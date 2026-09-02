# DeepSeek V4 Flash run report — stages 2–3, dots k=10, n=300 (2026-09-02)

Run on a rented 2×H200 (Vast.ai, 282 GB VRAM, 2.7 s/example):
`20_lens_readout.py --model deepseek --n 300 --k 10 --lens both`, then
`--k 0` (answers only), `21_analyze_readout.py`, `30_compare_lenses.py`,
`22_agreement_check.py`. Outputs in `results/*deepseek*`. Decode mode: `exact`
(301 single-token numerals), so the paper's criterion applies as written.

## 0. What it took to get V4 Flash running (all in `run_deepseek.md` §5)

Four things the dev model never showed: V4 ships no Jinja chat template
(prompts come from `encoding/encoding_dsv4.py`, `thinking_mode="chat"`);
transformers 5.16.x crashes in its FP8 loader unless `common.patch_fp8_tp_plan_bug`
is applied; the FP8 matmul needs `pip install "kernels>=0.16,<0.17"`; and V4's
residual is four hyper-connection streams `[B, S, 4, 4096]`, which
`common.collapse_hyper_connection_streams` collapses with the model's own
`hc_head` before the lens reads them (verified: at the lens's target layer the
logit lens then reproduces the model's own next-token distribution).

## 1. Local weights vs the API (22_agreement_check.py)

| k | local acc | API acc | same integer | McNemar p |
|---|---|---|---|---|
| 0 | 35.0% | 32.7% | 75.7% | 0.21 |
| 10 | 46.3% | 44.7% | 78.7% | 0.38 |

Uplift k=0→10: **+11.3 points locally** (42 wrong→right, 8 right→wrong,
p = 1.2·10⁻⁶) vs +12.0 via the API (46 / 10, p = 1.2·10⁻⁶); gap −0.7 points.
No unparsed replies; wrong answers are off by a median of 8 (the model
computes, it does not echo an operand). The mechanism read out below is the
one stage 1 measured.

## 2. The paper's Figure 3 on V4 Flash (logit lens)

139 correct / 161 wrong examples. Per-cell chance (shuffled-quantity control)
is 0.5%; the control row of `fig3_deepseek_dots-10_logit.png` is black.

- **Operands are retrieved on the filler dots**, in a band at layers 34–40 of
  42: A2 up to 70% of correct examples at its best position, A1 up to 60%;
  99% of correct examples decode each operand somewhere in the filler (control
  27–29%, the any-of-420-cells criterion being generous). Median first layer
  34 for both; A1 and A2 co-decode at one layer in 95% of examples.
- **The sum appears only after the filler**: 3% max in the filler region at any
  layer below 38, then 19% → 47% → 52% → 91% at the answer position over
  layers 38–41. Median first layer in the filler, where it appears at all: 38.
- **Wrong examples: retrieval without composition.** A1/A2 decoded in the
  filler at 96% / 93% (vs 99%), the sum at the answer position at ~5% (vs 91%).
- **Position striping.** At layer 35, A2 is decoded at filler positions
  0, 2, 4, 6, 7 (18–52%) and barely at 1, 3, 5, 8 (≤ 4%); at layer 37, A1 peaks
  at positions 1, 4 and the last dot. The two facts occupy different dots —
  consistent with parallel, position-distributed retrieval. Worth confirming
  at n=1500.
- Nothing is decoded below layer ~30 in either lens; the whole computation is
  late.

## 3. J-lens vs logit lens

**At the paper's criterion (top numeric token = value): a null.** The
per-layer max-decode curves in `jlens_vs_logit_deepseek_dots-10.png` coincide
for A1, A2 and sum; difference maps are within ±0.06; the summaries agree
(sum first layer 36.5 vs 38, per-cell sum 1.7% vs 1.5%, controls 0.3–0.4%).
The J-lens's *controls* are higher on the any-in-filler statistic (37–42% vs
27–29%), i.e. it has a stronger generic numeric bias, so raw fractions favour
it slightly for the wrong reason; per-cell excess over control is the fair
comparison and is equal within noise.

**With a rank criterion (value in the top-10 numeric tokens at the best
filler position): a modest, real edge, late.**

| layer | A2 top-10, logit | A2 top-10, J-lens | sum top-10, logit | sum top-10, J-lens |
|---|---|---|---|---|
| 34 | 56% | 32% | 1% | 0% |
| 35 | 76% | 87% | 1% | 4% |
| 36 | 76% | 85% | 7% | 12% |
| 38 | 43% | 67% | 7% | 41% |
| 39 | 16% | 18% | 35% | 48% |

The J-lens sees the composition about one layer earlier and more sharply
(layer 38: 41% vs 7%), and holds the operands at median rank in the low
thousands below layer 33 where the logit lens has them at tens of thousands —
coherent but not informative, as on the dev model. Interpretation: on this
task the computation lives in layers 33–40 of 42, where `J_l` is close to the
identity, so the J-lens's advantage (readable mid-stack residuals) has little
to act on. The J-lens does not reveal hidden computation the logit lens misses
here; it sharpens the late picture. The only J-lens-only trace in the filler
is A2 at layers 5–20 at 4–5% vs a 3% control — unresolved at n=300.

## 4. Suggested next runs on the box

- `--n 1500 --k 10 --lens both` (~70 min): paper scale, ~800 wrong examples
  for the signature, resolves the position striping and the early-layer A2
  trace.
- `--n 300 --k 25` (~15 min): does the lens picture track the uplift plateau
  stage 1 saw at k=25?
- Add rank-based curves (top-10) to `30_compare_lenses.py` so the comparison
  in §3 comes out of the pipeline.
