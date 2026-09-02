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

## 4. Later runs: k=25, n=1500, and the k=0 tail

- **k=25 (n=300):** accuracy 46.3% (API 44.3%), uplift +11.3 vs +11.7 — the
  stage-1 plateau. Same retrieval band, same per-cell densities (A1/A2/sum
  3.5/3.6/1.4% vs 0.5% control), same wrong-example signature, identical tail
  table. Fifteen extra dots hold more copies of the same late retrieval and
  add nothing.
- **n=1500 (k=10):** 706 correct / 794 wrong; every n=300 number reproduces
  to the second decimal. Local vs API 47.1% vs 45.3% (p=0.015, 66 vs 40
  flips; a 1.7-point serving-stack difference), same-integer agreement 81%.
- **Parallel or serial?** (new metrics in `21`): depth onset A2−A1 has median
  −1 (k=10) / 0 (k=25) layers with [later, same, earlier] ≈ [33–40%, 11–20%,
  45–56%] — no ordering in depth; position A2−A1 is +0.35 dots at k=10 (n.s.)
  and −0.8 at k=25 (p≈0.04) — opposite signs, no left-to-right order; the
  top-10 dot-set overlap equals its chance level (Jaccard 0.05–0.15 vs
  0.05–0.10) — the two facts land on dots independently; both are in one
  vector's top-10 in 19–24% (k=10) / 45–48% (k=25) of examples.
- **The k=0 tail reads the same** (`--k 0` now reads the tail). Same 300
  examples, best layer, correct examples: A2 at the answer token 68% (k=0) vs
  71% (k=10), sum 88% vs 91%. Retrieval is a property of any post-question
  position, not of the dots. What the filler changes is A1 availability at
  the answer position (top-10: stable-right 59→69%, rescued 50→76%,
  stable-wrong 40→66%) — but stable-wrong examples have both operands
  present at k=10 and still fail (sum rank 26), so A1 availability is not
  sufficient, and the failure is in a step the lens does not resolve.

## 5. Causal tests (`50_filler_patching.py`, figure `causal_tests_deepseek_dots-10.png`)

Every condition below edits ONLY the ten dot positions during the prefill
and greedy-generates; n=300, Wilson 95% CIs (±5.5 points), predictions
written in the script's docstring before the run. Plumbing: self-patch
reproduced clean on 100% of examples; clean reproduced the readout run's
answers on 300/300; the control swap sat at clean accuracy.

| intervention on the dots | accuracy | prediction changed | answer became donor-derived |
|---|---|---|---|
| clean | 46.3% | — | 0.3% (chance) |
| donor residual @30 / @35 / @39 | 44.7 / 44.0 / 47.0% | 26 / 23 / 14% | 0.7% |
| donor residual, layers 33–40 at once | 45.0% | 28% | 0.7% |
| **donor residual, all 42 layers** | **42.7%** [37–48] | 30% | 0.0% |
| J-swap A1→A1′ (α=1 / α=2) | 46.3 / 46.3% | 8 / 19% | 0.3% |
| J-swap A2→A2′ | 45.7% | 10% | 0.3% |
| J-swap control pair | 46.7% | 8% | 0.3% |
| J-ablate A1 / A2 / both | 46.3 / 44.0 / 45.0% | 7 / 10 / 8% | 0.3–0.7% |

(k=0, no filler at all: 35.0%.)

**Result.** Nothing done to the dots' content moves accuracy more than
~4 points, and no intervention transfers an operand: 2 donor-derived
answers in 1,800 patched generations against a 0.7% chance floor. This holds
for whole-position patches at single layers, across the whole retrieval band
(no layer left to heal the patch), and at every layer — where the dots are
entirely another question's computation and accuracy is still 42.7%, with
the 42 filler-rescued examples still correct 76% of the time. The J-lens
coordinate swaps and ablations are null as a corollary: if replacing
everything on the dots does not change the answer, replacing one
lens-readable coordinate cannot. (The J-null therefore says nothing about
whether computation is "hidden" from the verbalizable subspace — that
question only arises once a read exists.)

**The dots are not completely inert.** With the donor's dots at all layers,
30% of predictions change, and those changes are graded, not random: they
move toward the donor's sum in 77–80% of cases (56% for a shuffled-donor
control) and regress on the donor's operands, Δpred ≈ 0.25·ΔA1 + 0.34·ΔA2
(r = 0.31 and 0.54). The answer position blends in roughly a quarter to a
third of what the dots say about the operands — more A2 than A1, the
opposite of the pre-registered guess — for a minority of examples. A leak,
not a channel.

**Conclusion.** On V4 Flash, dots k=10, 2-fact addition: the operands are
decodable on the dots (the paper's picture), the same retrieval happens at
every post-question token, and the answer position does not depend on the
dots' content — the 11-point uplift is carried by the dots' *presence*, not
by computation performed on them. "Hidden computation across filler tokens"
is not what produces the uplift in this model; what the presence of extra
positions does (attention structure, distance, sink capacity) is the open
question this leaves.

Caveats: single model, single task, single filler type; the J-lens was fit
on pretraining text at positions ≤128; the `hc_head` collapse of the
four-stream residual is our choice; the J-swap's effectiveness on the
residual was not verified with a lens readout under the edit (the `lenscheck`
condition exists but was not run) — immaterial to the conclusion, since the
wholesale patches are null, but the write-up should say "assumed", not
"verified".

**Audit of the causal code (done before the write-up).** Three properties
of the run itself rule out a silent no-op or a wrong-position patch: the
self-patch (own residual back into the dots) reproduced clean on 300/300
while donor patches changed 14–30% of answers, so the hook writes into the
live computation and the captured residuals match the generate prefill; the
changes are dose-dependent (J-swap α=2 changes 19% vs 8% at α=1; later
single-layer patches change fewer answers, 26% at layer 30 vs 14% at 39);
and what changes carries the donor's numbers — among changed answers, Δpred
correlates 0.55–0.65 with the donor's ΔA2 at every patch layer, which no
misplaced or inert patch could produce. Offline: every donor has disjoint
elements and a different sum, clean predictions equal the readout run's
cached answers, the dot positions are the ones the readouts decode operands
at, and the model's config gives every attention layer a dense 128-token
sliding window, so the answer position sees the dots directly at all layers.
One hardening came out of the audit: `generate()` infers an attention mask
from `input_ids != pad_token_id` whenever pad differs from the generation
config's eos, and our prompts contain the eos token (it closes each few-shot
turn); on V4 Flash pad == eos == 1 so no mask was inferred, but every
generate call now passes an explicit all-ones mask. The script also now
records the full-vocabulary rank of each candidate answer at the first
generated token — the paper's own metric — for future runs.

**Reconciliation with the paper's KV transplants.** The paper's causal
evidence (DeepSeek V3) is a *rank* effect: transplanting the donor's filler
KV moves the donor answer's rank from ~90 to ~15 (2-fact, k=50; 1-fact,
k=100), "most changes corrupt the target's own answer rather than fully swap
it", full swaps are 13% only at k=100, the effect is "stronger for longer
filler", and accuracy under transplant is not reported. Our k=10 result is
the same phenomenon at the other end of that dose curve: the donor's
operands leak into the answer (slope 0.1–0.3 on ΔA2, 30% of answers change),
full swaps do not occur (0.3%), and — the number the paper did not measure —
accuracy holds. The dots' content is causal in the paper's graded sense and
not what carries the uplift.

## 6. Suggested next runs on the box

- Run `50_filler_patching.py --conditions lenscheck --out-suffix _lenscheck`
  (~5 min) to verify the J-swap moved the lens coordinate; rerun the band
  conditions to get the donor-answer *ranks* (now logged) for a direct
  comparison with the paper's 90 → 15.
- k=50 or k=100 with the same script: the paper's effect grows with k; ours
  is at k=10, where the uplift has already plateaued.
- Ablate the dots' *presence* rather than their content: attention-mask the
  dots out at the answer position (or at all positions) while keeping them in
  the prompt — if accuracy falls to the k=0 level, the benefit is attention
  to those positions (sink/structure), not their content.
- `--filler counting --k 10` for generality; a second model with a published
  lens (Qwen3.5-27B/122B-A10B) to check this is not V4-specific.
