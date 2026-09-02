# Filler tokens, the Jacobian lens, and DeepSeek V4 Flash: what the dots do

*Simon Bernier — MATS 12.0 (Neel Nanda stream), September 2026. Consolidated
results; run-level detail in `report_dev_run.md` and `report_deepseek_run.md`.*

## The question

Brauer, Verdun & Marks (*Reading Between the Dots*, arXiv 2607.03502) show that
appending filler tokens (rows of dots) between a two-fact arithmetic question
and the answer raises accuracy in models that answer without chain of thought,
and use a logit lens to argue that the operands are retrieved *on the dots*:
hidden computation across filler tokens. This project replicates that on
DeepSeek V4 Flash — a model the paper predates — and asks two things the paper
could not: does the Jacobian lens (J-lens; Gurnee et al. 2026), which reads
intermediate layers in the final-layer basis, see more than the logit lens?
And is the computation on the dots causally used?

Task, prompts and test set follow the paper's Appendix A (2-fact atomic-number
addition, five few-shot examples containing the same filler, k dots appended,
greedy decoding, reasoning off). n = 300 examples unless stated; the n = 1500
run reproduces every n = 300 number.

## 1. The behavioural effect is real and the local weights reproduce it

Through the API (stage 1, n = 1500), dots raise 2-fact accuracy from 35.4% at
k = 0 to 45.3% at k = 10 and plateau (46.3% at k = 25). On the rented weights
the same 300 examples give 35.0% → 46.3% → 46.3%, the same integer as the API
on 76–79% of examples, and an uplift of +11.3 points (42 wrong→right vs 8
right→wrong, McNemar p ≈ 10⁻⁶) against +12.0 via the API. Everything read out
below is read from a model that shows the effect the paper describes.

## 2. The paper's lens picture replicates

With the logit lens (the paper's tool) on the k = 10 prompts
(`fig3_deepseek_dots-10_logit.png`): both operands are decodable on the dot
positions in a band at layers 34–40 of 42 (A2 in up to 70% of correct
examples at its best dot, A1 up to 60%; chance per cell 0.5%, measured by
decoding another example's operands with the same criterion); the sum is not
on the dots — it appears at the answer position from layer 38 and reaches 91%
at layer 41; and in wrong examples the operands are still on the dots
(96%/93%) while the sum is absent (5% at the answer position). Retrieval of
the two facts is parallel — same layers, no ordering in depth or along the
dots, each fact landing on dots independently of the other — and nothing at
all is decodable below layer ~30. These are the paper's Figure 3 and its
"retrieval without composition" signature, on a newer model, with a chance
level the paper did not report.

## 3. The J-lens does not see more than the logit lens here

With the paper's criterion (top numeric token) the two lenses give the same
maps, curves and summary numbers. With a softer top-10 rank criterion the
J-lens is sharper at the peak (A2 87% vs 76% at layer 35; sum 41% vs 7% at
layer 38) but not earlier — the logit lens ramps up a layer or two sooner.
The reason is visible in the maps: on this task everything happens in the
last quarter of the network, where the J-lens's transport matrix is close to
the identity, so its advantage (readable mid-stack residuals) has nothing to
act on. The J-lens's only distinctive behaviour is coherence in early layers
— median operand rank in the low thousands where the logit lens gives tens of
thousands — which the control shows is generic numeric bias, not information.

## 4. The dots are not where the retrieval is special

Reading the tail tokens (`Answer`, `:`, `<|Assistant|>`, `</think>`) shows the
operands decoded there at the same rates as on the best dot — and, decisively,
a k = 0 prompt with no dots at all retrieves them at its own tail tokens just
as well (A2 at the answer token: 68% at k = 0 vs 71% at k = 10; sum 88% vs
91%). Retrieval is a property of every position after the question, not of
the filler. What the filler measurably changes is how often the *first*
operand is available at the answer position (top-10: 50% → 76% for the
examples the filler rescues) — but examples with both operands present still
fail at the same rate, so this is a correlate, not the mechanism.

## 5. Causal tests: the dots' content is not what the answer uses

Every intervention edits only the ten dot positions during the prefill and
lets the model generate (`50_filler_patching.py`,
`causal_tests_deepseek_dots-10.png`; n = 300, Wilson 95% CIs ±5.5 points;
predictions written before the run; self-patch reproduces clean on 100% of
examples).

| intervention on the dots | accuracy | predictions changed | answer became a donor-derived value |
|---|---|---|---|
| none | 46.3% | — | 0.3% (chance) |
| another question's residual at layer 30 / 35 / 39 | 44.7 / 44.0 / 47.0% | 26 / 23 / 14% | 0.7% |
| another question's residual at layers 33–40 together | 45.0% | 28% | 0.7% |
| another question's residual at **all 42 layers** | 42.7% | 30% | 0.0% |
| J-lens swap of the A1 coordinate → A1′ (α = 1 / 2) | 46.3 / 46.3% | 8 / 19% | 0.3% |
| J-lens swap of the A2 coordinate → A2′ | 45.7% | 10% | 0.3% |
| J-lens swap of two absent numbers (control) | 46.7% | 8% | 0.3% |
| J-lens ablation of A1 / A2 / both | 46.3 / 44.0 / 45.0% | 7 / 10 / 8% | 0.3–0.7% |

No filler at all: 35.0%.

Nothing done to the dots' content — up to replacing them, at every layer,
with the dots of a different question — moves accuracy more than ~4 points
toward the no-filler level, and no intervention transfers an operand: two
donor-derived answers in 1,800 patched generations, at the chance rate. The
examples the filler rescues stay correct 76% of the time with foreign dots.
The J-lens swaps and ablations are null as a consequence: if replacing
everything on the dots does not change the answer, replacing one coordinate
cannot. The dots are not inert — when they carry a foreign question, 30% of
answers shift, and the shifts regress on the foreign operands (slope ~0.1
over all examples, 0.25–0.34 among the changed ones): a graded leak the
answer position blends in, but not a channel it depends on.

This is consistent with the paper's own transplant numbers, read carefully:
their evidence is a *rank* shift of the donor answer (~90 → ~15 at k = 50–100)
with full swaps only 13% of the time at k = 100 and "most changes corrupt the
target's own answer", growing with k; accuracy under transplant is not
reported. At k = 10 we see the small end of that dose curve — the leak — and
add the measurement they did not make: accuracy does not depend on it.

The code was audited before this was written (`report_deepseek_run.md` §5):
the self-patch is bit-exact, the effects are dose-dependent, and what changes
carries the donor's numbers — properties no misplaced or inert patch could
produce.

## Conclusion

On DeepSeek V4 Flash, the filler-token uplift on 2-fact addition is real,
reproducible on local weights, and accompanied by exactly the lens picture
the paper reports. But the operands the lens reads on the dots are read on
every post-question token, with or without filler, and the answer position
does not depend on them: the +11 points survive the dots' content being
replaced wholesale. The uplift comes from the dots' *presence* — extra
positions after the question — not from computation performed on them.
"Hidden computation across filler tokens", in the sense of computation that
the answer needs and that happens on the filler, is not what produces the
effect in this model at k = 10 — the dots' content is causal in the paper's
graded, rank-level sense and irrelevant to accuracy. The J-lens neither reveals such computation nor is
needed to rule it out; it confirms the logit-lens picture and sharpens it by
a layer.

What the presence of extra positions does — attention structure, distance
from the question, sink capacity for the retrieval heads — is the question
this leaves open, and the natural next test is to keep the dots in the prompt
but mask them out of attention.

## What this does not show

- One model, one task, one filler type (dots), k ∈ {10, 25} for the lens
  and k = 10 for the causal tests, greedy decoding. The paper's transplant
  effect grows with k and is reported at k = 50–100; a k = 100 patching run
  would close that gap. The paper's DeepSeek V3 and Kimi K2 may differ; the dev model
  (Qwen3.5-4B) cannot do the task (1%) and was used only to debug the code.
- The lens results are correlational by construction; the causal results are
  about the dots' *content*, not their presence.
- The J-lens was fit on 25 pretraining documents at positions ≤ 128; readouts
  here are at positions 250–290 of a chat prompt. The four hyper-connection
  residual streams were collapsed with the model's own `hc_head` (verified at
  the target layer). The J-swap's effect on the residual was assumed rather
  than verified with a lens readout under the edit — immaterial given the
  wholesale-patch nulls, but stated.
- 42 rescued examples is a small group; the n = 1500 readout has 706/794
  correct/wrong but the causal runs are n = 300.

## Reproduction

Stage 1 (API): `build_filler_uplift_dataset.py`, `run_filler_uplift_sweep.py`.
Stages 2–3 (GPU, 2×H200): `20_lens_readout.py --model deepseek --k {0,10,25}
--lens both`, `21_analyze_readout.py`, `30_compare_lenses.py`,
`22_agreement_check.py`. Causal: `50_filler_patching.py --model deepseek
--n 300 --k 10` (all conditions; `--conditions band` for the whole-band and
all-layer patches). Environment and DeepSeek-specific fixes (no Jinja
template, transformers 5.16 FP8 loader, `kernels`, hyper-connection collapse):
`run_deepseek.md`. Dev-model run and the bugs it exposed (reasoning mode on,
truncated tail, no chance level): `report_dev_run.md`.
