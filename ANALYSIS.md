# What algorithm runs across the filler — and what can the J-lens add?

This note is the interpretation companion to `21_analyze_readout.py` and
`30_compare_lenses.py`. It states the
paper's answer as testable expectations, then the hypotheses the J-lens can
discriminate that the logit lens cannot. Every claim below maps to a number in
`algorithm_summary_<tag>.json` or a panel in the generated figures.

## The paper's picture (what a successful replication looks like)

On 2-fact addition, Brauer et al. find a three-stage algorithm laid out in
space (filler positions) and depth (layers):

1. **Parallel retrieval, positionally specialized.** A1 is decoded most
   strongly at *early* filler positions, A2 at *later* ones, both from middle
   layers. In `21` this shows up as `A1_mean_position < A2_mean_position` and a
   high `parallel_A1_A2_same_layer_frac` (both addends readable at the same
   layer, different positions). Their transplant experiments (paper Sec. 4.3)
   show this positional structure is causal, not a lens artifact.
2. **Late composition.** The sum A1+A2 crystallizes in the last layers, mostly
   right before the answer position: `sum_first_layer_median` well above the
   A1/A2 first layers, `sum_decoded_in_post_frac` well above
   `sum_decoded_in_filler_frac`.
3. **Failure = failed composition, not failed retrieval.** On wrong examples,
   A1/A2 remain decodable but the sum is absent (`wrong_examples_logit` block:
   high A1/A2, near-zero sum). This is the paper's most striking diagnostic
   claim, and it is worth checking whether it survives the model swap to
   DeepSeek V4 Flash.

The comparison with explicit CoT (paper Fig. 4) suggests filler computation is
the model's ordinary single-pass computation *spread over extra positions*,
not a serialized derivation. The parallelism question at the heart of this
project — is filler uplift about extra *serial* depth or extra *parallel*
width? — is exactly what stage 1 vs stage 2 measures: parallel retrieval
across positions with a single late composition step is a width story.

## Why the logit lens might under-report, and where the J-lens differs

The logit lens asks: "if the residual at layer L were the *final* residual,
what token would it predict?" It reads a mid-layer state through the final
unembedding with no account of the computation still to come. It therefore
only sees features that are already written in the output vocabulary basis. A
value that mid-layers represent in a different basis — one the remaining
layers would *transport* into the readout basis — is invisible to it.

The Jacobian lens transports layer-L residuals through the (linearized)
remaining computation before unembedding. Same code path in `jlens`
(`use_jacobian=True/False`), so `20` gives an exact apples-to-apples
comparison at every (layer, position).

Concrete discriminations the comparison can make:

* **Earlier crystallization.** If the sum is *computed* earlier than the logit
  lens can *read* it, the J-lens should show `sum_first_layer_median` several
  layers below the logit lens's, and the per-layer curves in
  `jlens_vs_logit_<tag>.png` should shift left. If the two lenses agree, the
  late appearance of the sum is a fact about the computation, not about the
  readout — itself a publishable calibration point for the paper's Fig. 3.
* **Sum in the filler region.** The paper occasionally sees the sum "semi-early
  in filler" but mostly at the answer slot. If the J-lens finds the sum
  reliably inside the filler where the logit lens does not
  (`sum_decoded_in_filler_frac` gap), that revises the story: composition
  happens *in* the filler and is merely stored in a non-readout basis until
  the end. That would be a genuinely new claim relative to the paper.
* **Cleaner positional structure.** If the J-lens decode fractions for A1/A2
  are higher at the same (layer, position) cells, the unsupervised-decoding
  pipeline of the paper (their Sec. 5) has headroom: swap its logit-lens
  readout for the J-lens and re-measure judge accuracy.
* **Null result guardrail.** If J-lens ≈ logit lens everywhere on this task,
  that is still a result: filler intermediates live in the readout basis, and
  the interesting J-lens applications are elsewhere. The `00_smoke_test.py`
  assertion (lenses must differ *somewhere*) protects against mistaking a
  plumbing no-op for this null.

## Caveats to keep in front of you

* **Model swap.** The paper used DeepSeek V3 and Kimi K2; this replication
  targets DeepSeek V4 Flash because that is where a published J-lens exists.
  Stage 1 (accuracy sweep) is therefore a genuine *generalization* check, not a
  strict replication — V4 Flash may show smaller or larger uplift, and the
  paper itself found Qwen 3 480B gets little uplift on addition. If V4 Flash
  shows no 2-fact uplift, fall back to 1-fact addition (bigger effect in the
  paper: 54%→72%) before concluding anything.
* **Small behavioral effect.** DeepSeek V3's 2-fact uplift was ~3 points on
  1500 examples. Even at paper-scale n the McNemar flip test is the sensitive
  statistic, not the raw accuracy difference; `results/fig2_summary.csv`
  reports both.
* **Quantization.** The J-lens was presumably fit on unquantized residuals;
  running 4-bit changes the residual stream slightly. If J-lens results look
  noisy, check the smoke test's disagreement pattern on the dev model at the
  same precision first.
* **Numeric-token readout.** Like the paper, `20` restricts the "decoded
  value" to numeric tokens and exact match. Ranks of the true values are also
  stored, so near-misses (rank 1–5) can be analyzed without re-running.
