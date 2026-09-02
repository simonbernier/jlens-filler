# %% [markdown]
# # Stage 2 analysis — Figure-3 heatmaps + the "what algorithm?" summary
#
# Stage 2 is the paper replication proper: run `20_lens_readout.py` with
# `LENS = "logit"`, then this notebook reproduces the paper's logit-lens picture
# on DeepSeek V4 Flash — decode-fraction heatmaps over layer × position, correct
# vs wrong examples, A1/A2/sum.
#
# It works identically on a `jlens` or a `both` readout (one figure per lens
# present). The J-lens **vs** logit-lens comparison lives in
# `30_compare_lenses.py`.
#
# Outputs in `results/`:
#
# * `fig3_<tag>_<lens>.png` — heatmaps: correct/wrong × A1/A2/sum, per lens
# * `algorithm_summary_<tag>_<lens...>.json` + a printed report
#
# `TAG = ""` picks up whatever `20_lens_readout.py` wrote last, so the usual flow 
# is: run 20, run this, look at the figure. Headless equivalent:
#
# ```bash
# python 21_analyze_readout.py --readout results/lens_readout_deepseek_dots-10_logit.csv
# ```

# %% Config
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt

from lens_analysis import (
    QUANTITIES,
    algorithm_summary,
    find_readouts,
    grid,
    has_control,
    load,
    mode_note,
    n_filler_of,
    print_report,
    readout_mode,
    tag_of,
    tail_summary,
)

TAG = "dev_dots-10"        # "" = the most recently written condition in OUTDIR
OUTDIR = "results"
READOUTS: list[str] = []   # set explicitly to override the TAG lookup


def _running_as_script() -> bool:
    """True for `python 21_analyze_readout.py ...`; False in a notebook cell."""
    return __name__ == "__main__" and "ipykernel" not in sys.modules


if _running_as_script() and len(sys.argv) > 1:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", nargs="+", default=None,
                    help="results/lens_readout_<tag>_<lens>.csv from 20 "
                         "(several files of the same tag are merged)")
    ap.add_argument("--tag", default=TAG,
                    help="condition tag, e.g. deepseek_dots-10 (default: newest)")
    ap.add_argument("--outdir", default=None)
    _a = ap.parse_args()
    READOUTS, TAG = _a.readout or [], _a.tag
    if _a.outdir:
        OUTDIR = _a.outdir
    elif READOUTS:
        OUTDIR = os.path.dirname(READOUTS[0]) or "."

READOUTS = READOUTS or find_readouts(TAG, OUTDIR)
TAG = tag_of(READOUTS[0])
print(f"tag {TAG!r}, {len(READOUTS)} file(s):")
for p in READOUTS:
    print("  ", p)

# %% [markdown]
# ## 1. Load the readout
#
# `load` merges the CSVs, joins the ground truth from `answers_<tag>.csv`, and
# refuses a readout written before the tokenizer-aware decode criterion existed.

# %%
df = load(READOUTS)
n_filler = n_filler_of(df)
lenses = sorted(df.lens.unique())
print(f"{df.idx.nunique()} examples | {df.layer.nunique()} layers | "
      f"{df.pos.nunique()} positions ({n_filler} filler) | lenses: {lenses} | "
      f"decode mode: {readout_mode(df)} | "
      f"accuracy {df.groupby('idx')['correct'].first().mean():.2%}")

# %% [markdown]
# ## 2. Figure 3 — decode fraction over layer × position
#
# One figure per lens in the readout. Rows are correct / wrong examples (plus
# the shuffled-quantity control on the correct examples: the chance level of
# the decode criterion, cell by cell), columns are A1 / A2 / sum; the dashed
# line marks the end of the filler region. The paper's signature for wrong
# examples is A1 and A2 present but sum absent — retrieval without composition.

# %%
def fig3(df, lens: str, n_filler: int, tag: str, outdir: str):
    sub = df[df.lens == lens]
    n_cor = sub[sub.correct].idx.nunique()
    n_wrg = sub[~sub.correct].idx.nunique()
    # rows: (label, example mask, column prefix, n). The third row is the
    # shuffled-quantity control on the same examples as the first — the chance
    # level of every cell above it, in the same colour scale.
    rows = [("correct", sub.correct, "match", n_cor),
            ("wrong", ~sub.correct, "match", n_wrg)]
    if has_control(sub):
        if n_cor:
            rows.append(("control, correct", sub.correct, "ctrl", n_cor))
        else:                     # nothing correct (dev model): control over all
            rows.append(("control, all", sub.correct | True, "ctrl",
                         sub.idx.nunique()))
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.25 * len(rows)),
                             sharex=True, sharey=True)
    im = None
    for r, (label, mask, col, n_ex) in enumerate(rows):
        for c, q in enumerate(QUANTITIES):
            ax = axes[r, c]
            if n_ex == 0:
                ax.set_axis_off()
                ax.set_title(f"{q} — no {label} examples")
                continue
            layers, poss, g = grid(sub[mask], f"{col}_{q}")
            im = ax.imshow(g, aspect="auto", origin="lower", vmin=0, vmax=1,
                           cmap="magma",
                           extent=[poss[0] - .5, poss[-1] + .5,
                                   layers[0] - .5, layers[-1] + .5])
            ax.axvline(n_filler - 0.5, color="w", ls="--", lw=1)
            ax.set_title(f"{q}  ({label}, n={n_ex})", fontsize=10)
            if r == len(rows) - 1:
                ax.set_xlabel("position (filler → | answer)")
            if c == 0:
                ax.set_ylabel("source layer")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02,
                     label="fraction of examples decoded (top numeric token)")
    title = f"{lens} — 2-fact addition, {tag}{mode_note(df)}"
    if has_control(sub):
        title += "\ncontrol = same criterion against another example's A1/A2/sum (chance)"
    fig.suptitle(title)
    path = os.path.join(outdir, f"fig3_{tag}_{lens}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    plt.show()


for lens in lenses:
    fig3(df, lens, n_filler, TAG, OUTDIR)

# %% [markdown]
# ## 3. "What algorithm?" summary
#
# First-decode layers, in-filler decode fractions, position centre-of-mass, and
# the parallel-retrieval check (A1 and A2 co-decoded at one layer, different
# positions). `ANALYSIS.md` says what each pattern would mean.

# %%
summary = algorithm_summary(df, n_filler)
js = os.path.join(OUTDIR, f"algorithm_summary_{TAG}_{'-'.join(lenses)}.json")
with open(js, "w") as f:
    json.dump(summary, f, indent=2)
print(f"wrote {js}")
print_report(summary)

# %% [markdown]
# ## 4. The post-filler tail — are the dots doing anything 'Answer:' doesn't?
#
# The operands are decoded at the tail tokens of a filler prompt as well as on
# the dots. This table (best-over-layers decode fraction per tail token, correct
# examples) is comparable across k — run 20 with `--k 0` (which reads only the
# tail) and put the two tables side by side: if the k=0 tail retrieves the
# operands as well as the k=10 tail does, the dots add retrieval *sites*, not a
# computation that would otherwise be missing.

# %%
tail = tail_summary(df, n_filler)
tail_csv = os.path.join(OUTDIR, f"tail_summary_{TAG}.csv")
tail.to_csv(tail_csv, index=False)
print(f"tail tokens (from_end -1 = the answer is predicted here), k={n_filler}:")
print(tail.to_string(index=False))
print(f"wrote {tail_csv}")

if len(lenses) < 2:
    print(f"only the {lenses[0]!r} lens here — for the J-lens vs logit-lens "
          f"comparison, run 20 with the other lens, then 30_compare_lenses.py")

# %%
