# %% [markdown]
# # Stage 3 — J-lens vs logit-lens (the new result over the paper)
#
# Takes readout CSVs from `20_lens_readout.py` that together cover BOTH lenses
# for one condition (either one `LENS = "both"` file, or the stage-2 logit file
# plus a jlens file), and asks where the Jacobian lens sees more than the paper's
# logit-lens readout:
#
# * difference maps (J-lens − logit-lens decode fraction) per quantity;
# * per-layer curves of the best-over-positions decode fraction, per lens — a
#   leftward J-lens shift = earlier crystallization than the logit lens can read
#   (see `ANALYSIS.md` for what each outcome would mean);
# * a side-by-side algorithm summary (first-layer medians, in-filler fractions).
#
# Outputs in `results/`:
#
# * `jlens_vs_logit_<tag>.png`
# * `compare_summary_<tag>.json` + a printed report
#
# Run cells top-to-bottom in VS Code (`# %%` = one Jupyter cell). `TAG = ""`
# picks up the most recently written condition and merges every lens file it has.
# Headless equivalent:
#
# ```bash
# python 30_compare_lenses.py --readout results/lens_readout_deepseek_dots-10_logit.csv \
#                                       results/lens_readout_deepseek_dots-10_jlens.csv
# ```

# %% Config
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from lens_analysis import (QUANTITIES, algorithm_summary, find_readouts, grid,
                           load, mode_note, n_filler_of, print_report,
                           readout_mode, tag_of)

TAG = ""            # "" = the most recently written condition in OUTDIR
OUTDIR = "results"
READOUTS: list[str] = []   # set explicitly to override the TAG lookup


def _running_as_script() -> bool:
    """True for `python 30_compare_lenses.py ...`; False in a notebook cell."""
    return __name__ == "__main__" and "ipykernel" not in sys.modules


if _running_as_script() and len(sys.argv) > 1:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", nargs="+", default=None,
                    help="readout CSV(s) from 20 covering both lenses for one "
                         "condition (one 'both' file, or logit + jlens files)")
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
# ## 1. Load both lenses for one condition

# %%
df = load(READOUTS)
lenses = set(df.lens.unique())
assert {"jlens", "logit"} <= lenses, (
    f"need both lenses, found {sorted(lenses)} — run 20 with the missing lens "
    f"(or LENS = 'both') so this condition has both CSVs")
n_filler = n_filler_of(df)
print(f"{df.idx.nunique()} examples | {df.layer.nunique()} layers | "
      f"{df.pos.nunique()} positions ({n_filler} filler) | "
      f"decode mode: {readout_mode(df)} | "
      f"accuracy {df.groupby('idx')['correct'].first().mean():.2%}")

# %% [markdown]
# ## 2. Difference maps + per-layer decode curves
#
# Top row: J-lens − logit-lens decode fraction (red = the J-lens reads a quantity
# the logit lens misses). Bottom row: for each lens, the best decode fraction
# over positions at each layer — a leftward shift of the J-lens curve means the
# quantity is present earlier than the logit lens can see. Correct examples only.

# %%
def compare_lenses(df, n_filler: int, tag: str, outdir: str):
    cor = df[df.correct]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for c, q in enumerate(QUANTITIES):
        layers_j, poss, gj = grid(cor[cor.lens == "jlens"], f"match_{q}")
        layers_l, _, gl = grid(cor[cor.lens == "logit"], f"match_{q}")
        ax = axes[0, c]
        vmax = max(np.abs(gj - gl).max(), 1e-9)
        im = ax.imshow(gj - gl, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[poss[0] - .5, poss[-1] + .5,
                               layers_j[0] - .5, layers_j[-1] + .5])
        ax.axvline(n_filler - 0.5, color="k", ls="--", lw=1)
        ax.set_title(f"{q}: J-lens − logit-lens", fontsize=10)
        ax.set_xlabel("position")
        if c == 0:
            ax.set_ylabel("source layer")
        fig.colorbar(im, ax=ax, fraction=0.046)
        # per-layer curves: best-over-positions decode fraction
        ax2 = axes[1, c]
        for lens, g, layers in (("jlens", gj, layers_j), ("logit", gl, layers_l)):
            ax2.plot(layers, np.nanmax(g, axis=1), marker=".", label=lens)
        ax2.set_xlabel("source layer")
        ax2.set_title(f"{q}: max decode fraction over positions", fontsize=10)
        if c == 0:
            ax2.set_ylabel("decode fraction")
            ax2.legend(frameon=False)
    fig.suptitle(f"J-lens vs logit-lens (correct examples) — {tag}{mode_note(df)}")
    fig.tight_layout()
    path = os.path.join(outdir, f"jlens_vs_logit_{tag}.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    plt.show()


compare_lenses(df, n_filler, TAG, OUTDIR)

# %% [markdown]
# ## 3. Side-by-side algorithm summary
#
# The same aggregation 21 prints, but with both lenses in one report so the
# first-decode layers and in-filler fractions line up column-for-column.
# A J-lens advantage = higher decoded-in-filler fractions and/or a smaller
# first-decode layer at matched positions.

# %%
summary = algorithm_summary(df, n_filler)
js = os.path.join(OUTDIR, f"compare_summary_{TAG}.json")
with open(js, "w") as f:
    json.dump(summary, f, indent=2)
print(f"wrote {js}")
print_report(summary)

# %%
