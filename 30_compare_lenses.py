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
#   (see `ANALYSIS.md` for what each outcome would mean) — with the
#   shuffled-quantity control dotted underneath: a curve that sits on its
#   control is chance, and a J-lens/logit-lens gap that the controls share is
#   a property of the lens's numeric-token bias, not of the computation;
# * the same per-layer curves under a softer, rank-based criterion (the
#   quantity is among the top-10 numeric tokens at some filler position) —
#   where a lens that reads a quantity before it wins the argmax shows up first;
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

from lens_analysis import (QUANTITIES, RANK_TOP, algorithm_summary, find_readouts,
                           grid, has_control, load, mode_note, n_filler_of,
                           print_report, rank_curve, readout_mode, tag_of)

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
MIN_CORRECT = 20    # fewer correct examples than this and the maps are noise


def compare_lenses(df, n_filler: int, tag: str, outdir: str):
    cor = df[df.correct]
    n_cor = cor.idx.nunique()
    subset = f"correct examples, n={n_cor}"
    if n_cor < MIN_CORRECT:          # e.g. the dev model, which gets ~1% right
        cor = df
        subset = f"ALL examples, n={df.idx.nunique()} — only {n_cor} correct"
        print(f"only {n_cor} correct examples: comparing lenses over all of them")
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5))
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
        # per-layer curves: best-over-positions decode fraction, with the
        # shuffled-quantity control (dotted) as the chance level of the same
        # max-over-positions statistic
        ax2 = axes[1, c]
        for lens, g, layers, color in (("jlens", gj, layers_j, "C0"),
                                       ("logit", gl, layers_l, "C1")):
            ax2.plot(layers, np.nanmax(g, axis=1), marker=".", color=color, label=lens)
            if has_control(cor):
                _, _, gc = grid(cor[cor.lens == lens], f"ctrl_{q}")
                ax2.plot(layers, np.nanmax(gc, axis=1), ls=":", color=color,
                         label=f"{lens} control")
        ax2.set_xlabel("source layer")
        ax2.set_title(f"{q}: max decode fraction over positions", fontsize=10)
        if c == 0:
            ax2.set_ylabel("decode fraction")
            ax2.legend(frameon=False, fontsize=8)
        # rank criterion: quantity in the top-RANK_TOP numeric tokens at some
        # filler position. Softer than the argmax, so a lens that reads a
        # quantity before it wins the argmax shows up here first. The control
        # (dotted) exists only in readouts written with ctrl_rank_* columns.
        ax3 = axes[2, c]
        for lens, color in (("jlens", "C0"), ("logit", "C1")):
            d = cor[cor.lens == lens]
            ax3.plot(rank_curve(d, f"rank_{q}"), marker=".", color=color, label=lens)
            if f"ctrl_rank_{q}" in d:
                ax3.plot(rank_curve(d, f"ctrl_rank_{q}"), ls=":", color=color,
                         label=f"{lens} control")
        ax3.set_xlabel("source layer")
        ax3.set_title(f"{q}: in top-{RANK_TOP} numeric tokens at best filler position",
                      fontsize=10)
        if c == 0:
            ax3.set_ylabel(f"fraction of examples")
            ax3.legend(frameon=False, fontsize=8)
    fig.suptitle(f"J-lens vs logit-lens ({subset}) — {tag}{mode_note(df)}")
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
