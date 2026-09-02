"""
Shared loading + aggregation helpers for the lens-readout CSVs written by
20_lens_readout.py. Used by 21_analyze_readout.py (per-lens Figure-3 view)
and 30_compare_lenses.py (J-lens vs logit-lens).

Decode criterion (paper Sec. 4.2): a quantity counts as decoded at
(layer, position) when the TOP NUMERIC token equals its ground-truth value.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

QUANTITIES = ["A1", "A2", "sum"]


def tag_of(readout_csv: str) -> str:
    """results/lens_readout_deepseek_dots-10_logit.csv -> deepseek_dots-10"""
    base = os.path.basename(readout_csv).replace("lens_readout_", "")
    base = re.sub(r"\.csv$", "", base)
    return re.sub(r"_(logit|jlens|both)$", "", base)


def find_readouts(tag: str = "", outdir: str = "results") -> list[str]:
    """Every readout CSV for one condition. `tag=""` picks the newest one written.

    21 and 30 both want "the CSVs for one condition" — and 30 wants both lenses
    of it, which may be one `--lens both` file or a logit file plus a jlens file.
    Globbing for that beats retyping paths in a notebook cell.
    """
    paths = sorted(glob.glob(os.path.join(outdir, "lens_readout_*.csv")))
    if not paths:
        raise SystemExit(
            f"no lens_readout_*.csv in {outdir}/ — run 20_lens_readout.py first")
    if not tag:
        tag = tag_of(max(paths, key=os.path.getmtime))
    hits = [p for p in paths if tag_of(p) == tag]
    if not hits:
        raise SystemExit(f"no readout CSVs tagged {tag!r} in {outdir}/ "
                         f"(tags present: {sorted({tag_of(p) for p in paths})})")
    # A `both` file already covers both lenses. Merging it with older
    # single-lens files of the same condition would let their rows override
    # its own (load() keeps the last duplicate), so prefer it alone.
    both = [p for p in hits if p.endswith("_both.csv")]
    if both and len(hits) > 1:
        print(f"[find_readouts] using {both[0]} and ignoring older single-lens "
              f"files for the same condition: {[p for p in hits if p not in both]}")
        hits = both
    return hits


def answers_path(readout_csv: str) -> str:
    return os.path.join(os.path.dirname(readout_csv) or ".",
                        f"answers_{tag_of(readout_csv)}.csv")


def load(readout_csvs: str | list[str]) -> pd.DataFrame:
    """Load one or more readout CSVs (same tag) and merge in the ground truth.

    The match_* decode indicators are written by 20_lens_readout.py, because
    whether a quantity counts as decoded depends on the tokenizer's numeric
    mode (exact vs first-token prefix — see paper_tasks.NumericReadout).
    """
    if isinstance(readout_csvs, str):
        readout_csvs = [readout_csvs]
    tags = {tag_of(p) for p in readout_csvs}
    assert len(tags) == 1, f"readout files mix conditions: {tags}"
    df = pd.concat([pd.read_csv(p) for p in readout_csvs], ignore_index=True)
    df = df.drop_duplicates(subset=["lens", "idx", "layer", "pos"], keep="last")

    ans_csv = answers_path(readout_csvs[0])
    if not os.path.exists(ans_csv):
        raise SystemExit(f"companion answers file not found: {ans_csv}")
    ans = pd.read_csv(ans_csv)[["idx", "a1", "a2", "target"]]
    df = df.merge(ans, on="idx", how="left")
    missing = [q for q in QUANTITIES if f"match_{q}" not in df]
    if missing:
        raise SystemExit(
            f"{readout_csvs[0]} has no match_* columns ({missing}) — it predates "
            "the tokenizer-aware numeric readout. Re-run 20_lens_readout.py.")
    return df


def readout_mode(df: pd.DataFrame) -> str:
    """"exact" | "prefix" | "?" — which decode criterion produced these rows."""
    if "readout_mode" not in df:
        return "?"
    modes = sorted(set(df.readout_mode.dropna()))
    return modes[0] if len(modes) == 1 else "/".join(modes)


def mode_note(df: pd.DataFrame) -> str:
    """Short caption suffix so a prefix-mode figure is never mistaken for exact."""
    m = readout_mode(df)
    return "" if m == "exact" else f"  [decode: {m}-token match]"


def n_filler_of(df: pd.DataFrame) -> int:
    """Number of filler positions in a readout (0 for a k=0 tail-only readout)."""
    filler = df[df.pos_type == "filler"]
    return int(filler.pos.max()) + 1 if len(filler) else 0


RANK_TOP = 10


def rank_curve(df: pd.DataFrame, col: str, top: int = RANK_TOP) -> pd.Series:
    """Per layer: fraction of examples whose best (lowest) full-vocab rank of
    a quantity over the FILLER positions is below `top`. `col` is rank_<q> (or
    ctrl_rank_<q> for the shuffled control).

    A softer criterion than the argmax decode: it credits a quantity that is
    in the running before it wins, which is where a lens that reads earlier
    layers should show up first.
    """
    best = df[df.pos_type == "filler"].groupby(["idx", "layer"])[col].min()
    return (best < top).groupby("layer").mean()


def grid(df: pd.DataFrame, col: str):
    """(layers, positions, matrix[layer, pos]) of mean(col)."""
    p = df.pivot_table(index="layer", columns="pos", values=col, aggfunc="mean")
    p = p.sort_index().sort_index(axis=1)
    return p.index.to_numpy(), p.columns.to_numpy(), p.to_numpy()


# --------------------------------------------------------------------------- #
# "What algorithm?" summary
# --------------------------------------------------------------------------- #
def has_control(df: pd.DataFrame) -> bool:
    """Readouts written since the shuffled-quantity control carry ctrl_* columns."""
    return all(f"ctrl_{q}" in df for q in QUANTITIES)


def algorithm_summary(df: pd.DataFrame, n_filler: int) -> dict:
    out = {}
    fill = df[(df.pos < n_filler) & df.correct]
    post = df[(df.pos >= n_filler) & df.correct]
    ctrl = has_control(df)
    for lens in sorted(df.lens.unique()):
        s = {}
        fl = fill[fill.lens == lens]
        po = post[post.lens == lens]
        # 1. Does each quantity ever decode in the filler region? "Ever" is an
        #    any() over layers x positions, so it saturates on noise — the
        #    shuffled-quantity control (another example's A1/A2/sum, same
        #    criterion) is what it must beat.
        for q in QUANTITIES:
            per_ex = fl.groupby("idx")[f"match_{q}"].any()
            s[f"{q}_decoded_in_filler_frac"] = round(float(per_ex.mean()), 3)
            if ctrl:
                per_ex_c = fl.groupby("idx")[f"ctrl_{q}"].any()
                s[f"{q}_decoded_in_filler_frac_control"] = round(float(per_ex_c.mean()), 3)
                # cell-level: mean decode fraction over (layer, pos), vs control
                s[f"{q}_cell_frac"] = round(float(fl[f"match_{q}"].mean()), 3)
                s[f"{q}_cell_frac_control"] = round(float(fl[f"ctrl_{q}"].mean()), 3)
            # earliest layer at which it first decodes (median over examples)
            firsts = (fl[fl[f"match_{q}"]].groupby("idx")["layer"].min())
            s[f"{q}_first_layer_median"] = (
                float(firsts.median()) if len(firsts) else None)
            # position center-of-mass (where in the filler it lives)
            m = fl[fl[f"match_{q}"]]
            s[f"{q}_mean_position"] = (
                round(float(m.pos.mean()), 2) if len(m) else None)
        # 1b. rank criterion: the layer where "in the top-10 numeric tokens at
        #     some filler position" peaks, and the fraction there
        for q in QUANTITIES:
            curve = rank_curve(fl, f"rank_{q}")
            if len(curve):
                s[f"{q}_rank{RANK_TOP}_peak_layer"] = int(curve.idxmax())
                s[f"{q}_rank{RANK_TOP}_peak_frac"] = round(float(curve.max()), 3)
        # 2. sum in filler vs at the answer tail
        s["sum_decoded_in_post_frac"] = round(
            float(po.groupby("idx")["match_sum"].any().mean()), 3) if len(po) else None
        # 3. serial or parallel? see parallelism_summary
        s.update(parallelism_summary(fl))
        out[lens] = s
    # 4. wrong-example signature: retrieval without composition
    ref_lens = "logit" if "logit" in set(df.lens) else sorted(df.lens.unique())[0]
    wf = df[(df.pos < n_filler) & (~df.correct) & (df.lens == ref_lens)]
    if wf.idx.nunique():
        out[f"wrong_examples_{ref_lens}"] = {
            q: round(float(wf.groupby("idx")[f"match_{q}"].any().mean()), 3)
            for q in QUANTITIES}
    return out


def parallelism_summary(fl: pd.DataFrame) -> dict:
    """Is A1/A2 retrieval on the filler serial or parallel? `fl` = filler rows of
    one lens, correct examples. Four statistics, each paired within example:

    * depth_onset_A2_minus_A1 — first layer where A2 is decoded minus the same
      for A1. Serial-in-depth retrieval (A1, then A2) sits above zero;
      parallel retrieval is centred on zero. Reported as median and the
      fraction of examples with A2 later / same / earlier.
    * position_A2_minus_A1 — mean filler position of A2 cells minus A1 cells.
      A dot only attends backwards, so serial-in-position retrieval would put
      A2 on later dots. Median plus a two-sided sign test.
    * position_overlap_jaccard — at layers where both are present, the overlap
      of the dot sets carrying A1 and carrying A2 (mean over example-layers),
      and the fraction of those with NO shared dot. Two facts on different
      dots at the same layer is what "parallel, position-distributed" means.
      Uses the top-10 rank criterion, not the argmax: a cell's argmax is one
      token, so argmax sets are disjoint by construction and would say
      nothing.
    * both_in_top10_same_cell_frac — examples with at least one (layer, dot)
      where A1 and A2 are both in the top-10 numeric tokens: both facts
      readable from one residual vector at once (superposition).
    """
    s: dict = {}
    first = fl[fl.match_A1].groupby("idx").layer.min().to_frame("A1").join(
        fl[fl.match_A2].groupby("idx").layer.min().to_frame("A2"), how="inner")
    if len(first):
        d = first.A2 - first.A1
        s["depth_onset_A2_minus_A1_median"] = float(d.median())
        s["depth_onset_A2_later_same_earlier"] = [round(float((d > 0).mean()), 3),
                                                  round(float((d == 0).mean()), 3),
                                                  round(float((d < 0).mean()), 3)]
    pos = fl[fl.match_A1].groupby("idx").pos.mean().to_frame("A1").join(
        fl[fl.match_A2].groupby("idx").pos.mean().to_frame("A2"), how="inner")
    if len(pos):
        d = (pos.A2 - pos.A1)
        nz = d[d != 0]
        s["position_A2_minus_A1_median"] = round(float(d.median()), 2)
        s["position_A2_later_sign_test_p"] = round(_sign_test(int((nz > 0).sum()),
                                                              len(nz)), 4)
    top1, top2 = fl.rank_A1 < RANK_TOP, fl.rank_A2 < RANK_TOP
    both = (fl.assign(t1=top1, t2=top2).groupby(["idx", "layer"])
              .agg(a1=("t1", "any"), a2=("t2", "any")))
    both = both[both.a1 & both.a2].index
    if len(both):
        jac, jac_chance, disjoint = [], [], 0
        n_dots = fl.pos.nunique()
        sub = fl.set_index(["idx", "layer"]).loc[both]
        for key, g in sub.groupby(level=[0, 1]):
            p1 = set(g[g.rank_A1 < RANK_TOP].pos)
            p2 = set(g[g.rank_A2 < RANK_TOP].pos)
            jac.append(len(p1 & p2) / len(p1 | p2))
            disjoint += not (p1 & p2)
            # chance: the same two set sizes placed independently on n_dots
            e_inter = len(p1) * len(p2) / n_dots
            jac_chance.append(e_inter / (len(p1) + len(p2) - e_inter))
        s["position_overlap_jaccard_mean"] = round(float(np.mean(jac)), 3)
        s["position_overlap_jaccard_chance"] = round(float(np.mean(jac_chance)), 3)
        s["position_disjoint_frac"] = round(disjoint / len(jac), 3)
        s["n_example_layers_with_both"] = len(jac)
    top = (fl.rank_A1 < RANK_TOP) & (fl.rank_A2 < RANK_TOP)
    s["both_in_top10_same_cell_frac"] = round(float(top.groupby(fl.idx).any().mean()), 3)
    return s


def _sign_test(k: int, n: int) -> float:
    """Two-sided exact sign test p-value for k successes in n."""
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def tail_summary(df: pd.DataFrame, n_filler: int) -> pd.DataFrame:
    """Decode fraction at each post-filler tail token, best over layers, on
    correct examples — one row per (lens, tail token from the end). The same
    table for a k=0 readout and a filler readout says whether the operands are
    retrieved at 'Answer:' regardless of the dots (then the dots add sites, not
    new computation) or only when the filler is present.
    """
    tail = df[(df.pos >= n_filler) & df.correct].copy()
    tail["from_end"] = tail.pos - tail.pos.max() - 1          # -1 = last token
    rows = []
    for (lens, fe), g in tail.groupby(["lens", "from_end"]):
        by_layer = g.groupby("layer")
        row = dict(lens=lens, from_end=int(fe))
        for q in QUANTITIES:
            row[f"{q}_max_over_layers"] = round(float(by_layer[f"match_{q}"].mean().max()), 3)
            row[f"{q}_top10_max"] = round(float(by_layer[f"rank_{q}"]
                                                .apply(lambda r: (r < RANK_TOP).mean()).max()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def print_report(s: dict):
    print("\n=================== WHAT ALGORITHM? ===================")
    for lens in [k for k in s if not k.startswith("wrong_examples_")]:
        d = s[lens]
        print(f"\n--- {lens} (correct examples) ---")
        for q in QUANTITIES:
            ctrl = (f" (control {d[f'{q}_decoded_in_filler_frac_control']})"
                    if f"{q}_decoded_in_filler_frac_control" in d else "")
            cell = (f" | per-cell {d[f'{q}_cell_frac']} vs control "
                    f"{d[f'{q}_cell_frac_control']}" if f"{q}_cell_frac" in d else "")
            rank = (f" | top-{RANK_TOP} peak {d[f'{q}_rank{RANK_TOP}_peak_frac']} "
                    f"at layer {d[f'{q}_rank{RANK_TOP}_peak_layer']}"
                    if f"{q}_rank{RANK_TOP}_peak_layer" in d else "")
            print(f"  {q:>4}: decoded-in-filler {d[f'{q}_decoded_in_filler_frac']:>6}"
                  f"{ctrl} | first layer (median) {d[f'{q}_first_layer_median']} | "
                  f"mean position {d[f'{q}_mean_position']}{cell}{rank}")
        print(f"  sum decoded in post/answer tail: {d['sum_decoded_in_post_frac']}")
        if "depth_onset_A2_minus_A1_median" in d:
            print(f"  serial or parallel? depth onset A2−A1: median "
                  f"{d['depth_onset_A2_minus_A1_median']} layers, "
                  f"[later, same, earlier] = {d['depth_onset_A2_later_same_earlier']}; "
                  f"position A2−A1: median {d.get('position_A2_minus_A1_median')} dots "
                  f"(sign test p={d.get('position_A2_later_sign_test_p')}); "
                  f"top-10 dot-set overlap Jaccard {d.get('position_overlap_jaccard_mean')} "
                  f"(chance {d.get('position_overlap_jaccard_chance')}), "
                  f"disjoint in {d.get('position_disjoint_frac')} of "
                  f"{d.get('n_example_layers_with_both')} example-layers; "
                  f"both in top-10 at one cell: {d['both_in_top10_same_cell_frac']}")
    for key in [k for k in s if k.startswith("wrong_examples_")]:
        print(f"\n--- wrong examples ({key.replace('wrong_examples_', '')} lens) ---")
        print("  ", s[key], "  <- paper's signature: A1/A2 present, sum absent")
    print("\nReading guide: retrieval-then-composition = A1/A2 first-layer << "
          "sum first-layer; parallel retrieval = depth onset A2−A1 centred on 0 "
          "with the two facts on different dots (low Jaccard) at the same layer; "
          "J-lens 'sees more' = higher decoded-in-filler fractions and/or smaller "
          "first-layer at matched positions. A fraction that does not beat its "
          "shuffled control is noise, whatever the lens.")
    print("=======================================================\n")
