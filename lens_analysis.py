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
    return int(df[df.pos_type == "filler"].pos.max()) + 1


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
        # 2. sum in filler vs at the answer tail
        s["sum_decoded_in_post_frac"] = round(
            float(po.groupby("idx")["match_sum"].any().mean()), 3) if len(po) else None
        # 3. parallel retrieval: same example, same layer, A1 and A2
        #    simultaneously decoded at DIFFERENT positions
        both = (fl.groupby(["idx", "layer"])
                  .agg(a1_any=("match_A1", "any"), a2_any=("match_A2", "any")))
        par_ex = both[both.a1_any & both.a2_any].reset_index().idx.nunique()
        n_ex = fl.idx.nunique()
        s["parallel_A1_A2_same_layer_frac"] = (
            round(par_ex / n_ex, 3) if n_ex else None)
        out[lens] = s
    # 4. wrong-example signature: retrieval without composition
    ref_lens = "logit" if "logit" in set(df.lens) else sorted(df.lens.unique())[0]
    wf = df[(df.pos < n_filler) & (~df.correct) & (df.lens == ref_lens)]
    if wf.idx.nunique():
        out[f"wrong_examples_{ref_lens}"] = {
            q: round(float(wf.groupby("idx")[f"match_{q}"].any().mean()), 3)
            for q in QUANTITIES}
    return out


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
            print(f"  {q:>4}: decoded-in-filler {d[f'{q}_decoded_in_filler_frac']:>6}"
                  f"{ctrl} | first layer (median) {d[f'{q}_first_layer_median']} | "
                  f"mean position {d[f'{q}_mean_position']}{cell}")
        print(f"  sum decoded in post/answer tail: {d['sum_decoded_in_post_frac']}")
        print(f"  A1 & A2 co-decoded at one layer (parallel retrieval): "
              f"{d['parallel_A1_A2_same_layer_frac']}")
    for key in [k for k in s if k.startswith("wrong_examples_")]:
        print(f"\n--- wrong examples ({key.replace('wrong_examples_', '')} lens) ---")
        print("  ", s[key], "  <- paper's signature: A1/A2 present, sum absent")
    print("\nReading guide: retrieval-then-composition = A1/A2 first-layer << "
          "sum first-layer; position specialization = mean_position(A1) < "
          "mean_position(A2); J-lens 'sees more' = higher decoded-in-filler "
          "fractions and/or smaller first-layer at matched positions. A fraction "
          "that does not beat its shuffled control is noise, whatever the lens.")
    print("=======================================================\n")
