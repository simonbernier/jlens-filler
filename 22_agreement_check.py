# %% [markdown]
# # Stage 2 check — do the local weights reproduce the API's accuracy?
#
# Stage 1 measured the filler-token uplift through the API; stages 2–3 read the
# mechanism off local weights. That only means anything if the local model
# *behaves* like the one the API served — above all on **2-fact addition**,
# where the API sweep showed a large, highly significant uplift.
#
# This notebook pairs the two, per example:
#
# * `results/answers_<model>_dots-<k>.csv` — greedy answers from `20_lens_readout.py`
# * `results/fig2_raw.jsonl` — the stage-1 API sweep
#
# Those join on `idx` **only if 20 ran with `SOURCE = "fig2"`**, which replays
# stage 1's own examples and its exact rendered prompts. (`pt.build_dataset` does
# not reproduce that test set at any seed — it holds out a different 10 elements
# for few-shot, so it samples from a different pool. Cell 2 verifies the join
# element-pair by element-pair and refuses rather than report a bogus number.)
#
# What comes out:
#
# 1. **per k** — local vs API accuracy on the same examples, exact-prediction
#    agreement, and an exact McNemar test of local against API;
# 2. **uplift** — the effect you actually care about: accuracy(k) − accuracy(0),
#    measured locally and via the API, each with its own paired McNemar. Needs a
#    local k=0 run (`python 20_lens_readout.py --k 0`, which is answers-only and
#    cheap — no lens);
# 3. a figure overlaying the two accuracy-vs-k curves.
#
# Run cells top-to-bottom in VS Code (`# %%` = one Jupyter cell).

# %% Config
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paper_tasks as pt

MODEL = "deepseek"     # registry key the local answers were written under
FILLER = "dots"        # stage 1 swept dots only
TASK = "2fact"         # "2fact" | "1fact" — 20 does 2-fact
API_RAW = "results/fig2_raw.jsonl"
OUTDIR = "results"


def _running_as_script() -> bool:
    """True for `python 22_agreement_check.py ...`; False in a notebook cell."""
    return __name__ == "__main__" and "ipykernel" not in sys.modules


if _running_as_script() and len(sys.argv) > 1:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--filler", default=FILLER)
    ap.add_argument("--task", default=TASK, choices=["2fact", "1fact"])
    ap.add_argument("--api-raw", default=API_RAW)
    ap.add_argument("--outdir", default=OUTDIR)
    _a = ap.parse_args()
    MODEL, FILLER, TASK = _a.model, _a.filler, _a.task
    API_RAW, OUTDIR = _a.api_raw, _a.outdir

# local answer files, keyed by k
_pat = re.compile(rf"answers_{re.escape(MODEL)}_{re.escape(FILLER)}-(\d+)\.csv$")
LOCAL_FILES = {}
for path in sorted(glob.glob(os.path.join(OUTDIR, f"answers_{MODEL}_{FILLER}-*.csv"))):
    m = _pat.search(os.path.basename(path))
    if m:
        LOCAL_FILES[int(m.group(1))] = path
if not LOCAL_FILES:
    raise SystemExit(
        f"no answers_{MODEL}_{FILLER}-*.csv in {OUTDIR}/ — run 20_lens_readout.py "
        f"first (with SOURCE='fig2')")
if not os.path.exists(API_RAW):
    raise SystemExit(f"{API_RAW} not found — this is the stage-1 sweep output")

print(f"local runs found: k={sorted(LOCAL_FILES)}")
for k in sorted(LOCAL_FILES):
    print("  ", LOCAL_FILES[k])

# %% [markdown]
# ## 1. Load both sides
#
# The API side is one row per (task, k, idx). The local side is one row per idx
# per k-file. Both carry `pred` (the parsed integer) and `correct`.

# %%
api_rows = []
with open(API_RAW, encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("task") == TASK:
            api_rows.append(r)
api = pd.DataFrame(api_rows).drop_duplicates(subset=["k", "idx"], keep="last")
print(f"API: {len(api)} rows, k={sorted(api.k.unique())}, "
      f"{api.idx.nunique()} distinct examples")

local = pd.concat(
    [pd.read_csv(p).assign(k=k) for k, p in sorted(LOCAL_FILES.items())],
    ignore_index=True)
local["key"] = local.elem_a.str.cat(local.elem_b, sep="+")
print(f"local: {len(local)} rows, k={sorted(local.k.unique())}")

# %% [markdown]
# ## 2. Verify the join before trusting it
#
# `idx` is only a shared identifier if 20 replayed stage 1's dataset. Comparing
# the element pair behind each `idx` catches a `SOURCE='synthetic'` run — where
# the pairs are drawn from a different pool and the overlap is near zero —
# before any accuracy number gets reported.

# %%
check = local.merge(api[["k", "idx", "key", "target"]], on=["k", "idx"],
                    how="inner", suffixes=("_local", "_api"))
if check.empty:
    raise SystemExit("no (k, idx) pairs in common — the local runs cover "
                     f"k={sorted(local.k.unique())}, the API sweep "
                     f"k={sorted(api.k.unique())}")

same_pair = check.key_local == check.key_api
same_target = check.target_local == check.target_api
print(f"{len(check)} paired rows; element pair matches on "
      f"{same_pair.mean():.1%}, target on {same_target.mean():.1%}")
if not same_pair.all():
    bad = check[~same_pair].head(3)
    raise SystemExit(
        f"idx does NOT identify the same example on both sides "
        f"({(~same_pair).sum()} of {len(check)} disagree, e.g. "
        f"{bad.key_local.tolist()} locally vs {bad.key_api.tolist()} via the API).\n"
        f"20_lens_readout.py must run with SOURCE='fig2' so it replays stage 1's "
        f"own examples and prompts; a synthetic test set is not comparable.")
print("join verified — same examples on both sides")

# %% [markdown]
# ## 3. Per-k agreement
#
# `agree_correct` is outcome agreement (both right or both wrong);
# `agree_pred` is the sharper one — the two produced the *same integer*.
# McNemar here tests local against API on the same examples: `b` = API wrong and
# local right, `c` = API right and local wrong. A large p-value is the good
# outcome (no detectable accuracy difference), which is the opposite of how the
# test is read in the stage-1 sweep.

# %%
def paired(df_local: pd.DataFrame, df_api: pd.DataFrame) -> pd.DataFrame:
    """Inner-join one k on idx, keeping both sides' pred/correct."""
    a = df_api[["idx", "pred", "correct"]].rename(
        columns={"pred": "pred_api", "correct": "correct_api"})
    b = df_local[["idx", "pred", "correct"]].rename(
        columns={"pred": "pred_local", "correct": "correct_local"})
    return a.merge(b, on="idx", how="inner")


rows = []
for k in sorted(LOCAL_FILES):
    p = paired(local[local.k == k], api[api.k == k])
    if p.empty:
        print(f"k={k}: no overlap with the API sweep, skipping")
        continue
    n = len(p)
    acc_l, acc_a = p.correct_local.mean(), p.correct_api.mean()
    b = int((~p.correct_api & p.correct_local).sum())   # API wrong -> local right
    c = int((p.correct_api & ~p.correct_local).sum())   # API right -> local wrong
    rows.append(dict(
        k=k, n=n,
        acc_local=acc_l, se_local=pt.binomial_se(acc_l, n),
        acc_api=acc_a, se_api=pt.binomial_se(acc_a, n),
        delta=acc_l - acc_a,
        agree_correct=float((p.correct_local == p.correct_api).mean()),
        agree_pred=float((p.pred_local == p.pred_api).mean()),
        api_wrong_local_right=b, api_right_local_wrong=c,
        mcnemar_p=pt.mcnemar_exact(b, c)))

agree = pd.DataFrame(rows).sort_values("k")
agree.to_csv(os.path.join(OUTDIR, f"agreement_{MODEL}_{FILLER}.csv"), index=False)
print(agree.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nwrote {OUTDIR}/agreement_{MODEL}_{FILLER}.csv")

for _, r in agree.iterrows():
    verdict = ("no detectable difference" if r.mcnemar_p >= 0.05
               else "LOCAL AND API DIFFER significantly")
    print(f"\nk={int(r.k):>3}  n={int(r.n)}  local {r.acc_local:.1%} vs API "
          f"{r.acc_api:.1%}  (Δ {r.delta:+.1%})")
    print(f"      same answer on {r.agree_pred:.1%} of examples, same outcome on "
          f"{r.agree_correct:.1%}")
    print(f"      McNemar p={r.mcnemar_p:.3g}  ->  {verdict}")

# %% [markdown]
# ## 4. Uplift — the number that has to survive
#
# Agreement at a single k is reassuring but not the point: stage 3 asks what the
# filler *does*, so the local model has to show the same **effect**, not just the
# same level. This needs a local k=0 run; without one the cell says so and stops.
#
# Both uplifts are paired McNemar on the same fixed test set, exactly as in the
# stage-1 summary, so the two columns are directly comparable.

# %%
def uplift(df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    """accuracy(k) - accuracy(0) on paired examples, with an exact McNemar."""
    base = df[df.k == 0].set_index("idx")["correct"]
    out = []
    for k in ks:
        if k == 0:
            continue
        cur = df[df.k == k].set_index("idx")["correct"]
        shared = base.index.intersection(cur.index)
        if not len(shared):
            continue
        b0, ck = base.loc[shared], cur.loc[shared]
        bb, cc = int((~b0 & ck).sum()), int((b0 & ~ck).sum())
        out.append(dict(k=k, n=len(shared),
                        acc_0=b0.mean(), acc_k=ck.mean(),
                        uplift=ck.mean() - b0.mean(),
                        flips_wrong_to_right=bb, flips_right_to_wrong=cc,
                        mcnemar_p=pt.mcnemar_exact(bb, cc)))
    return pd.DataFrame(out)


ks_local = sorted(LOCAL_FILES)
if 0 not in ks_local:
    print("no local k=0 run, so no local uplift to compare.\n"
          "  python 20_lens_readout.py --model {m} --k 0 --n {n}   "
          "(answers only — k=0 has no filler region, so no lens work)"
          .format(m=MODEL, n=int(agree.n.max())))
    up_local = pd.DataFrame()
else:
    shared_idx = set(local[local.k == 0].idx)
    up_local = uplift(local[local.idx.isin(shared_idx)], ks_local)
    up_api = uplift(api[api.idx.isin(shared_idx) & api.k.isin(ks_local)], ks_local)
    comp = up_local.merge(up_api, on="k", suffixes=("_local", "_api"))
    comp["uplift_gap"] = comp.uplift_local - comp.uplift_api
    comp.to_csv(os.path.join(OUTDIR, f"uplift_compare_{MODEL}_{FILLER}.csv"),
                index=False)
    print(comp[["k", "n_local", "acc_0_local", "acc_k_local", "uplift_local",
                "mcnemar_p_local", "acc_0_api", "acc_k_api", "uplift_api",
                "mcnemar_p_api", "uplift_gap"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {OUTDIR}/uplift_compare_{MODEL}_{FILLER}.csv")
    for _, r in comp.iterrows():
        print(f"\nk={int(r.k):>3}  uplift local {r.uplift_local:+.1%} "
              f"(p={r.mcnemar_p_local:.3g})  vs API {r.uplift_api:+.1%} "
              f"(p={r.mcnemar_p_api:.3g})   gap {r.uplift_gap:+.1%}")
        if r.mcnemar_p_local >= 0.05:
            print("      the local uplift is NOT significant — the mechanism you "
                  "are about to read out may not be present in this run")

# %% [markdown]
# ## 5. Accuracy vs k — local against the API
#
# k=0 is a plotted point in its own slot rather than a horizontal line, so both
# series' no-filler baselines are compared the same way as their k>0 points.

# %%
plot_ks = sorted(agree.k)
nz = [k for k in plot_ks if k != 0]
xpos = {k: i for i, k in enumerate(nz)}
xpos[0] = -1.15

fig, ax = plt.subplots(figsize=(7, 4.4))
if 0 in xpos and 0 in plot_ks:
    ax.axvline(-0.6, color="0.75", ls=":", lw=1, zorder=0)
for label, acc_col, se_col, color, marker in [
        (f"API ({TASK}, stage 1)", "acc_api", "se_api", "#d1495b", "s"),
        (f"local weights ({MODEL})", "acc_local", "se_local", "#4878d0", "o")]:
    g = agree.sort_values("k")
    ax.errorbar([xpos[k] for k in g.k], g[acc_col] * 100, yerr=g[se_col] * 100,
                marker=marker, ms=5, capsize=3, lw=1.8, color=color, label=label)
ax.set_xticks(([xpos[0]] if 0 in plot_ks else []) + list(range(len(nz))))
ax.set_xticklabels((["none"] if 0 in plot_ks else []) + [str(k) for k in nz])
ax.set_xlabel("Number of Filler [k]")
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"{TASK} — local weights vs the API sweep, same examples "
             f"(n={int(agree.n.max())})")
ax.legend(frameon=False)
fig.tight_layout()
path = os.path.join(OUTDIR, f"agreement_{MODEL}_{FILLER}.png")
fig.savefig(path, dpi=150)
print(f"wrote {path}")
plt.show()

# %%
