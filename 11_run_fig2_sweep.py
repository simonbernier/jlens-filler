# %% [markdown]
# # Stage 1b — Figure 2 sweep: DeepSeek V4 Flash via API, dot filler
#
# Runs the datasets built by `10_build_fig2_dataset.py`
# (`data/fig2_1fact.jsonl`, `data/fig2_2fact.jsonl`) against the API and
# reproduces Figure 2 of *Reading Between the Dots* (arXiv:2607.03502):
# accuracy vs number of dot filler tokens, one panel per task, with the
# no-filler (k=0) baseline as a dashed line, ±1 binomial-SE error bars, and
# exact McNemar tests of each k against baseline.
#
# Resumable: replies are appended to `results/fig2_raw.jsonl`; rerunning the
# sweep cell skips everything already answered. That also means growing N in 10
# only costs the NEW examples here.
#
# Run cells top-to-bottom in VS Code (# %% = one Jupyter cell).

# %% Config
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm  # pip install tqdm

import api_common as api

# temperature=0 (greedy), matching the paper's released eval code (their vLLM
# SamplingParams AND API calls both use temperature=0). One greedy sample per
# example gives a deterministic per-example outcome, which is what the fixed
# test set + McNemar flip counts are built on. V4's serving default of 0.6 is
# a chat default, not a methodological choice — sampling at 0.6 would require
# several samples per example to estimate per-example accuracy (m x cost) and
# would add noise to the k=0 vs k pairing.
TEMPERATURE = 0.0

DATASETS = {"1fact": "data/fig2_1fact.jsonl",
            "2fact": "data/fig2_2fact.jsonl"}
RAW_PATH = "results/fig2_raw.jsonl"
os.makedirs("results", exist_ok=True)

# %% [markdown]
# ## 1. Load the datasets from 10

# %%
records: list[dict] = []
for task, path in DATASETS.items():
    assert os.path.exists(path), f"{path} missing — run 10_build_fig2_dataset.py first"
    with open(path) as f:
        records += [json.loads(line) for line in f]

meta = json.load(open("data/fig2_meta.json"))
KS = meta["ks"]
print(f"{len(records)} (example, k) records; ks={KS}; "
      f"n per condition: 1fact={meta['n_1fact']}, 2fact={meta['n_2fact']}")
assert meta["api_model"] == api.API_MODEL and meta.get("provider") == api.PROVIDER, (
    "datasets were built against a different model/provider than api_common now "
    "pins — the knowledge filter and the sweep must share one serving stack")

# %% [markdown]
# ## 2. API client + one cheap smoke call
#
# If the smoke call 404s with "No endpoints found", the PROVIDER pin in
# api_common.py matches none of the endpoints printed here — pick a live slug.

# %%
client = api.make_client()
api.list_endpoints()
api.smoke_call(client)

# %% [markdown]
# ## 3. Sweep (resumable)
#
# One line per (task, k, example) in `results/fig2_raw.jsonl`. Interrupt freely;
# rerun this cell to continue where it stopped.

# %%
def load_done(path: str) -> dict:
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[(r["task"], r["k"], r["idx"])] = r
    return done


done = load_done(RAW_PATH)
todo = [r for r in records if (r["task"], r["k"], r["idx"]) not in done]
print(f"{len(done)} cached, {len(todo)} API calls to make")
lock = threading.Lock()


def work(rec):
    reply = api.query(client, rec["messages"], temperature=TEMPERATURE)
    pred = api.parse_answer(reply)
    out = dict(task=rec["task"], k=rec["k"], idx=rec["idx"], key=rec["key"],
               target=rec["target"], reply=reply.strip()[:64],
               pred=pred, correct=bool(pred == rec["target"]))
    with lock:
        with open(RAW_PATH, "a") as f:
            f.write(json.dumps(out) + "\n")
    return out


with ThreadPoolExecutor(max_workers=api.WORKERS) as pool:
    futures = [pool.submit(work, r) for r in todo]
    for fut in tqdm(as_completed(futures), total=len(futures),
                    desc="API sweep", unit="call"):
        fut.result()
print("sweep complete")

# %% [markdown]
# ## 4. Summary table — accuracy, SE, McNemar vs baseline
#
# McNemar uses the paper's design: the SAME fixed test set at every k, so each
# example's k=0 vs k>0 outcomes pair up. b = wrong→right flips, c = right→wrong.

# %%
import pandas as pd

from paper_tasks import binomial_se, mcnemar_exact

df = pd.DataFrame(load_done(RAW_PATH).values())
summary = []
for task, tdf in df.groupby("task"):
    base = tdf[tdf.k == 0].set_index("idx")["correct"]
    for k, g in tqdm(tdf.groupby("k"), total=tdf.k.nunique(),
                     desc=f"summarizing {task}", unit="cond", leave=False):
        acc = g["correct"].mean()
        rec = dict(task=task, k=int(k), n=len(g),
                   accuracy=acc, se=binomial_se(acc, len(g)))
        if k != 0 and len(base):
            m = g.set_index("idx")["correct"].reindex(base.index).dropna()
            b = int(((~base.loc[m.index]) & m).sum())
            c = int((base.loc[m.index] & (~m)).sum())
            rec.update(flips_wrong_to_right=b, flips_right_to_wrong=c,
                       mcnemar_p=mcnemar_exact(b, c))
        summary.append(rec)

sdf = pd.DataFrame(summary).sort_values(["task", "k"])
sdf.to_csv("results/fig2_summary.csv", index=False)
print(sdf.to_string(index=False))
print("\nwrote results/fig2_summary.csv")

# %% [markdown]
# ## 5. Figure 2 — accuracy vs k, one panel per task
#
# Overlaid for context: the paper's own dot-filler curves, transcribed from
# `plotting/plot_filler_accuracy.py` in the paper's code release
# (github.com/kaleybrauer/filler-token-reasoning). Caveats for the comparison:
# those runs used local 4-bit checkpoints (V3-0324-AWQ, Kimi-K2 W4A16), each
# model's own knowledge-filtered fact pool, and (for 1-fact) also k values we
# don't sweep — so the overlay says "how does V4 Flash's uplift compare in
# shape and size", not "same benchmark, different model". The paper's Task-1
# panel is DeepSeek V3 only, so 1-fact has no Kimi reference.

# %%
import math

import matplotlib.pyplot as plt

TASK_TITLES = {"1fact": "1-fact addition", "2fact": "2-fact addition"}

# accuracy in %, dots filler, restricted to our KS; n per condition as published
PAPER_REF = {
    "1fact": {
        "DeepSeek V3 (paper)": dict(
            n=800, marker="s", color="#d1495b",
            acc={0: 54.0, 5: 62.1, 10: 62.9, 25: 66.8, 50: 67.6, 100: 69.4}),
    },
    "2fact": {
        "DeepSeek V3 (paper)": dict(
            n=1500, marker="s", color="#d1495b",
            acc={0: 20.8, 5: 21.9, 10: 23.1, 25: 23.5, 50: 24.1, 100: 23.9}),
        "Kimi K2 (paper)": dict(
            n=1500, marker="^", color="#2a9d8f",
            acc={0: 27.3, 5: 29.8, 10: 31.7, 25: 35.1, 50: 33.7, 100: 34.3}),
    },
}

OURS_COLOR = "#4878d0"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, task in zip(axes, ["1fact", "2fact"]):
    g = sdf[sdf.task == task].sort_values("k")
    ks = [k for k in g.k if k != 0]
    xpos = {k: i for i, k in enumerate(ks)}

    base = g[g.k == 0]
    if len(base):
        b = base.iloc[0]
        ax.axhline(b.accuracy * 100, color="gray", ls="--", lw=1,
                   label="no filler (k=0), this work")
        ax.fill_between([-0.4, len(ks) - 0.6],
                        (b.accuracy - b.se) * 100, (b.accuracy + b.se) * 100,
                        color="gray", alpha=0.15, lw=0)

    # paper reference curves (muted, dashed; own k=0 baseline as dotted hline)
    for label, ref in PAPER_REF[task].items():
        rks = [k for k in ks if k in ref["acc"]]
        accs = [ref["acc"][k] for k in rks]
        ses = [math.sqrt((a / 100) * (1 - a / 100) / ref["n"]) * 100 for a in accs]
        ax.axhline(ref["acc"][0], color=ref["color"], ls=":", lw=0.9,
                   alpha=0.55, label="_nolegend_")
        ax.errorbar([xpos[k] for k in rks], accs, yerr=ses,
                    marker=ref["marker"], ms=4, capsize=2, lw=1.4, ls="--",
                    color=ref["color"], alpha=0.75, mfc="white", label=label)

    gk = g[g.k != 0]
    ax.errorbar([xpos[k] for k in gk.k], gk.accuracy * 100, yerr=gk.se * 100,
                marker="o", ms=4.5, capsize=3, lw=2.0, color=OURS_COLOR,
                zorder=5, label="DeepSeek V4 Flash (this work)")
    # asterisk over k values with a significant McNemar flip test
    for _, row in gk.iterrows():
        if row.get("mcnemar_p", 1.0) < 0.05:
            ax.annotate("*", (xpos[row.k], (row.accuracy + row.se) * 100 + 1),
                        ha="center", color=OURS_COLOR, zorder=5)

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.10)   # headroom for the asterisks
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(-0.4, len(ks) - 0.6)
    ax.set_xlabel("Number of Filler [k]")
    ax.set_ylabel("Accuracy (%)")
    n = int(g.n.max())
    ax.set_title(f"{TASK_TITLES[task]} (ours: n={n}/condition)")

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels),
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Filler-token uplift, dot filler (Fig. 2 replication) — "
             f"{api.API_MODEL} vs the paper's models", y=1.02)
fig.tight_layout()
fig.savefig("results/fig2_accuracy_vs_k.png", dpi=150, bbox_inches="tight")
print("wrote results/fig2_accuracy_vs_k.png")
plt.show()

# %%
