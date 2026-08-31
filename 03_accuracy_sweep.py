"""
Figure 2 replication (2-fact addition panel): accuracy vs filler length k
on DeepSeek V4 Flash via the chat-completions API.

The behavioral sweep needs no hidden states, so it runs against the API
(paper: "uplift is robust to whether the model was evaluated with API or
locally with quantization", Appendix B). The lens scripts (04/05) run on
local weights. Cross-check one condition locally if you want to be extra
careful that the API serves the same snapshot you rent.

Design choices copied from the paper:
  * same FIXED test set for every (filler, k) condition -> per-example flips
    are comparable, McNemar applies (paper Fig. 2 caption + Sec. 3);
  * few-shot examples contain the same filler as the eval condition;
  * temperature 0, answer parsed as the first integer in the reply;
  * error bars: +/-1 SE under a binomial model.

Run (resumable — re-running skips already-answered examples):
    export DEEPSEEK_API_KEY=sk-...
    python 03_accuracy_sweep.py --n 300
    python 03_accuracy_sweep.py --plot-only          # re-plot from cache
    python 03_accuracy_sweep.py --fillers dots counting c-scram --ks 0 10 50

Output (results/):
    accuracy_raw.jsonl   one record per (example, filler, k) with the reply
    accuracy_summary.csv per-condition accuracy, SE, McNemar vs k=0
    fig2_accuracy_vs_k.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import paper_tasks as pt

DEFAULT_KS = [0, 1, 5, 10, 25, 50, 100]
DEFAULT_FILLERS = ["dots", "counting"]


# --------------------------------------------------------------------------- #
# API plumbing
# --------------------------------------------------------------------------- #
def make_client(base_url: str):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai  (any OpenAI-compatible client works)")
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("Set DEEPSEEK_API_KEY (or OPENAI_API_KEY) in your environment.")
    return OpenAI(api_key=key, base_url=base_url)


def query_one(client, api_model: str, messages, max_retries: int = 5) -> str:
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=0.0,
                max_tokens=8,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # rate limits / transient network
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return ""


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def condition_key(kind: str, k: int) -> str:
    return "baseline" if k == 0 else f"{kind}-{k}"


def load_done(path: str):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[(r["cond"], r["idx"])] = r
    return done


def run_sweep(args):
    client = make_client(args.base_url)
    dataset = pt.build_dataset(args.n, seed=args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "accuracy_raw.jsonl")
    done = load_done(raw_path)
    write_lock = threading.Lock()

    # baseline (k=0) runs once, not once per filler type
    conditions = [("none", 0)] if 0 in args.ks else []
    conditions += [(f, k) for f in args.fillers for k in args.ks if k != 0]

    todo = []
    for kind, k in conditions:
        cond = condition_key(kind, k)
        for ex in dataset:
            if (cond, ex.idx) not in done:
                todo.append((cond, kind, k, ex))
    print(f"{len(done)} cached, {len(todo)} calls to make "
          f"({len(conditions)} conditions x {args.n} examples)")

    def work(item):
        cond, kind, k, ex = item
        msgs = pt.build_messages(ex, kind, k)
        reply = query_one(client, args.api_model, msgs)
        pred = pt.parse_answer(reply)
        rec = dict(cond=cond, idx=ex.idx, elem_a=ex.elem_a, elem_b=ex.elem_b,
                   a1=ex.a1, a2=ex.a2, target=ex.target,
                   filler=kind, k=k, reply=reply.strip()[:64],
                   pred=pred, correct=bool(pred == ex.target))
        with write_lock:
            with open(raw_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    t0, n_done = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, it) for it in todo]
        for fut in as_completed(futures):
            fut.result()
            n_done += 1
            if n_done % 100 == 0:
                rate = n_done / (time.time() - t0)
                print(f"  {n_done}/{len(todo)}  ({rate:.1f} req/s)")
    print("sweep complete")


# --------------------------------------------------------------------------- #
# Summary + plot
# --------------------------------------------------------------------------- #
def summarize_and_plot(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_path = os.path.join(args.outdir, "accuracy_raw.jsonl")
    rows = list(load_done(raw_path).values())
    if not rows:
        sys.exit(f"no results in {raw_path}; run without --plot-only first")
    df = pd.DataFrame(rows)

    base = df[df.k == 0].set_index("idx")["correct"] if (df.k == 0).any() else None

    summary = []
    for (kind, k), g in df.groupby(["filler", "k"]):
        n = len(g)
        acc = g["correct"].mean()
        rec = dict(filler=kind, k=int(k), n=n, accuracy=acc,
                   se=pt.binomial_se(acc, n))
        if base is not None and k != 0:
            m = g.set_index("idx")["correct"].reindex(base.index).dropna()
            b = int(((~base.loc[m.index]) & m).sum())   # wrong -> right
            c = int((base.loc[m.index] & (~m)).sum())   # right -> wrong
            rec.update(flips_wrong_to_right=b, flips_right_to_wrong=c,
                       mcnemar_p=pt.mcnemar_exact(b, c))
        summary.append(rec)
    sdf = pd.DataFrame(summary).sort_values(["filler", "k"])
    csv = os.path.join(args.outdir, "accuracy_summary.csv")
    sdf.to_csv(csv, index=False)
    print(sdf.to_string(index=False))
    print(f"\nwrote {csv}")

    # ---- Figure 2 style: categorical x axis of k values ----
    ks = sorted(df.k.unique())
    xpos = {k: i for i, k in enumerate(ks)}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    base_row = sdf[sdf.k == 0]
    if len(base_row):
        b = base_row.iloc[0]
        ax.axhline(b.accuracy * 100, color="gray", ls="--", lw=1,
                   label=f"no filler ({b.accuracy*100:.1f}%)")
        ax.fill_between([-0.4, len(ks) - 0.6],
                        (b.accuracy - b.se) * 100, (b.accuracy + b.se) * 100,
                        color="gray", alpha=0.15, lw=0)
    for kind, g in sdf[sdf.k != 0].groupby("filler"):
        g = g.sort_values("k")
        ax.errorbar([xpos[k] for k in g.k], g.accuracy * 100, yerr=g.se * 100,
                    marker="o", ms=4, capsize=3, label=kind)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of Filler [k]")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"2-fact addition — {args.api_model} (n={int(sdf.n.max())}/condition)")
    ax.legend(frameon=False)
    fig.tight_layout()
    png = os.path.join(args.outdir, "fig2_accuracy_vs_k.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=300, help="examples per condition")
    ap.add_argument("--seed", type=int, default=0, help="test-set seed (keep fixed!)")
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    ap.add_argument("--fillers", nargs="+", default=DEFAULT_FILLERS,
                    choices=list(pt.FILLER_KINDS))
    ap.add_argument("--api-model", default="deepseek-v4-flash",
                    help="API model id (check your provider's id for V4 Flash)")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        run_sweep(args)
    summarize_and_plot(args)


if __name__ == "__main__":
    main()
