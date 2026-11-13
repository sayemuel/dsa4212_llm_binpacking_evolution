"""Aggregate search runs into the figures and tables used in the report.

  python analyze.py
Reads results/runs/*/{proposals.jsonl,summary.json}, results/test_eval.json,
results/disagreement.json. Writes report/figures/*.pdf|png, results/summary_tables.json
and report/tables/*.tex.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "report", "figures")
TAB = os.path.join(ROOT, "report", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

RUN_RE = re.compile(r"^(evolve|sample)_(claude-[a-z]+-5-5)(-high)?_b(\d+)_s(\d+)$")
BF_TRAIN = None  # filled from summaries
PRETTY = {"claude-haiku-5-5": "Haiku", "claude-opus-5-5": "Opus"}


def load_runs():
    runs = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*"))):
        m = RUN_RE.match(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        recs = [json.loads(l) for l in open(os.path.join(d, "proposals.jsonl"))]
        cfg = json.load(open(os.path.join(d, "config.json")))
        runs.append({"id": os.path.basename(d), "mode": m.group(1), "model": m.group(2),
                     "effort": cfg.get("effort", "low"), "budget": int(m.group(4)),
                     "seed": int(m.group(5)), "recs": recs,
                     "summary": json.load(open(os.path.join(d, "summary.json")))})
    return runs


def config_name(r):
    return f"{PRETTY[r['model']]}-{r['effort']} {r['mode']}"


def best_curve(recs, seed_excess):
    best, out = seed_excess, []
    for r in sorted(recs, key=lambda r: r["t"]):   # completion order
        if r["valid"] and r["train_excess"] is not None:
            best = min(best, r["train_excess"])
        out.append(best)
    return np.array(out)


def fig_convergence(runs):
    """Haiku-low evolve vs sample, all seeds (300- and 800-budget runs pooled
    over their common prefix, then the 800-runs alone to the end)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    col = {"evolve": "#2a6fdb", "sample": "#d1495b"}
    for ax, budgets, title in [(axes[0], (300, 800), "first 300 proposals (6 seeds each)"),
                               (axes[1], (800,), "800-proposal runs (3 seeds each)")]:
        for mode in ("evolve", "sample"):
            rs = [r for r in runs if r["model"] == "claude-haiku-5-5" and r["effort"] == "low"
                  and r["mode"] == mode and r["budget"] in budgets]
            if not rs:
                continue
            curves = [best_curve(r["recs"], r["summary"]["seed_train_excess"]) for r in rs]
            n = min(len(c) for c in curves)
            n = min(n, 300) if len(budgets) == 2 else n
            C = np.stack([c[:n] for c in curves])
            x = np.arange(1, n + 1)
            for c in C:
                ax.plot(x, c, color=col[mode], alpha=0.25, lw=0.8)
            ax.plot(x, np.median(C, 0), color=col[mode], lw=2, label=f"{mode} (median of {len(rs)})")
        ax.axhline(BF_TRAIN, color="grey", ls=":", lw=1, label="best fit")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("LLM proposals")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("best train excess over L1 (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "convergence.pdf"))
    fig.savefig(os.path.join(FIG, "convergence.png"), dpi=110)
    plt.close(fig)


def fig_disagreement(runs):
    path = os.path.join(ROOT, "results", "disagreement.json")
    if not os.path.exists(path):
        return {}
    rows = json.load(open(path))
    byrun = {r["id"]: r for r in runs}
    groups = defaultdict(list)
    for row in rows:
        if row["disagree"] is None or row["run"] not in byrun:
            continue
        groups[config_name(byrun[row["run"]])].append((row["disagree"], row["train_excess"]))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    palette = ["#2a6fdb", "#d1495b", "#3c9d5d", "#8a5cf6", "#e09f3e", "#555555"]
    stats = {}
    for i, (name, v) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        d = np.array([x[0] for x in v]); e = np.array([x[1] for x in v])
        big = len(v) < 500
        ax.scatter(np.maximum(d, 1e-4), e, s=18 if big else 5, alpha=0.8 if big else 0.35,
                   marker="D" if big else "o", color=palette[i % 6], label=f"{name} (n={len(v)})",
                   edgecolors="k" if big else "none", linewidths=0.3)
        stats[name] = {"n": len(v), "median_disagree": float(np.median(d)),
                       "frac_identical": float(np.mean(d == 0)),
                       "frac_gt_1pct": float(np.mean(d > 0.01)),
                       "frac_beat_bf": float(np.mean(e < BF_TRAIN - 1e-9))}
    ax.axhline(BF_TRAIN, color="grey", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_ylim(0, 12)
    ax.set_xlabel("disagreement with best fit on best-fit states (0 plotted at 1e-4)")
    ax.set_ylabel("train excess (%)")
    ax.legend(fontsize=6, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "disagreement.pdf"))
    fig.savefig(os.path.join(FIG, "disagreement.png"), dpi=110)
    plt.close(fig)
    return stats


def run_table(runs):
    table = []
    for r in runs:
        recs = r["recs"]; s = r["summary"]
        valid = [x for x in recs if x["valid"]]
        exc = [x["train_excess"] for x in valid]
        table.append({
            "run": r["id"], "config": config_name(r), "budget": r["budget"], "seed": r["seed"],
            "proposals": len(recs),
            "invalid_rate": float(np.mean([not x["valid"] for x in recs])),
            "beats_seed_rate": float(np.mean([e < s["seed_train_excess"] - 1e-9 for e in exc])) if exc else 0.0,
            "unique_behaviours": len({tuple(x["train_bins"]) for x in valid}),
            "best_train_excess": s["best_train_excess"],
            "selected_valid_excess": s["selected"]["valid_excess"],
            "usd": s["total_usd"], "input_tokens": s["total_input_tokens"],
            "output_tokens": s["total_output_tokens"], "wall_min": s["wall_seconds"] / 60,
            "usd_per_proposal": s["total_usd"] / max(1, len(recs)),
            "mean_output_tokens": float(np.mean([x["output_tokens"] for x in recs])),
            "stopped_early": s.get("stopped_early"),
        })
    return table


def group_table(table):
    groups = defaultdict(list)
    for t in table:
        groups[(t["config"], t["budget"])].append(t)
    out = {}
    for (cfg, b), rows in sorted(groups.items()):
        def ms(k):
            v = [r[k] for r in rows]
            return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out[f"{cfg} | b{b}"] = {"n_runs": len(rows), **{k: ms(k) for k in
                                ["best_train_excess", "selected_valid_excess", "invalid_rate",
                                 "beats_seed_rate", "unique_behaviours", "usd", "usd_per_proposal",
                                 "mean_output_tokens", "proposals"]}}
    return out


def test_table(runs):
    path = os.path.join(ROOT, "results", "test_eval.json")
    if not os.path.exists(path):
        return {}
    te = json.load(open(path))
    byrun = {r["id"]: r for r in runs}
    splits = list(dict.fromkeys(row["split"] for row in te))
    per = defaultdict(dict)
    for row in te:
        per[row["method"]][row["split"]] = row["mean_excess_pct"]
    # group run-methods by configuration -> mean, min, max over seeds
    rows = {}
    for m in ["first_fit", "best_fit", "funsearch_or", "best_fit_decreasing(offline)",
              "gap_band(distilled)"]:
        if m in per:
            rows[m] = {s: (per[m][s], None, None) for s in splits}
    cfg_methods = defaultdict(list)
    for m in per:
        if m in byrun:
            r = byrun[m]
            cfg_methods[f"{config_name(r)} b{r['budget']}"].append(m)
    for cfg, ms in sorted(cfg_methods.items()):
        rows[f"{cfg} (n={len(ms)})"] = {
            s: (float(np.mean([per[m][s] for m in ms])), float(np.min([per[m][s] for m in ms])),
                float(np.max([per[m][s] for m in ms]))) for s in splits}
    # LaTeX
    lines = ["\\begin{tabular}{l" + "r" * len(splits) + "}", "\\toprule",
             "Method & " + " & ".join(s.replace("_", "\\_") for s in splits) + " \\\\", "\\midrule"]
    for name, d in rows.items():
        cells = []
        for s in splits:
            m, lo, hi = d[s]
            cells.append(f"{m:.2f}" if lo is None or lo == hi else f"{m:.2f} \\scriptsize[{lo:.1f},{hi:.1f}]")
        lines.append(name.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "test.tex"), "w").write("\n".join(lines))
    return rows


def evolve_vs_sample_stats(runs):
    """Haiku-low: best train excess after n proposals per seed; Mann-Whitney
    (location) and Levene (spread) tests; first proposal reaching < 3%; pooled
    selected-program validation excess."""
    from scipy import stats
    out = {}
    hl = [r for r in runs if r["model"] == "claude-haiku-5-5" and r["effort"] == "low"]
    for n in (100, 300, 800):
        e = [best_curve(r["recs"], BF_TRAIN)[n - 1] for r in hl if r["mode"] == "evolve" and len(r["recs"]) >= n]
        s = [best_curve(r["recs"], BF_TRAIN)[n - 1] for r in hl if r["mode"] == "sample" and len(r["recs"]) >= n]
        out[f"best_at_{n}"] = {"evolve": e, "sample": s,
                               "mannwhitney_p": float(stats.mannwhitneyu(e, s).pvalue),
                               "levene_p": float(stats.levene(e, s).pvalue)}
    for mode in ("evolve", "sample"):
        first, sel = [], []
        for r in hl:
            if r["mode"] != mode:
                continue
            recs = sorted(r["recs"], key=lambda x: x["t"])
            first.append(next((i + 1 for i, x in enumerate(recs)
                               if x["valid"] and x["train_excess"] < 3.0), None))
            sel.append(r["summary"]["selected"]["valid_excess"])
        out[f"{mode}_first_below_3pct"] = first
        out[f"{mode}_selected_valid_mean_sd"] = (float(np.mean(sel)), float(np.std(sel, ddof=1)))
    return out


def main():
    global BF_TRAIN
    runs = load_runs()
    BF_TRAIN = runs[0]["summary"]["seed_train_excess"]
    fig_convergence(runs)
    dis = fig_disagreement(runs)
    table = run_table(runs)
    groups = group_table(table)
    tests = test_table(runs)
    evs = evolve_vs_sample_stats(runs)
    print(json.dumps(evs, indent=1))
    out = {"runs": table, "groups": groups, "disagreement": dis, "evolve_vs_sample": evs,
           "test": {k: {s: v[0] for s, v in d.items()} for k, d in tests.items()},
           "total_usd": float(sum(t["usd"] for t in table))}
    json.dump(out, open(os.path.join(ROOT, "results", "summary_tables.json"), "w"), indent=1)
    for k, v in groups.items():
        print(f"{k:32s} n={v['n_runs']} best_train={v['best_train_excess'][0]:.2f}±{v['best_train_excess'][1]:.2f} "
              f"sel_valid={v['selected_valid_excess'][0]:.2f}±{v['selected_valid_excess'][1]:.2f} "
              f"beat={v['beats_seed_rate'][0]:.2f} $/prop={v['usd_per_proposal'][0]:.4f} "
              f"outtok={v['mean_output_tokens'][0]:.0f} $={v['usd'][0]:.2f}")
    print("TOTAL USD (completed runs):", round(out["total_usd"], 3))


if __name__ == "__main__":
    main()
