"""Exploratory analyses added after the main experiments (CPU only, no LLM calls).

  python explore.py bestofn     # is sampling just best-of-N from one fixed distribution?
  python explore.py leftovers   # what the bins look like at the end, and item size shapes
  python explore.py landscape   # full (a, b) grid of the distilled gap-band rule
  python explore.py ideas       # which coding ideas the LLM tried, and how often they worked

Everything reads the saved runs in results/runs/ and writes JSON to results/explore_*.json and
figures to report/figures/.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import datasets
from binpack import BASELINES, evaluate, weibull_instance, uniform_instance
from mechanism import gap_band, load_program

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RUNS = os.path.join(ROOT, "results", "runs")
FIG = os.path.join(ROOT, "report", "figures")
RES = os.path.join(ROOT, "results")
RUN_RE = re.compile(r"^(evolve|sample)_(claude-[a-z]+-5-5)(-high)?_b(\d+)_s(\d+)$")
BLUE, RED, GREY = "#2a6fdb", "#d1495b", "#777777"


def load_runs():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        m = RUN_RE.match(os.path.basename(d))
        if not m:
            continue
        recs = [json.loads(l) for l in open(os.path.join(d, "proposals.jsonl"))]
        recs.sort(key=lambda r: r["idx"])
        out.append({"id": os.path.basename(d), "mode": m.group(1), "model": m.group(2),
                    "effort": "high" if m.group(3) else "low", "budget": int(m.group(4)),
                    "recs": recs})
    return out


def bf_train():
    return evaluate(BASELINES["best_fit"], datasets.TRAIN)["mean_excess_pct"]


def scores(recs, fallback):
    """Training excess per proposal; invalid proposals count as no improvement."""
    return np.array([r["train_excess"] if r.get("valid") and r.get("train_excess") is not None
                     else fallback for r in recs])


# ---------------------------------------------------------------------------------------------
def run_bestofn():
    """If sample-mode proposals are independent draws from one distribution F, then the best
    score after N proposals is the minimum of N draws from F. We estimate F from all pooled
    Haiku-low sample proposals and compare the predicted curve with the observed runs."""
    BF = bf_train()
    runs = [r for r in load_runs() if r["model"] == "claude-haiku-5-5" and r["effort"] == "low"]
    samp = [r for r in runs if r["mode"] == "sample"]
    evo = [r for r in runs if r["mode"] == "evolve"]
    pool = np.concatenate([scores(r["recs"], BF) for r in samp])

    # Are the six sampling runs draws from the same distribution? (Kruskal-Wallis)
    kw = stats.kruskal(*[scores(r["recs"], BF) for r in samp])
    # Lag-1 autocorrelation within sample runs (independence check)
    ac = [float(np.corrcoef(s[:-1], s[1:])[0, 1]) for s in (scores(r["recs"], BF) for r in samp)]

    rng = np.random.default_rng(0)
    Nmax, B = 800, 2000
    draws = rng.choice(pool, size=(B, Nmax), replace=True)
    best = np.minimum(BF, np.minimum.accumulate(draws, axis=1))  # the seed is always kept
    lo, med, hi = np.percentile(best, [5, 50, 95], axis=0)
    p3 = float(np.mean(pool < 3.0))  # chance a single proposal is below 3%
    N = np.arange(1, Nmax + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    ax = axes[0]
    ax.fill_between(N, lo, hi, color=RED, alpha=0.15, label="i.i.d. model, 90% band")
    ax.plot(N, med, color=RED, lw=1.5, ls="--", label="i.i.d. model, median")
    for i, r in enumerate(samp):
        s = np.minimum(BF, np.minimum.accumulate(scores(r["recs"], BF)))
        ax.plot(np.arange(1, len(s) + 1), s, color=RED, lw=0.9, alpha=0.8,
                label="sample runs" if i == 0 else None)
    for i, r in enumerate(evo):
        s = np.minimum(BF, np.minimum.accumulate(scores(r["recs"], BF)))
        ax.plot(np.arange(1, len(s) + 1), s, color=BLUE, lw=0.9, alpha=0.8,
                label="evolve runs" if i == 0 else None)
    ax.axhline(BF, color=GREY, ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("LLM proposals N"); ax.set_ylabel("best train excess (%)")
    ax.set_title("best of N: observed vs independent-draw model", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=6.5)

    ax = axes[1]
    ax.plot(N, 1 - (1 - p3) ** N, color=RED, lw=1.5, label=f"sample, i.i.d. model (p={p3:.4f})")
    # empirical: fraction of runs that have reached <3% by N (runs shorter than N are dropped)
    for mode, rs, c in [("sample", samp, RED), ("evolve", evo, BLUE)]:
        frac, xs = [], []
        for n in N:
            alive = [r for r in rs if len(r["recs"]) >= n]
            if len(alive) < 3:
                break
            frac.append(np.mean([np.min(scores(r["recs"][:n], BF)) < 3.0 for r in alive]))
            xs.append(n)
        ax.step(xs, frac, where="post", color=c, lw=1.2, alpha=0.9, label=f"{mode}, observed runs")
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("LLM proposals N"); ax.set_ylabel("P(best excess < 3%)")
    ax.set_title("chance of reaching 3% within N proposals", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bestofn.pdf")); fig.savefig(os.path.join(FIG, "bestofn.png"), dpi=110)
    plt.close(fig)

    # how many observed sample-run curves stay inside the 90% band at their final N?
    inside = []
    for r in samp:
        s = np.minimum(BF, np.minimum.accumulate(scores(r["recs"], BF))); n = len(s)
        inside.append(bool(lo[n - 1] <= s[-1] <= hi[n - 1]))
    evo_inside = []
    for r in evo:
        s = np.minimum(BF, np.minimum.accumulate(scores(r["recs"], BF))); n = len(s)
        evo_inside.append({"run": r["id"], "final": float(s[-1]), "band": [float(lo[n - 1]), float(hi[n - 1])]})

    # Minimum attainable two-sided p for Mann-Whitney with 6 vs 6 (exact): 2 / C(12, 6)
    from math import comb
    out = {"pool_size": int(len(pool)), "p_single_below_3": p3,
           "n_for_95pct_below_3": int(np.ceil(np.log(0.05) / np.log(1 - p3))) if p3 > 0 else None,
           "pool_quantiles": {q: float(np.quantile(pool, q)) for q in (0.001, 0.01, 0.05, 0.1, 0.5)},
           "kruskal_between_sample_runs": {"H": float(kw.statistic), "p": float(kw.pvalue)},
           "lag1_autocorr_sample_runs": ac,
           "model_median_best_at": {n: float(med[n - 1]) for n in (10, 50, 100, 300, 800)},
           "model_band_at": {n: [float(lo[n - 1]), float(hi[n - 1])] for n in (100, 300, 800)},
           "sample_runs_final_inside_band": inside, "evolve_finals_vs_band": evo_inside,
           "mwu_min_two_sided_p_6v6": 2 / comb(12, 6), "bf_train": BF}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RES, "explore_bestofn.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------
def final_bins(inst, priority):
    C = inst.capacity
    bins = np.full(len(inst.items), C, dtype=float)
    for item in inst.items:
        valid = np.nonzero(bins - item >= 0)[0]
        s = np.asarray(priority(float(item), bins[valid]), dtype=float)
        bins[valid[int(np.argmax(s))]] -= item
    return bins[bins != C]  # leftover space of every used bin


def run_leftovers():
    """Distribution of leftover space at the end of packing, and why the 'awkward gap' band
    depends on the item size distribution."""
    d = json.load(open(os.path.join(RES, "mechanism_distill.json")))
    a, b, c = d["best_params"]
    H = {"best fit": BASELINES["best_fit"],
         "Haiku evolve (s2)": load_program(os.path.join(RUNS, "evolve_claude-haiku-5-5_b300_s2", "best_program.py")),
         "distilled band": gap_band(a, b, c),
         "FunSearch": BASELINES["funsearch_or"]}
    cols = {"best fit": GREY, "Haiku evolve (s2)": BLUE, "distilled band": "#2a9d8f", "FunSearch": "#e9a03b"}
    insts = [weibull_instance(5000, 1000 + i) for i in range(5)]
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    edges = np.arange(-0.5, 60.5, 2)
    for name, fn in H.items():
        left = np.concatenate([final_bins(I, fn) for I in insts])
        frac_full = float(np.mean(left == 0)); band = float(np.mean((left >= a * 100) & (left <= b * 100)))
        out[name] = {"bins": int(len(left)), "frac_exactly_full": frac_full,
                     "frac_leftover_in_band": band, "mean_leftover": float(left.mean()),
                     "frac_leftover_1_to_3": float(np.mean((left >= 1) & (left <= 3)))}
        h, _ = np.histogram(left, bins=edges)
        ax.step(edges[:-1] + 1, h / len(left), where="mid", color=cols[name], lw=1.3,
                label=f"{name}: {100*frac_full:.0f}% full, {100*band:.0f}% in band")
    ax.axvspan(a * 100, b * 100, color="#2a9d8f", alpha=0.08)
    ax.set_yscale("log"); ax.set_xlabel("leftover space in a used bin (C = 100)")
    ax.set_ylabel("share of bins"); ax.set_title("final bins, Weibull n = 5000 (5 instances)", fontsize=9)
    ax.legend(fontsize=6.3); ax.grid(alpha=0.3)

    ax = axes[1]
    w = np.concatenate([weibull_instance(5000, 1000 + i).items for i in range(5)])
    u = np.concatenate([uniform_instance(500, 4000 + i).items for i in range(10)])
    ax.hist(w / 100, bins=np.linspace(0, 1, 51), density=True, alpha=0.55, color=BLUE,
            label="Weibull(3, 45), C = 100")
    ax.hist(u / 150, bins=np.linspace(0, 1, 51), density=True, alpha=0.55, color=RED,
            label="uniform 20 to 100, C = 150")
    ax.axvspan(a, b, color="#2a9d8f", alpha=0.15, label=f"penalised gap band [{a}, {b}]")
    ax.set_xlabel("item size as a fraction of capacity"); ax.set_ylabel("density")
    ax.set_title("item sizes vs the gap band", fontsize=9); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
    out["weibull_items_below_band_top"] = float(np.mean(w / 100 <= b))
    out["uniform_items_below_band_top"] = float(np.mean(u / 150 <= b))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "leftovers.pdf")); fig.savefig(os.path.join(FIG, "leftovers.png"), dpi=110)
    plt.close(fig)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RES, "explore_leftovers.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------
def run_landscape():
    """Train excess of the hard gap-band rule (c = 1) over a finer (a, b) grid."""
    A = np.round(np.arange(0.0, 0.101, 0.01), 2)
    Bs = np.round(np.arange(0.05, 0.501, 0.025), 3)
    Z = np.full((len(A), len(Bs)), np.nan)
    for i, a in enumerate(A):
        for j, b in enumerate(Bs):
            if b <= a:
                continue
            Z[i, j] = evaluate(gap_band(float(a), float(b), 1.0), datasets.TRAIN)["mean_excess_pct"]
        print("a", a, np.nanmin(Z[i]), flush=True)
    BF = bf_train()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis_r",
                   extent=[Bs[0] - 0.0125, Bs[-1] + 0.0125, A[0] - 0.005, A[-1] + 0.005],
                   vmin=np.nanmin(Z), vmax=BF + 1)
    i, j = np.unravel_index(np.nanargmin(Z), Z.shape)
    ax.plot(Bs[j], A[i], "r*", ms=10)
    ax.contour(Bs, A, Z, levels=[BF], colors="white", linewidths=1, linestyles="--")
    cb = fig.colorbar(im, ax=ax); cb.set_label("train excess (%)")
    ax.set_xlabel("band upper edge b (fraction of C)"); ax.set_ylabel("band lower edge a")
    ax.set_title("gap-band rule (c = 1); dashed: equal to best fit", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "landscape.pdf")); fig.savefig(os.path.join(FIG, "landscape.png"), dpi=110)
    plt.close(fig)
    out = {"a": A.tolist(), "b": Bs.tolist(), "Z": [[None if np.isnan(v) else float(v) for v in row] for row in Z],
           "best": {"a": float(A[i]), "b": float(Bs[j]), "train": float(Z[i, j])},
           "frac_cells_beating_bf": float(np.nanmean(Z < BF)), "bf_train": BF}
    print(json.dumps(out["best"]), out["frac_cells_beating_bf"])
    json.dump(out, open(os.path.join(RES, "explore_landscape.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------
IDEAS = {
    "exact/near-fit bonus": r"(perfect|exact|near[_ ]?(full|fit|perfect))",
    "gap band / dead zone": r"(dead|awkward|band|zone|medium|mid)",
    "exponential or power": r"(np\.exp|\*\* ?[0-9.]+|np\.power)",
    "log or sqrt": r"(np\.log|np\.sqrt|math\.log|math\.sqrt)",
    "uses item size in thresholds": r"item ?[*/] ?[0-9.]|[0-9.] ?\* ?item",
    "special-cases the empty bin": r"(empty|== ?cap|>= ?cap|max\(bins\) ?==|bins ?== ?np\.max)",
    "neighbour differencing": r"\[1:\] ?-=|np\.diff",
}


def run_ideas():
    """Tag each valid Haiku-low proposal by regex on its code and compare success rates."""
    BF = bf_train()
    runs = [r for r in load_runs() if r["model"] == "claude-haiku-5-5" and r["effort"] == "low"]
    rows = []
    for r in runs:
        for p in r["recs"]:
            if not (p.get("valid") and p.get("code")):
                continue
            code = p["code"]
            body = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#"))
            tags = {k: bool(re.search(v, body, re.I)) for k, v in IDEAS.items()}
            n_lines = sum(1 for l in body.splitlines() if l.strip() and not l.strip().startswith(('"""', "'''")))
            rows.append({"mode": r["mode"], "excess": p["train_excess"], "beat": p["train_excess"] < BF - 1e-9,
                         "lines": n_lines, **tags})
    out = {"n": len(rows), "ideas": {}}
    for mode in ("sample", "evolve"):
        R = [x for x in rows if x["mode"] == mode]
        base = np.mean([x["beat"] for x in R])
        out["ideas"][mode] = {"base_beat_rate": float(base), "n": len(R)}
        for k in IDEAS:
            w = [x for x in R if x[k]]; wo = [x for x in R if not x[k]]
            if not w or not wo:
                continue
            tab = [[sum(x["beat"] for x in w), len(w) - sum(x["beat"] for x in w)],
                   [sum(x["beat"] for x in wo), len(wo) - sum(x["beat"] for x in wo)]]
            out["ideas"][mode][k] = {"share": len(w) / len(R), "beat_with": float(np.mean([x["beat"] for x in w])),
                                     "beat_without": float(np.mean([x["beat"] for x in wo])),
                                     "fisher_p": float(stats.fisher_exact(tab)[1]),
                                     "median_excess_with": float(np.median([x["excess"] for x in w]))}
    # code length vs quality
    L = np.array([x["lines"] for x in rows]); E = np.array([x["excess"] for x in rows])
    rho = stats.spearmanr(L, E)
    out["length_vs_excess_spearman"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}
    out["median_lines"] = {m: float(np.median([x["lines"] for x in rows if x["mode"] == m])) for m in ("sample", "evolve")}
    # Short table for the report (sample mode: the cleanest, no feedback)
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Idea found in the code & share & beat BF (with) & beat BF (without) & Fisher $p$ \\\\", "\\midrule"]
    for k in IDEAS:
        v = out["ideas"]["sample"].get(k)
        if v and v["share"] >= 0.05:  # rarer tags are too small to say anything
            pv = "$<$0.001" if v["fisher_p"] < 1e-3 else f"{v['fisher_p']:.3f}"
            lines.append(f"{k} & {100*v['share']:.0f}\\% & {100*v['beat_with']:.1f}\\% & {100*v['beat_without']:.1f}\\% & {pv} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(ROOT, "report", "tables", "ideas.tex"), "w").write("\n".join(lines) + "\n")
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RES, "explore_ideas.json"), "w"), indent=1)


if __name__ == "__main__":
    {"bestofn": run_bestofn, "leftovers": run_leftovers, "landscape": run_landscape,
     "ideas": run_ideas}[sys.argv[1]]()
