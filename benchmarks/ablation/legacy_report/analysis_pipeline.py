from __future__ import annotations

import glob
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata, wilcoxon


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "sources"
OUT = ROOT / "analysis_outputs"
OUT.mkdir(exist_ok=True)

DATASET_FILES = {
    "Prins": SRC / "core" / "VNDZou_Stage2_PRINS.xlsx",
    "Barreto": SRC / "core" / "VNDZou_Stage2_BARRETO.xlsx",
    "Tuzun": SRC / "core" / "VNDZou_Stage2_TUZUN.xlsx",
}
VND0_SUMMARY_FILES = {
    "Prins": SRC / "core" / "VND0_PRINS_Summary.xlsx",
    "Barreto": SRC / "core" / "VND0_BARRETO_Summary.xlsx",
    "Tuzun": SRC / "core" / "VND0_TUZUN_Summary.xlsx",
}


def finite(x):
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def as_float(x):
    return float(x) if finite(x) else np.nan


def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for j, k in enumerate(order):
        adjusted = min(1.0, (m - j) * p[k])
        running = max(running, adjusted)
        out[k] = running
    return out


def paired_rank_biserial(diff):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if not len(d):
        return 0.0
    r = rankdata(np.abs(d), method="average")
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def safe_wilcoxon(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.allclose(a, b):
        return np.nan, 0.0
    try:
        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto")
        return float(res.pvalue), paired_rank_biserial(a - b)
    except ValueError:
        return np.nan, paired_rank_biserial(a - b)


def bootstrap_mean_ci(values, seed=20260722, reps=20000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    rng = np.random.default_rng(seed + len(x))
    means = np.empty(reps)
    for i in range(reps):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def cliffs_delta(x, y):
    """Positive values mean x tends to be larger than y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if not len(x) or not len(y):
        return np.nan
    gt = sum(np.sum(v > y) for v in x)
    lt = sum(np.sum(v < y) for v in x)
    return float((gt - lt) / (len(x) * len(y)))


def wtl(proposed, comparator, decimals=2):
    p = np.round(np.asarray(proposed, dtype=float), decimals)
    c = np.round(np.asarray(comparator, dtype=float), decimals)
    ok = np.isfinite(p) & np.isfinite(c)
    return int(np.sum(p[ok] < c[ok])), int(np.sum(p[ok] == c[ok])), int(np.sum(p[ok] > c[ok]))


def normalize_barreto(name):
    s = str(name).strip()
    mappings = {
        "Christ-50-5": "Ch50-5",
        "Christ-75-10": "Ch75-10",
        "Christ-100-10": "Ch100-10",
        "Gaskell-21-5": "Gaskell21-5",
        "Gaskell-29-5": "Gaskell29-5",
        "Gaskell-32-5b": "Gaskell32-5",
        "Gaskell-32-5": "Gaskell32-5",
        "Gaskell-36-5": "Gaskell36-5",
        "Min-27-5": "Mi27-5",
        "Min-134-8": "Mi134-8",
        "Or-117-14": "Or117-14",
    }
    return mappings.get(s, s.replace("Christ-", "Ch").replace("Gaskell-", "Gaskell").replace("Min-", "Mi").replace("Or-", "Or"))


def normalize_lit_instance(dataset, row):
    raw = row["Problem Number"]
    if dataset == "Barreto":
        return normalize_barreto(raw)
    if dataset == "Prins":
        s = str(raw).strip().replace("BIS", "bis")
        suffix = s.split("-")[1]
        return f"{s}-{suffix}"
    num = str(raw).strip()
    if num.endswith(".0"):
        num = num[:-2]
    depots = int(float(row["Depots"]))
    return f"P{num}-{depots}"


def load_literature():
    path = SRC / "literature" / "Lit_Results.xlsx"
    frames = []
    for dataset in ["Barreto", "Prins", "Tuzun"]:
        d = pd.read_excel(path, sheet_name=dataset, header=1)
        d["Dataset"] = dataset
        d["Instance"] = [normalize_lit_instance(dataset, r) for _, r in d.iterrows()]
        if dataset != "Tuzun":
            d["Legacy BKS"] = pd.to_numeric(d["BKS (Zou et al. 2024)"], errors="coerce")
        else:
            d["Legacy BKS"] = pd.to_numeric(d["BKS"], errors="coerce")
        # RLHEA best is used as the common 2024 benchmark reference because the
        # legacy BKS column predates some improvements reported in the same table.
        d["Benchmark BKS"] = pd.to_numeric(d["RLHEA Best"], errors="coerce")
        frames.append(d)
    lit = pd.concat(frames, ignore_index=True, sort=False)

    # Add the 2025 deterministic single-scenario MS-VNS result as a secondary
    # recent comparator; the main stochastic problem in that paper is not mixed
    # with these deterministic tables.
    supp = SRC / "literature" / "Osorio_Mora_2025_Supplement.xlsx"
    sheets = {"Barreto": "SM-7 Barreto", "Prins": "SM-6 Prodhon", "Tuzun": "SM-5 Tuzun-Burke"}
    ms_rows = []
    for ds, sh in sheets.items():
        d = pd.read_excel(supp, sheet_name=sh)
        for _, r in d.iterrows():
            if pd.isna(r.get("Instance")):
                continue
            raw_instance = str(r.get("Instance")).strip()
            if "avg" in raw_instance.lower():
                continue
            if ds == "Barreto":
                instance = normalize_barreto(r["Instance"])
            elif ds == "Prins":
                s = str(r["Instance"]).strip().replace("BIS", "bis")
                instance = f"{s}-{s.split('-')[1]}"
            else:
                num = str(r["Instance"]).strip().replace(".0", "")
                instance = f"P{num}-{int(float(r['D']))}"
            ms_rows.append({
                "Dataset": ds,
                "Instance": instance,
                "MS-VNS Best": as_float(r.get("MS_VNS_Best")),
                "MS-VNS Avg.": as_float(r.get("MS_VNS_Avg")),
                "MS-VNS CPU": as_float(r.get("MS_VNS_Time")),
            })
    ms = pd.DataFrame(ms_rows)
    lit = lit.merge(ms, on=["Dataset", "Instance"], how="left")
    return lit


def load_vndzou():
    summaries, runs = [], []
    for ds, f in DATASET_FILES.items():
        s = pd.read_excel(f, sheet_name="02_Instance_Summary")
        s = pd.DataFrame({
            "Dataset": ds,
            "Instance": s["Instance"].astype(str),
            "Customers": pd.to_numeric(s.get("Customers"), errors="coerce"),
            "Candidate Depots": pd.to_numeric(s.get("Candidate Depots"), errors="coerce"),
            "Best": pd.to_numeric(s["Best Final Latency"], errors="coerce"),
            "Mean": pd.to_numeric(s["Mean Final Latency"], errors="coerce"),
            "Std": pd.to_numeric(s["Std Final Latency"], errors="coerce"),
            "CPU": pd.to_numeric(s["Mean CPU Time"], errors="coerce"),
            "Runs": pd.to_numeric(s["Number of Runs"], errors="coerce"),
            "Feasibility": pd.to_numeric(s["Feasibility Rate (%)"], errors="coerce"),
        })
        summaries.append(s)
        r = pd.read_excel(f, sheet_name="04_Run_Level_Details")
        r = pd.DataFrame({
            "Dataset": ds,
            "Instance": r["Instance"].astype(str),
            "Run": pd.to_numeric(r["Run"], errors="coerce"),
            "Latency": pd.to_numeric(r["Total Latency"], errors="coerce"),
            "CPU": pd.to_numeric(r["CPU Time"], errors="coerce"),
        })
        runs.append(r)
    return pd.concat(summaries, ignore_index=True), pd.concat(runs, ignore_index=True)


def load_vnd0():
    summaries, runs = [], []
    raw_glob = SRC / "vnd0_raw"
    for ds in ["Prins", "Barreto", "Tuzun"]:
        raw_summary = []
        for f in sorted((raw_glob / ds).glob("*.xlsx")):
            try:
                fs = pd.read_excel(f, sheet_name="Final_Summary").iloc[0]
                raw_summary.append(fs)
                rd = pd.read_excel(f, sheet_name="All_Run_Level_Details")
                rr = pd.DataFrame({
                    "Dataset": ds,
                    "Instance": rd["Instance"].astype(str),
                    "Run": pd.to_numeric(rd["Run"], errors="coerce"),
                    "Latency": pd.to_numeric(rd["Total Latency"], errors="coerce"),
                    "CPU": pd.to_numeric(rd["CPU Time"], errors="coerce"),
                })
                if "Feasible" in rd:
                    rr = rr[rd["Feasible"].astype(bool).to_numpy()]
                runs.append(rr)
            except Exception as exc:
                print(f"WARN raw read {f}: {exc}")
        raw_df = pd.DataFrame(raw_summary)
        fallback = pd.read_excel(VND0_SUMMARY_FILES[ds], sheet_name=0)
        if len(raw_df):
            combined = pd.concat([raw_df, fallback[~fallback["Instance"].astype(str).isin(raw_df["Instance"].astype(str))]], ignore_index=True)
        else:
            combined = fallback
        s = pd.DataFrame({
            "Dataset": ds,
            "Instance": combined["Instance"].astype(str),
            "Best": pd.to_numeric(combined["Best Latency"], errors="coerce"),
            "Mean": pd.to_numeric(combined["Mean Latency"], errors="coerce"),
            "Std": pd.to_numeric(combined["Std Latency"], errors="coerce"),
            "CPU": pd.to_numeric(combined["Mean CPU Time"], errors="coerce"),
            "Runs": pd.to_numeric(combined["Total Completed Runs"], errors="coerce"),
            "Feasibility": pd.to_numeric(combined["Feasibility Rate (%)"], errors="coerce"),
        })
        summaries.append(s)
    return pd.concat(summaries, ignore_index=True), pd.concat(runs, ignore_index=True)


def primary_comparison(v0, vz, lit):
    z = vz.merge(v0, on=["Dataset", "Instance"], how="outer", suffixes=("_Z", "_0"), indicator=True)
    bks = lit[["Dataset", "Instance", "Benchmark BKS"]].drop_duplicates()
    z = z.merge(bks, on=["Dataset", "Instance"], how="left")
    z["Best improvement (%)"] = 100 * (z["Best_0"] - z["Best_Z"]) / z["Best_0"]
    z["Mean improvement (%)"] = 100 * (z["Mean_0"] - z["Mean_Z"]) / z["Mean_0"]
    z["CPU ratio Z/0"] = z["CPU_Z"] / z["CPU_0"]
    z["CV_0"] = z["Std_0"] / z["Mean_0"]
    z["CV_Z"] = z["Std_Z"] / z["Mean_Z"]
    z["CV ratio Z/0"] = z["CV_Z"] / z["CV_0"].replace(0, np.nan)
    z["Gap BKS VND0 (%)"] = 100 * (z["Best_0"] - z["Benchmark BKS"]) / z["Benchmark BKS"]
    z["Gap BKS VNDzou (%)"] = 100 * (z["Best_Z"] - z["Benchmark BKS"]) / z["Benchmark BKS"]
    z["BKS hit VND0"] = np.round(z["Best_0"], 2) <= np.round(z["Benchmark BKS"], 2)
    z["BKS hit VNDzou"] = np.round(z["Best_Z"], 2) <= np.round(z["Benchmark BKS"], 2)
    z["Size"] = pd.cut(z["Customers"], [0, 50, 100, np.inf], labels=["Small (≤50)", "Medium (51–100)", "Large (>100)"], include_lowest=True)
    z.to_csv(OUT / "vnd0_vs_vndzou_instance.csv", index=False)

    rows = []
    for ds in ["Barreto", "Prins", "Tuzun", "Overall"]:
        d = z[(z["Dataset"] == ds) if ds != "Overall" else z["Dataset"].notna()]
        d = d[d["_merge"] == "both"].copy()
        for metric, label in [("Best", "best-of-runs"), ("Mean", "run mean")]:
            p, rbc = safe_wilcoxon(d[f"{metric}_0"], d[f"{metric}_Z"])
            wins, ties, losses = wtl(d[f"{metric}_Z"], d[f"{metric}_0"])
            imp = 100 * (d[f"{metric}_0"] - d[f"{metric}_Z"]) / d[f"{metric}_0"]
            ci_lo, ci_hi = bootstrap_mean_ci(imp)
            rows.append({
                "Dataset": ds,
                "Metric": label,
                "n": len(d),
                "VNDzou wins": wins,
                "Ties": ties,
                "VND0 wins": losses,
                "Mean improvement (%)": imp.mean(),
                "Mean improvement 95% CI lower": ci_lo,
                "Mean improvement 95% CI upper": ci_hi,
                "Median improvement (%)": imp.median(),
                "Wilcoxon p": p,
                "Rank-biserial (positive favors VNDzou)": rbc,
                "Median CPU ratio Z/0": d["CPU ratio Z/0"].median(),
                "Mean CPU ratio Z/0": d["CPU ratio Z/0"].mean(),
                "Median CV ratio Z/0": d["CV ratio Z/0"].median(),
                "BKS hits VND0": int(d["BKS hit VND0"].sum()),
                "BKS hits VNDzou": int(d["BKS hit VNDzou"].sum()),
            })
    summary = pd.DataFrame(rows)
    summary["Wilcoxon p (Holm)"] = np.nan
    for metric, idx in summary.groupby("Metric").groups.items():
        summary.loc[idx, "Wilcoxon p (Holm)"] = holm_adjust(summary.loc[idx, "Wilcoxon p"])
    summary.to_csv(OUT / "vnd0_vs_vndzou_summary.csv", index=False)

    size_rows = []
    d = z[z["_merge"] == "both"].copy()
    for size, g in d.groupby("Size", observed=True):
        size_rows.append({
            "Size": str(size), "n": len(g),
            "Mean best improvement (%)": g["Best improvement (%)"].mean(),
            "Mean run-mean improvement (%)": g["Mean improvement (%)"].mean(),
            "Median CPU ratio Z/0": g["CPU ratio Z/0"].median(),
        })
    pd.DataFrame(size_rows).to_csv(OUT / "vnd0_vs_vndzou_size.csv", index=False)
    return z, summary


def run_level_tests(v0runs, vzruns):
    rows = []
    keys = sorted(set(map(tuple, v0runs[["Dataset", "Instance"]].drop_duplicates().values)).intersection(
        set(map(tuple, vzruns[["Dataset", "Instance"]].drop_duplicates().values))))
    for ds, ins in keys:
        x = v0runs.loc[(v0runs.Dataset == ds) & (v0runs.Instance == ins), "Latency"].dropna().to_numpy()
        y = vzruns.loc[(vzruns.Dataset == ds) & (vzruns.Instance == ins), "Latency"].dropna().to_numpy()
        if len(x) and len(y):
            res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
            # Positive Cliff's delta is defined here as VND0 > VNDzou, favoring VNDzou.
            delta = cliffs_delta(x, y)
            rows.append({
                "Dataset": ds, "Instance": ins, "n VND0": len(x), "n VNDzou": len(y),
                "Mean VND0": np.mean(x), "Mean VNDzou": np.mean(y),
                "Mann-Whitney p": res.pvalue,
                "Cliff delta (positive favors VNDzou)": delta,
                "Mean improvement (%)": 100 * (np.mean(x) - np.mean(y)) / np.mean(x),
            })
    d = pd.DataFrame(rows)
    d["p Holm within dataset"] = np.nan
    for ds, idx in d.groupby("Dataset").groups.items():
        d.loc[idx, "p Holm within dataset"] = holm_adjust(d.loc[idx, "Mann-Whitney p"])
    d["Direction"] = np.where(d["Mean VNDzou"] < d["Mean VND0"], "VNDzou", np.where(d["Mean VNDzou"] > d["Mean VND0"], "VND0", "Tie"))
    d["Significant direction"] = np.where(d["p Holm within dataset"] < 0.05, d["Direction"], "Not significant")
    d.to_csv(OUT / "run_level_mannwhitney.csv", index=False)
    counts = d.groupby(["Dataset", "Significant direction"]).size().unstack(fill_value=0).reset_index()
    counts.to_csv(OUT / "run_level_significance_counts.csv", index=False)
    return d, counts


def literature_comparison(vz, lit):
    d = vz.merge(lit, on=["Dataset", "Instance"], how="left")
    d.to_csv(OUT / "literature_merged_instance.csv", index=False)
    methods = ["RGA", "MA", "GBILS", "SA-VND0", "SA-VND1", "SA-VND2", "M-ILS", "RLHEA", "MS-VNS"]
    rows = []
    for ds in ["Barreto", "Prins", "Tuzun", "Overall"]:
        g = d[(d.Dataset == ds) if ds != "Overall" else d.Dataset.notna()]
        for method in methods:
            for prop_col, lit_suffix, metric in [("Best", "Best", "best-of-runs"), ("Mean", "Avg.", "run mean")]:
                col = f"{method} {lit_suffix}"
                if col not in g:
                    continue
                q = g[[prop_col, col]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(q) < 2:
                    continue
                wins, ties, losses = wtl(q[prop_col], q[col])
                p, rbc = safe_wilcoxon(q[col], q[prop_col])  # positive favors VNDzou
                imp = 100 * (q[col] - q[prop_col]) / q[col]
                rows.append({
                    "Dataset": ds, "Method": method, "Metric": metric, "n": len(q),
                    "VNDzou wins": wins, "Ties": ties, "Literature wins": losses,
                    "Mean VNDzou improvement (%)": imp.mean(),
                    "Median VNDzou improvement (%)": imp.median(),
                    "Wilcoxon p": p,
                    "Rank-biserial (positive favors VNDzou)": rbc,
                })
    comp = pd.DataFrame(rows)
    # Correct the method family within each dataset and metric.
    comp["Wilcoxon p (Holm)"] = np.nan
    for (_, _), idx in comp.groupby(["Dataset", "Metric"]).groups.items():
        comp.loc[idx, "Wilcoxon p (Holm)"] = holm_adjust(comp.loc[idx, "Wilcoxon p"])
    comp.to_csv(OUT / "literature_comparison_summary.csv", index=False)

    rank_rows = []
    for metric, prop_col, suffix in [("best-of-runs", "Best", "Best"), ("run mean", "Mean", "Avg.")]:
        cols = {"VNDzou": prop_col}
        for method in methods:
            col = f"{method} {suffix}"
            if col in d:
                cols[method] = col
        rank_accum = {m: [] for m in cols}
        for _, row in d.iterrows():
            vals, labels = [], []
            for m, col in cols.items():
                val = as_float(row.get(col))
                if finite(val):
                    vals.append(val)
                    labels.append(m)
            if len(vals) >= 2:
                ranks = rankdata(vals, method="average")
                for m, r in zip(labels, ranks):
                    rank_accum[m].append(r)
        for m, ranks in rank_accum.items():
            rank_rows.append({"Metric": metric, "Method": m, "n": len(ranks), "Average rank": np.mean(ranks) if ranks else np.nan})
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(OUT / "literature_average_ranks.csv", index=False)

    # Friedman test on the complete, directly comparable 30-run methods.
    friedman = []
    core = ["VNDzou", "SA-VND0", "SA-VND1", "SA-VND2", "M-ILS", "RLHEA"]
    for metric, prop_col, suffix in [("best-of-runs", "Best", "Best"), ("run mean", "Mean", "Avg.")]:
        cdict = {"VNDzou": prop_col, **{m: f"{m} {suffix}" for m in core if m != "VNDzou"}}
        q = d[list(cdict.values())].apply(pd.to_numeric, errors="coerce").dropna()
        stat, p = friedmanchisquare(*[q[col].to_numpy() for col in cdict.values()])
        friedman.append({"Metric": metric, "Methods": ", ".join(cdict), "n": len(q), "Friedman chi-square": stat, "p": p})
    pd.DataFrame(friedman).to_csv(OUT / "literature_friedman.csv", index=False)
    return d, comp, rank_df


def load_ablation():
    rows = []
    for f in sorted((SRC / "ablation").glob("*.xlsx")):
        cfg = f.stem
        d = pd.read_excel(f, sheet_name=0)
        for _, r in d.iterrows():
            rows.append({
                "Configuration": cfg,
                "Instance": str(r["Instance"]),
                "Best": as_float(r.get("Best Latency")),
                "Mean": as_float(r.get("Mean Latency")),
                "Std": as_float(r.get("Std Latency")),
                "CPU": as_float(r.get("Mean CPU Time")),
                "BKS": as_float(r.get("Reference BKS")),
                "Runs": as_float(r.get("Total Completed Runs", r.get("Number of Runs"))),
                "Feasibility": as_float(r.get("Feasibility Rate (%)")),
            })
    d = pd.DataFrame(rows)
    d["Best gap (%)"] = 100 * (d["Best"] - d["BKS"]) / d["BKS"]
    d["Mean gap (%)"] = 100 * (d["Mean"] - d["BKS"]) / d["BKS"]
    d.to_csv(OUT / "ablation_instance.csv", index=False)
    return d


def ablation_analysis(d):
    complete = [c for c, g in d.groupby("Configuration") if g["Instance"].nunique() == 10]
    rank_best = []
    rank_mean = []
    for ins, g in d[d.Configuration.isin(complete)].groupby("Instance"):
        gb = g.set_index("Configuration")["Best"].reindex(complete)
        gm = g.set_index("Configuration")["Mean"].reindex(complete)
        rank_best.extend(zip(complete, rankdata(gb, method="average")))
        rank_mean.extend(zip(complete, rankdata(gm, method="average")))
    rb = pd.DataFrame(rank_best, columns=["Configuration", "Rank"]).groupby("Configuration")["Rank"].mean()
    rm = pd.DataFrame(rank_mean, columns=["Configuration", "Rank"]).groupby("Configuration")["Rank"].mean()
    rows = []
    for cfg, g in d.groupby("Configuration"):
        for base in ["VND0", "VNDZou"]:
            b = d[d.Configuration == base][["Instance", "Best", "Mean"]]
            m = g.merge(b, on="Instance", suffixes=("", "_base"))
            bw, bt, bl = wtl(m["Best"], m["Best_base"]) if len(m) else (0, 0, 0)
            mw, mt, ml = wtl(m["Mean"], m["Mean_base"]) if len(m) else (0, 0, 0)
            if base == "VND0":
                v0_best = f"{bw}/{bt}/{bl}"
                v0_mean = f"{mw}/{mt}/{ml}"
            else:
                vz_best = f"{bw}/{bt}/{bl}"
                vz_mean = f"{mw}/{mt}/{ml}"
        rows.append({
            "Configuration": cfg, "n": g.Instance.nunique(),
            "Mean best gap (%)": g["Best gap (%)"].mean(),
            "Median best gap (%)": g["Best gap (%)"].median(),
            "Mean run-mean gap (%)": g["Mean gap (%)"].mean(),
            "Median CPU": g["CPU"].median(), "Mean CPU": g["CPU"].mean(),
            "BKS hits": int((np.round(g["Best"], 2) <= np.round(g["BKS"], 2)).sum()),
            "Feasibility mean (%)": g["Feasibility"].mean(),
            "Complete-design best rank": rb.get(cfg, np.nan),
            "Complete-design mean rank": rm.get(cfg, np.nan),
            "Best W/T/L vs VND0": v0_best, "Mean W/T/L vs VND0": v0_mean,
            "Best W/T/L vs VNDzou": vz_best, "Mean W/T/L vs VNDzou": vz_mean,
        })
    s = pd.DataFrame(rows)

    # Pareto frontier in the complete ten-instance design: lower gap and CPU are better.
    s["Pareto efficient"] = False
    idx = s[s.n == 10].index
    for i in idx:
        dominated = False
        for j in idx:
            if i == j:
                continue
            if (s.loc[j, "Mean best gap (%)"] <= s.loc[i, "Mean best gap (%)"] and
                    s.loc[j, "Median CPU"] <= s.loc[i, "Median CPU"] and
                    (s.loc[j, "Mean best gap (%)"] < s.loc[i, "Mean best gap (%)"] or
                     s.loc[j, "Median CPU"] < s.loc[i, "Median CPU"])):
                dominated = True
                break
        s.loc[i, "Pareto efficient"] = not dominated
    s.to_csv(OUT / "ablation_summary.csv", index=False)

    contrasts = [
        ("VND0", "VND0-H3", "Construction: ECWA → H3"),
        ("VND0", "VND0-SO", "Strategic oscillation"),
        ("VND0-SO", "VND0-SO-DEPOT", "Depot-swap on VND0-SO"),
        ("VND0", "VND0-CAND20", "Candidate list δ=20"),
        ("VND0-CAND20", "VND0-Adv-CAND20", "Adaptive operator ordering"),
        ("VND0-CAND20", "VND0-Q-CAND20", "Q-learning operator ordering"),
        ("VNDZou-SO-DEPOT", "VNDZou-Q-SO-DEPOT", "Q-learning on VNDzou-SO-DEPOT"),
        ("VNDZou-SO-DEPOT", "VNDZou-SO-DEPOT-CAND20", "Candidate filtering on VNDzou-SO-DEPOT"),
        ("VNDZou-CAND20", "VNDZou-CAND30", "Candidate label CAND20 → CAND30"),
    ]
    cr = []
    for base, variant, label in contrasts:
        a = d[d.Configuration == base][["Instance", "Best", "Mean", "CPU"]]
        b = d[d.Configuration == variant][["Instance", "Best", "Mean", "CPU"]]
        m = a.merge(b, on="Instance", suffixes=("_base", "_variant"))
        if not len(m):
            continue
        pb, rbv = safe_wilcoxon(m["Best_base"], m["Best_variant"])
        pm, rmv = safe_wilcoxon(m["Mean_base"], m["Mean_variant"])
        cr.append({
            "Contrast": label, "Base": base, "Variant": variant, "n": len(m),
            "Best improvement (%)": (100 * (m.Best_base - m.Best_variant) / m.Best_base).mean(),
            "Mean improvement (%)": (100 * (m.Mean_base - m.Mean_variant) / m.Mean_base).mean(),
            "Median CPU ratio variant/base": (m.CPU_variant / m.CPU_base).median(),
            "Best W/T/L": "/".join(map(str, wtl(m.Best_variant, m.Best_base))),
            "Mean W/T/L": "/".join(map(str, wtl(m.Mean_variant, m.Mean_base))),
            "Best Wilcoxon p": pb, "Best rank-biserial": rbv,
            "Mean Wilcoxon p": pm, "Mean rank-biserial": rmv,
        })
    cr = pd.DataFrame(cr)
    cr["Best p (Holm)"] = holm_adjust(cr["Best Wilcoxon p"])
    cr["Mean p (Holm)"] = holm_adjust(cr["Mean Wilcoxon p"])
    cr.to_csv(OUT / "ablation_component_contrasts.csv", index=False)
    return s, cr


def plot_primary(instance):
    d = instance[instance["_merge"] == "both"].copy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    colors = {"Barreto": "#0B7285", "Prins": "#1864AB", "Tuzun": "#5F3DC4"}
    for ax, ds in zip(axes, ["Barreto", "Prins", "Tuzun"]):
        g = d[d.Dataset == ds].sort_values("Mean improvement (%)")
        ax.barh(g.Instance, g["Mean improvement (%)"], color=np.where(g["Mean improvement (%)"] >= 0, colors[ds], "#C92A2A"))
        ax.axvline(0, color="#343A40", lw=0.8)
        ax.set_title(f"{ds} (n={len(g)})", loc="left", fontweight="bold")
        ax.set_xlabel("Run-mean latency improvement of VNDzou over VND0 (%)")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", color="#DEE2E6", linewidth=0.6)
    fig.suptitle("Instance-level solution-quality change", fontsize=16, fontweight="bold")
    fig.savefig(OUT / "fig_primary_improvement.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    ds = d.groupby("Dataset").agg(
        mean_improvement=("Mean improvement (%)", "mean"),
        cpu_ratio=("CPU ratio Z/0", "median"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for _, r in ds.iterrows():
        ax.scatter(r.cpu_ratio, r.mean_improvement, s=120, color=colors[r.Dataset], edgecolor="white", linewidth=1.2)
        ax.annotate(r.Dataset, (r.cpu_ratio, r.mean_improvement), xytext=(6, 6), textcoords="offset points", fontsize=11, fontweight="bold")
    ax.axhline(0, color="#495057", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Median CPU ratio (VNDzou / VND0, log scale)")
    ax.set_ylabel("Mean run-mean latency improvement (%)")
    ax.set_title("Quality–runtime trade-off by benchmark set", loc="left", fontsize=15, fontweight="bold")
    ax.grid(color="#DEE2E6", linewidth=0.6)
    fig.savefig(OUT / "fig_quality_runtime.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(summary):
    s = summary[summary.n == 10].copy()
    fig, ax = plt.subplots(figsize=(10.2, 7.2), constrained_layout=True)
    colors = np.where(s["Pareto efficient"], "#0B7285", "#ADB5BD")
    ax.scatter(s["Median CPU"], s["Mean best gap (%)"], s=95, c=colors, edgecolor="white", linewidth=1.0)

    # Short labels and log-scaled axes keep the dense low-gap cluster legible.
    # Full configuration names and values are reported immediately before this
    # figure in the ablation table.
    short_labels = {
        "VND0": "0",
        "VND0-H3": "0-H3",
        "VND0-SO": "0-SO",
        "VND0-SO-DEPOT": "0-SO-D",
        "VND0-CAND20": "0-C20",
        "VND0-Adv-CAND20": "0-Adv-C20",
        "VND0-Q-CAND20": "0-Q-C20",
        "VNDZou": "Z",
        "VNDZou-CAND20": "Z-C20",
        "VNDZou-SO-DEPOT": "Z-SO-D",
        "VNDZou-Q-SO-DEPOT": "Z-Q-SO-D",
        "VNDZou-SO-DEPOT-CAND20": "Z-SO-D-C20",
    }
    offsets = {
        "VND0-H3": (6, 6),
        "VND0-Q-CAND20": (-8, 20),
        "VND0-CAND20": (-8, -18),
        "VND0-Adv-CAND20": (-8, 4),
        "VNDZou-SO-DEPOT": (7, 9),
        "VND0": (7, -14),
        "VND0-SO": (7, 13),
        "VND0-SO-DEPOT": (7, -15),
        "VNDZou-CAND20": (7, 5),
        "VNDZou": (7, 12),
        "VNDZou-Q-SO-DEPOT": (-7, 10),
        "VNDZou-SO-DEPOT-CAND20": (-7, 12),
    }
    for _, r in s.iterrows():
        dx, dy = offsets.get(r.Configuration, (6, 5))
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            short_labels.get(r.Configuration, r.Configuration),
            (r["Median CPU"], r["Mean best gap (%)"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.0,
            ha=ha,
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.82),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median CPU time (s, log scale)")
    ax.set_ylabel("Mean best-to-BKS gap (%, log scale)")
    ax.set_title("Barreto ablation: solution quality versus runtime", loc="left", fontsize=15, fontweight="bold")
    ax.grid(color="#DEE2E6", linewidth=0.6)
    ax.text(
        0.99,
        0.98,
        "Teal: non-dominated. Labels: 0=VND0, Z=VNDzou, D=DEPOT, C20=CAND20.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#495057",
    )
    fig.savefig(OUT / "fig_ablation_pareto.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_literature_ranks(ranks):
    d = ranks[ranks.Metric == "run mean"].dropna().sort_values("Average rank", ascending=False)
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    colors = ["#1864AB" if m == "VNDzou" else "#ADB5BD" for m in d.Method]
    ax.barh(d.Method, d["Average rank"], color=colors)
    for y, (_, r) in enumerate(d.iterrows()):
        ax.text(r["Average rank"] + 0.05, y, f"{r['Average rank']:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Average rank (lower is better; available-case ranking)")
    ax.set_title("Cross-literature ranking of average solution quality", loc="left", fontsize=14, fontweight="bold")
    ax.grid(axis="x", color="#DEE2E6", linewidth=0.6)
    fig.savefig(OUT / "fig_literature_ranks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    lit = load_literature()
    vz, vzruns = load_vndzou()
    v0, v0runs = load_vnd0()
    vz.to_csv(OUT / "vndzou_summary_normalized.csv", index=False)
    v0.to_csv(OUT / "vnd0_summary_normalized.csv", index=False)
    lit.to_csv(OUT / "literature_normalized.csv", index=False)

    instance, primary = primary_comparison(v0, vz, lit)
    run_tests, run_counts = run_level_tests(v0runs, vzruns)
    lit_merged, lit_comp, lit_ranks = literature_comparison(vz, lit)
    ab = load_ablation()
    ab_summary, ab_contrasts = ablation_analysis(ab)
    plot_primary(instance)
    plot_ablation(ab_summary)
    plot_literature_ranks(lit_ranks)

    coverage = {
        "VNDzou_instances": vz.groupby("Dataset").Instance.nunique().to_dict(),
        "VND0_summary_instances": v0.groupby("Dataset").Instance.nunique().to_dict(),
        "VND0_raw_instances": v0runs.groupby("Dataset").Instance.nunique().to_dict(),
        "common_primary_instances": instance[instance._merge == "both"].groupby("Dataset").Instance.nunique().to_dict(),
        "run_level_common_instances": run_tests.groupby("Dataset").Instance.nunique().to_dict(),
        "VND0_run_counts": v0.groupby("Dataset").Runs.agg(["min", "median", "max"]).to_dict("index"),
        "VNDzou_run_counts": vz.groupby("Dataset").Runs.agg(["min", "median", "max"]).to_dict("index"),
        "missing_primary": instance[instance._merge != "both"][["Dataset", "Instance", "_merge"]].to_dict("records"),
        "ablation_config_instances": ab.groupby("Configuration").Instance.nunique().to_dict(),
    }
    with open(OUT / "coverage_and_budgets.json", "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False, default=json_default)

    compact = {
        "coverage": coverage,
        "primary": primary.to_dict("records"),
        "run_level_counts": run_counts.to_dict("records"),
        "literature_overall": lit_comp[lit_comp.Dataset == "Overall"].to_dict("records"),
        "ablation_summary": ab_summary.to_dict("records"),
        "ablation_contrasts": ab_contrasts.to_dict("records"),
    }
    with open(OUT / "key_results.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False, default=json_default)

    print(json.dumps(coverage, indent=2, default=json_default))
    print("\nPRIMARY\n", primary.to_string(index=False))
    print("\nRUN-LEVEL COUNTS\n", run_counts.to_string(index=False))
    print("\nLITERATURE OVERALL\n", lit_comp[lit_comp.Dataset == "Overall"].to_string(index=False))
    print("\nABLATION\n", ab_summary.sort_values("Mean best gap (%)").to_string(index=False))
    print("\nCONTRASTS\n", ab_contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
