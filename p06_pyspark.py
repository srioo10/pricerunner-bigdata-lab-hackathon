"""
Step 6 - Run two of the techniques in PySpark and measure how much faster
(or slower) Spark is than the single-machine Python libraries.

Two techniques, both run on the product-listing data (out/clean.csv,
34,459 listings, 10 categories):

  A. Association rules (Apriori family)
       Spark  : pyspark.ml.fpm.FPGrowth            (distributed FP-growth, PFP;
                Spark MLlib has no Apriori, so FPGrowth stands in for it)
       Single : mlxtend apriori + association_rules (classic level-wise Apriori)
                mlxtend fpgrowth + association_rules (single-machine FP-growth)
       Transaction of one listing = set(title tokens) + {"cat=<category>"}
       minSupport = 0.01, minConfidence = 0.8 for every engine, the same
       thresholds as the Apriori section (p03), so the counts can be compared.

  B. Decision Tree classification of `category` from the cleaned title
       Spark  : Tokenizer -> CountVectorizer(binary, vocab 1000) -> StringIndexer
                -> DecisionTreeClassifier(maxDepth 20, entropy)
       Single : sklearn CountVectorizer(binary, 1000 features)
                + DecisionTreeClassifier(entropy, max_depth 20)
       Same 80/20 split for both (one shuffled index, seed 42). Scored with
       accuracy and macro F1 (macro, as in p04).

  (Optional extra) K-Means k=10 on TF-IDF title features, Spark vs sklearn,
  at 1x only, with time and silhouette.

Then a scalability study: the data is replicated 1x, 2x, 5x, 10x, 20x and both
engines are timed (median of 3 runs) on each size. Speedup = t_single / t_spark.
Spark session start-up is timed separately and the JVM is warmed up before any
timed run, so the numbers measure the algorithms, not JVM boot.

Spark runs in local[*] mode (all cores of this machine); mlxtend and sklearn's
tree are single-threaded. The speedups are therefore local multicore Spark vs
single-threaded Python, not a cluster result, and results.md says so.

Outputs (all under out/spark/):
  timings.csv           one row per (technique, engine, size, repeat)
  summary.json          every 1x number the report text needs (see write_report)
  results.md            tables, rules, metrics and the interpretation; the
                        verdicts (faster / slower / break-even) are computed
                        from the median timings, not typed in
  fig_speedup_arm.png   association-rule time vs data size
  fig_speedup_dt.png    decision-tree time vs data size
  run_log.txt           full console log of the run

Usage:  ../.venv/bin/python p06_pyspark.py            (full run)
        ../.venv/bin/python p06_pyspark.py --quick    (sizes 1,2 and 1 repeat; smoke test)
        ../.venv/bin/python p06_pyspark.py --report-only SRC DST
            rebuild results.md and the figures in DST from SRC/summary.json and
            SRC/timings.csv, without starting Spark (seconds instead of ~20 min)
"""

import os
import sys
import time
import json
import statistics
import collections
import resource
import platform

# Spark must bind to loopback (the machine may have no resolvable hostname);
# set before pyspark is imported so the launcher picks it up.
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import warnings
# pyspark 4.3 warns that pandas 3 is "not fully supported"; the conversions used
# here (plain string / array<string> columns) are checked by the agreement tests.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*copy keyword is deprecated.*")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless plotting
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17, "xtick.labelsize": 15,
                     "ytick.labelsize": 15, "legend.fontsize": 14})   # figures are shrunk to page width, so start large
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "out", "clean.csv")
OUT = os.path.join(HERE, "out", "spark")
P03_SWEEP = os.path.join(HERE, "out", "arm", "sweep_global.csv")
P03_RULES = os.path.join(HERE, "out", "arm", "rules_global_s1_c80.csv")

QUICK = "--quick" in sys.argv
REPORT_ONLY = "--report-only" in sys.argv
SIZES = [1, 2] if QUICK else [1, 2, 5, 10, 20]   # replication factors
REPEATS = 1 if QUICK else 3                     # median of this many runs
TIME_LIMIT = 300.0                              # stop growing a series after a run > 5 min
MIN_SUP, MIN_CONF = 0.01, 0.8                   # same thresholds as p03 (Section 4)
SEED = 42
CORES = os.cpu_count()

# Speedups within 5% of 1 are called break-even: with a median of 3 runs the
# timings move by a few percent between repeats, so a smaller gap means nothing.
FASTER, SLOWER = 1.05, 0.95
CLEAR = 1.5                                     # only from here on is a speedup called "clear"
DT_TOL = 0.01                                   # accuracy / macro-F1 gap still counted as "the same"


# ---------------------------------------------------------------------------
# Logging helper: everything printed goes to the console AND to run_log.txt
# (the file is opened only for a real run, so --report-only cannot clobber it)
# ---------------------------------------------------------------------------
_log_f = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if _log_f is not None:
        _log_f.write(s + "\n")
        _log_f.flush()


# ---------------------------------------------------------------------------
# Report generation. This part reads only the summary dict (ctx) and the
# timing rows, so --report-only can rebuild results.md without Spark. Numbers
# and verdicts in the text come from those two inputs, so a new run cannot
# leave stale claims behind.
# ---------------------------------------------------------------------------
SPARK_ENGINE = {"arm": "spark_fpgrowth", "dt": "spark_dt"}
SINGLE_ENGINES = {"arm": ["mlxtend_apriori", "mlxtend_fpgrowth"], "dt": ["sklearn_dt"]}
LABEL = {"spark_fpgrowth": "Spark FPGrowth", "mlxtend_apriori": "mlxtend Apriori",
         "mlxtend_fpgrowth": "mlxtend FP-growth", "spark_dt": "Spark DecisionTree",
         "sklearn_dt": "sklearn DecisionTree"}
SHORT = {"mlxtend_apriori": "Apriori", "mlxtend_fpgrowth": "FP-growth", "sklearn_dt": "sklearn"}
TECH_NAME = {"arm": "association rules", "dt": "decision tree"}
# what one row is in each technique's timing table
ROW_NAME = {"arm": "listings", "dt": "training listings"}
NUM_WORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def medians(tim):
    return (tim.groupby(["technique", "engine", "size"])
            .agg(rows=("rows", "first"), median_s=("seconds", "median"),
                 min_s=("seconds", "min"), max_s=("seconds", "max"), runs=("seconds", "size"),
                 peak_py_mb=("peak_py_mb", "max"), peak_jvm_mb=("peak_jvm_mb", "max"))
            .reset_index())


def speedup_table(med, tech):
    sub = med[med.technique == tech]
    p = sub.pivot(index="size", columns="engine", values="median_s")
    out = pd.DataFrame(index=p.index)
    # row counts of THIS technique (DT uses the training split, ARM all listings)
    out["rows"] = sub.groupby("size")["rows"].first()
    for e in [SPARK_ENGINE[tech]] + SINGLE_ENGINES[tech]:
        if e in p:
            out[e + "_s"] = p[e]
    for e in SINGLE_ENGINES[tech]:
        if e in p and SPARK_ENGINE[tech] in p:
            out["speedup_vs_" + e] = p[e] / p[SPARK_ENGINE[tech]]
    return out


def plot(med, tech, path, title, repeats):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    styles = {"spark_fpgrowth": ("#d9480f", "o"), "spark_dt": ("#d9480f", "o"),
              "mlxtend_apriori": ("#1c7ed6", "s"), "mlxtend_fpgrowth": ("#5c940d", "^"),
              "sklearn_dt": ("#1c7ed6", "s")}
    sub = med[med.technique == tech]
    for e in [SPARK_ENGINE[tech]] + SINGLE_ENGINES[tech]:
        s = sub[sub.engine == e].sort_values("rows")
        if s.empty:
            continue
        c, m = styles[e]
        ax.plot(s["rows"], s["median_s"], marker=m, color=c, label=LABEL[e], lw=1.8)
        # min-max whisker over the repeats
        ax.vlines(s["rows"], s["min_s"], s["max_s"], color=c, alpha=0.5, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # label the x axis at the actual sizes
    xs = sorted(sub["rows"].unique())
    ax.set_xticks(xs, [f"{x/1000:.0f}k" for x in xs])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # y axis: labelled ticks at 1-2-5 steps in plain seconds; minor ticks get no
    # label, otherwise every minor tick of the log axis is printed and they overlap
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    sizes = sorted(sub["size"].unique())
    ax.set_xlabel(f"rows (data replicated {', '.join(f'{z}x' for z in sizes)})")
    ax.set_ylabel(f"median time over {repeats} runs (s, log scale)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def verdict(v):
    return "faster" if v > FASTER else "slower" if v < SLOWER else "break-even"


def fmt(s):
    """Itemset as {a, b, c}, sorted so the same set always prints the same way."""
    return "{" + ", ".join(sorted(s)) + "}"


def size_list(sizes):
    """[1, 2, 5] -> '1x, 2x and 5x'."""
    z = [f"{s}x" for s in sizes]
    return z[0] if len(z) == 1 else ", ".join(z[:-1]) + " and " + z[-1]


def compare(med, tech, single, n_freq_items=None):
    """One paragraph comparing Spark with one single-machine engine, plus the
    facts the rest of the interpretation needs. The wording branches on the
    medians (FASTER / SLOWER / CLEAR), so it stays correct if a rerun moves
    the crossover."""
    st = speedup_table(med, tech)
    spark = SPARK_ENGINE[tech]
    col = "speedup_vs_" + single
    if col not in st or st[col].dropna().empty:
        return f"There are no comparable runs of {LABEL[spark]} and {LABEL[single]}.", None
    sp = st[col].dropna()
    sizes = list(sp.index)
    rows_of = lambda s: f"{int(st.loc[s, 'rows']):,} {ROW_NAME[tech]}"

    def holds_from(ok):
        """First size from which ok(speedup) is true at that and every larger size."""
        for s in sizes:
            if all(ok(sp[t]) for t in sizes if t >= s):
                return s
        return None

    faster_at = [s for s in sizes if sp[s] > FASTER]
    even_at = [s for s in sizes if SLOWER <= sp[s] <= FASTER]
    fast_from = holds_from(lambda v: v > FASTER)
    clear_from = holds_from(lambda v: v >= CLEAR)
    rising = len(sp) > 1 and all(b > a for a, b in zip(sp.values, sp.values[1:]))
    spark_sizes = list(st[spark + "_s"].dropna().index)
    cut_short = sizes[-1] < spark_sizes[-1]       # the single engine lacks the largest sizes

    steps = size_list(sizes)
    vals = [f"{sp[s]:.2f} at {s}x" for s in sizes]
    vals = vals[0] if len(vals) == 1 else ", ".join(vals[:-1]) + " and " + vals[-1]
    out = [f"Against {LABEL[single]}, the {LABEL[spark]} speedup is {vals}."]

    if fast_from is not None and fast_from == sizes[-1]:
        where = (f"the largest size at which {SHORT[single]} was run" if cut_short
                 else "the largest size tested")
        s = f"Spark is faster only at {fast_from}x ({rows_of(fast_from)}), {where}"
        if clear_from is not None:
            s += f", with a speedup of at least {CLEAR:.2f}"
        out.append(s + ".")
    elif fast_from is not None:
        s = f"Spark is faster from {fast_from}x ({rows_of(fast_from)}) up to {sizes[-1]}x"
        if clear_from == fast_from:
            s += f", with a speedup of at least {CLEAR:.2f} at each of these sizes"
        elif clear_from is not None:
            s += f", and the speedup reaches {CLEAR:.2f} or more from {clear_from}x"
        out.append(s + ".")
    elif faster_at:
        out.append(f"Spark is faster at {size_list(faster_at)} but not at every larger size, "
                   "so there is no single crossover size.")
    else:
        if even_at:
            out.append(f"Spark never beats {SHORT[single]} by more than 5% at any size up to {sizes[-1]}x ({rows_of(sizes[-1])}).")
        else:
            out.append(f"Spark is slower at every size up to {sizes[-1]}x ({rows_of(sizes[-1])}).")
        if rising:
            out.append(f"The speedup rises at each step, from {sp.iloc[0]:.2f} to {sp.iloc[-1]:.2f}"
                       + ("." if sp.iloc[-1] >= SLOWER else ", but stays below 1 in this run."))
        if single == "mlxtend_fpgrowth" and n_freq_items is not None:
            out.append(f"Single-machine FP-growth builds one compressed FP-tree over the "
                       f"{n_freq_items} frequent items, which stays cheap at {sizes[-1]}x, while "
                       "Spark FPGrowth pays its per-job overhead plus a shuffle of conditional "
                       "transactions between partitions.")
    if even_at:
        out.append(f"At {size_list(even_at)} the two are within 5% of each other (break-even).")

    a, b = st.loc[sizes[0], single + "_s"], st.loc[sizes[-1], single + "_s"]
    c, d = st.loc[sizes[0], spark + "_s"], st.loc[sizes[-1], spark + "_s"]
    out.append(f"Between {sizes[0]}x and {sizes[-1]}x the {SHORT[single]} time goes from "
               f"{a:.2f} s to {b:.2f} s ({b / a:.2f} times as long) and the Spark time from "
               f"{c:.2f} s to {d:.2f} s ({d / c:.2f} times).")
    facts = dict(tech=tech, single=single, first=sp.iloc[0], sizes=sizes,
                 g_single=b / a, g_spark=d / c)
    return " ".join(out), facts


def write_report(ctx, tim, outdir):
    """Build results.md and the two figures in outdir from ctx + timing rows."""
    os.makedirs(outdir, exist_ok=True)
    med = medians(tim)
    A, D, K = ctx["arm"], ctx["dt"], ctx.get("kmeans")
    cores, ms, mc = ctx["cores"], ctx["min_sup"], ctx["min_conf"]
    sizes, repeats = ctx["sizes"], ctx["repeats"]
    md = []

    md.append("# PySpark test drive: association rules and decision trees\n")
    md.append(f"Machine: {cores} logical cores (`os.cpu_count()` = {cores}), "
              f"{ctx['ram_total_gb']:.1f} GB RAM ({ctx['ram_avail_gb']:.1f} GB free at start). "
              f"Spark {ctx['spark_version']} in `local[*]` mode, driver memory {ctx['driver_mem']}, "
              f"`spark.sql.shuffle.partitions` = {ctx['npart']}. Python {ctx['python']}, "
              f"pandas {ctx['pandas']}.\n")

    # ---- setup: the caveats that apply to every number, stated once ----------
    md.append("\n## Setup\n")
    md.append(f"Spark runs in `local[*]` mode on all {cores} cores of this one machine; "
              "mlxtend Apriori, mlxtend FP-growth and sklearn's DecisionTreeClassifier are "
              "single-threaded. Every speedup in this report is therefore local multicore Spark "
              "against single-threaded Python on the same machine, with no network shuffle. It "
              "is not a cluster result. Speedup = t_single / t_spark, so a value above 1 means "
              f"Spark is faster. Above {FASTER:.2f} is called faster, below {SLOWER:.2f} slower, "
              "and anything in between break-even, because repeat timings move by a few percent.\n")
    md.append(f"The sizes {size_list([z for z in sizes if z > 1])} are the original listings "
              "replicated k times. Replication multiplies every count by k, so supports, confidences, "
              "itemsets and learned trees stay the same and only the amount of work grows. The "
              "study therefore measures throughput, not behaviour on new data.\n")
    md.append(f"Spark session start-up took {ctx['startup_s']:.2f} s and the JVM warm-up runs "
              f"{ctx['warmup_s']:.2f} s. Both are excluded from every timing below.\n")

    # ---- A. association rules ------------------------------------------------
    md.append(f"\n## A. Association rules (minSupport {ms:g}, minConfidence {mc:g})\n")
    md.append("Spark MLlib has no Apriori implementation, so its FPGrowth stands in as the "
              "Spark association-rule miner. Each listing is one basket: its distinct title "
              f"tokens plus `cat=<category>`. The thresholds are the ones used in Section 4 "
              f"(support {ms:.0%}, confidence {mc:.0%}).\n")
    n_rows_max = ctx["n_listings"] * max(sizes)
    full_gb = n_rows_max * A["n_items_all"] / 2**30          # 1 byte per bool cell
    md.append(f"Both mlxtend baselines start from the raw baskets and first make Apriori's own "
              f"first pass: count every item and keep the {A['n_freq_items']} items with support "
              f">= {ms:.0%} out of {A['n_items_all']:,} distinct items. Only those columns are "
              "one-hot encoded. This cannot change the result, because no frequent itemset "
              "contains an infrequent item, but it keeps memory down: a one-hot matrix of all "
              f"items at {max(sizes)}x ({n_rows_max:,} rows) would take {full_gb:.1f} GB before "
              "Apriori builds a single candidate. Apriori runs with `low_memory=False`. Section 4 "
              "(p03) runs Apriori on the full one-hot matrix of all items, so its run times are "
              "not comparable with the ones here.\n")
    md.append("| engine | frequent itemsets | rules (all consequent sizes) | rules with 1-item consequent | time at 1x (s, single run) |")
    md.append("|---|---|---|---|---|")
    md.append(f"| Spark FPGrowth | {A['n_fi_spark']} | {A['n_rules_spark']} | {A['n_rules_spark']} | {A['t_spark_1x']:.2f} |")
    md.append(f"| mlxtend Apriori | {A['n_fi_apr']} | {A['n_rules_apr']} | {A['n_rules1_apr']} | {A['t_apr_1x']:.2f} |")
    md.append(f"| mlxtend FP-growth | {A['n_fi_fpg']} | {A['n_rules_fpg']} | {A['n_rules1_fpg']} | {A['t_fpg_1x']:.2f} |\n")

    if A["same_fi"]:
        fi_txt = ("The three engines find the same frequent itemsets with the same counts "
                  f"({A['n_fi_spark']} itemsets).")
    else:
        fi_txt = (f"The frequent itemsets do not match: {A['fi_mismatch_apr']} itemsets differ "
                  f"between Spark and Apriori and {A['fi_mismatch_fpg']} between Spark and "
                  "FP-growth (missing in one engine, or a different count).")
    cons_txt = (f"Spark's `associationRules` only generates rules with a 1-item consequent, while "
                "mlxtend generates every consequent size, so Spark reports fewer rules in total "
                f"({A['n_rules_spark']} against {A['n_rules_apr']}).")
    if A["same_rules"]:
        r_txt = (f"Restricted to 1-item consequents, all three engines produce the same "
                 f"{A['n_rules_spark']} rules")
        if A.get("max_lift_diff") is not None:
            r_txt += f", and the largest lift difference on a shared rule is {A['max_lift_diff']:.1e}"
        r_txt += "."
    else:
        r_txt = (f"Restricted to 1-item consequents the rule sets do not match (Spark "
                 f"{A['n_rules_spark']}, Apriori {A['n_rules1_apr']}, FP-growth "
                 f"{A['n_rules1_fpg']}; the Spark-vs-Apriori differences are listed below).")
    md.append(f"{fi_txt} {cons_txt} {r_txt}\n")
    if A["rule_diff"]:
        on_edge = all(d["n_a"] and abs(d["n_ac"] / d["n_a"] - mc) < 1e-9 for d in A["rule_diff"])
        md.append("Rules kept by one engine only, with the confidence recomputed from Spark's "
                  "exact counts:" + (
                      f" each sits exactly on the {mc:g} threshold, so floating-point rounding "
                      "decides (Spark divides two counts, mlxtend divides two supports that were "
                      "already divided by N)." if on_edge else "") + "\n")
        for d in A["rule_diff"]:
            conf = d["n_ac"] / d["n_a"] if d["n_ac"] and d["n_a"] else float("nan")
            md.append(f"- {d['who']}: {fmt(d['ante'])} -> {fmt(d['cons'])}, confidence = "
                      f"{d['n_ac']}/{d['n_a']} = {conf:.17g}")
        md.append("")

    p3 = ctx.get("p03")
    if p3:
        mine = (A["n_fi_apr"], A["n_rules_apr"], A["n_cat_rules_apr"])
        theirs = (p3["n_itemsets"], p3["n_rules"], p3["n_cat_rules"])
        md.append(f"Comparison with Section 4 (p03, same thresholds): p03 reports {theirs[0]} "
                  f"frequent itemsets, {theirs[1]} rules and {theirs[2]} single-consequent rules "
                  f"with a `cat=` consequent; mlxtend Apriori here gives {mine[0]}, {mine[1]} and "
                  f"{mine[2]}" + (", the same counts.\n" if mine == theirs
                                  else ". The counts differ, so the two runs do not mine the same rules.\n"))

    # Many top rules are reorderings of one phrase ({hewlett, packard, enterprise}
    # with or without cat=CPUs), so keep one rule per itemset and drop rules that
    # contain a cat= item; those are counted separately below.
    pool = A.get("rules_by_lift") or A["top_rules"]
    seen, shown = set(), []
    for r in pool:
        items = frozenset(r["ante"]) | frozenset(r["cons"])
        if any(i.startswith("cat=") for i in items) or items in seen:
            continue
        seen.add(items)
        shown.append(r)
        if len(shown) == 15:
            break
    src = (f"all {A['n_rules_spark']} Spark rules" if len(pool) >= A["n_rules_spark"]
           else f"the {len(pool)} highest-lift Spark rules (the only ones this run saved)")
    md.append(f"{A['n_cat_rules_spark']} of the {A['n_rules_spark']} Spark rules have a `cat=` "
              "consequent, i.e. they say which title words imply a product category. The table "
              f"below starts from {src}, sorted by lift with ties broken by confidence, then "
              "support. It drops every rule that contains a `cat=` item and keeps one rule per "
              "distinct itemset (antecedent plus consequent), because many high-lift rules are "
              "reorderings of the same few words.\n")
    md.append("| # | antecedent | consequent | support | confidence | lift |")
    md.append("|---|---|---|---|---|---|")
    for i, r in enumerate(shown, 1):
        md.append(f"| {i} | {fmt(r['ante'])} | {fmt(r['cons'])} | "
                  f"{r['support']:.4f} | {r['confidence']:.3f} | {r['lift']:.2f} |")
    md.append("")

    # ---- B. decision tree ----------------------------------------------------
    md.append("\n## B. Decision tree: category from title\n")
    md.append(f"Same split for both: {D['n_train']} training and {D['n_test']} test listings "
              f"(one shuffled index, seed {ctx['seed']}). Features: binary bag of words over the "
              f"1000 most frequent title terms ({D['vocab_overlap']} of the 1000 terms are the "
              "same in both vocabularies). Tree: entropy impurity, max depth 20. Accuracy and "
              "macro F1 (the metric of Section 5) are computed with the same sklearn functions "
              "on the same test rows for both engines.\n")
    md.append("| engine | accuracy | macro F1 | train time at 1x (s, single run) | depth | nodes |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| Spark DecisionTreeClassifier | {D['acc_spark']:.4f} | {D['f1_spark']:.4f} | "
              f"{D['t_spark_1x']:.2f} | {D['depth_spark']} | {D['nodes_spark']} |")
    md.append(f"| sklearn DecisionTreeClassifier | {D['acc_sk']:.4f} | {D['f1_sk']:.4f} | "
              f"{D['t_sk_1x']:.2f} | {D['depth_sk']} | {D['nodes_sk']} |\n")
    d_acc = abs(D["acc_spark"] - D["acc_sk"]) * 100
    d_f1 = abs(D["f1_spark"] - D["f1_sk"]) * 100
    dt_same = d_acc <= DT_TOL * 100 and d_f1 <= DT_TOL * 100
    md.append(f"The two trees differ by {d_acc:.2f} points of accuracy and {d_f1:.2f} points of "
              f"macro F1, " + ("within" if dt_same else "outside") +
              f" the {DT_TOL * 100:.0f}-point tolerance used here for \"the same\". "
              "Spark's tree searches splits differently from sklearn's (level-wise, on binned "
              "features), so the two trees are not node-for-node identical. Train time includes "
              "building the vocabulary and the feature matrix; sklearn's tree is fit on a dense "
              "float32 matrix, and the densify step is inside its timing.\n")

    # ---- extra: k-means ------------------------------------------------------
    if K and K.get("ok"):
        md.append("\n## Extra: K-Means k=10 on TF-IDF title features (1x, single run)\n")
        md.append("| engine | time (s) | silhouette |")
        md.append("|---|---|---|")
        md.append(f"| Spark KMeans (k-means-parallel init, 20 iter) | {K['t_spark']:.2f} | "
                  f"{K['sil_spark']:.3f} (squared Euclidean, all rows) |")
        md.append(f"| sklearn KMeans (k-means++ init, 1 init, 20 iter, raw TF-IDF; not the "
                  f"10-init LSA model of Section 6) | {K['t_sk']:.2f} | {K['sil_sk']:.3f} "
                  "(Euclidean, 5000-row sample) |\n")
        md.append("The two silhouettes use different distance definitions, so compare each "
                  "against 0 (no structure) rather than against each other. Not part of the "
                  "scaling study.\n")
    elif K:
        md.append(f"\n## Extra: K-Means\n\nFailed: `{K.get('error')}`\n")

    # ---- scalability ---------------------------------------------------------
    md.append(f"\n## Scalability: data replicated {min(sizes)}x to {max(sizes)}x\n")
    md.append(f"Median of {repeats} runs per point; each speedup is followed by its range over those runs "
              "(worst to best pairing). Before the "
              "clock starts, the replicated data are built and cached in Spark; the "
              "single-machine side starts from an in-memory Python list.\n")

    def header(c):
        if c == "rows":
            return "rows"
        if c.startswith("speedup_vs_"):
            return "speedup vs " + SHORT[c[len("speedup_vs_"):]]
        return LABEL[c[:-2]] + " (s)"

    for tech, name in (("arm", "Association rules"), ("dt", "Decision tree (train)")):
        st = speedup_table(med, tech)
        md.append(f"\n### {name}\n")
        cols = list(st.columns)
        md.append("| size | " + " | ".join(header(c) for c in cols) + " |")
        md.append("|---" * (len(cols) + 1) + "|")
        for size, r in st.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if pd.isna(v):
                    cells.append("-")
                elif c == "rows":
                    cells.append(f"{int(v):,}")
                elif c.startswith("speedup_vs_"):
                    # range over the repeats: slowest single run / fastest Spark run .. fastest single / slowest Spark
                    t = tim[(tim.technique == tech) & (tim["size"] == size)].groupby("engine").seconds
                    lo, hi = t.min(), t.max()
                    sg, sp_ = c[len("speedup_vs_"):], SPARK_ENGINE[tech]
                    cells.append(f"{v:.2f} ({lo[sg] / hi[sp_]:.2f}-{hi[sg] / lo[sp_]:.2f})")
                else:
                    cells.append(f"{v:.2f}")
            md.append(f"| {size}x | " + " | ".join(cells) + " |")
    for key, size in ctx["stopped"].items():
        md.append(f"\n{LABEL[key.split('/')[1]]} was not run beyond {size}x because one run "
                  f"there took longer than {ctx['time_limit']:.0f} s.")
    for key, zs in ctx["mem_skipped"].items():
        md.append(f"\n{LABEL[key.split('/')[1]]} was not run at {size_list(zs)} because its "
                  "projected peak memory was above 60% of the free RAM.")

    # peak memory: one row per engine, at the largest size it ran
    big = med.sort_values("size").groupby(["technique", "engine"]).tail(1) \
        .sort_values(["technique", "engine"])
    md.append("\n### Peak memory at the largest size each engine ran\n")
    md.append("Peak resident set (VmHWM, reset before each run, max over repeats) of the Python "
              "process and of the Spark JVM, which in local mode is both driver and executor. "
              "The JVM column is only meaningful for Spark rows; for mlxtend and sklearn rows it "
              "is the idle Spark JVM that stays alive alongside.\n")
    md.append("| technique | engine | size | rows | peak Python MB | peak JVM MB |")
    md.append("|---|---|---|---|---|---|")
    for r in big.itertuples():
        md.append(f"| {TECH_NAME[r.technique]} | {LABEL[r.engine]} | {r.size}x | {int(r.rows):,} | "
                  f"{r.peak_py_mb:.0f} | {r.peak_jvm_mb:.0f} |")
    sgl = big[~big.engine.isin(SPARK_ENGINE.values())]
    spk = big[big.engine.isin(SPARK_ENGINE.values())]
    if len(sgl) and len(spk):
        w = sgl.loc[sgl.peak_py_mb.idxmax()]
        md.append(f"\nThe largest single-machine peak is {LABEL[w.engine]} at {w['size']}x with "
                  f"{w.peak_py_mb / 1024:.1f} GB in Python; the Spark runs peak at "
                  f"{spk.peak_py_mb.max():.0f} MB in Python and {spk.peak_jvm_mb.max() / 1024:.1f} GB "
                  f"in the JVM (heap limit {ctx['driver_mem']}).\n")

    md.append("\nFigures: `fig_speedup_arm.png`, `fig_speedup_dt.png` (log-log, median with "
              "min-max whiskers).\n")

    # ---- interpretation ------------------------------------------------------
    md.append("\n## Interpretation\n")
    paras, facts = [], []
    for tech in ("arm", "dt"):
        for single in SINGLE_ENGINES[tech]:
            text, f = compare(med, tech, single, A["n_freq_items"])
            paras.append(text)
            if f:
                facts.append(f)

    # the speedups only mean something if both sides produced the same output
    if A["same_fi"] and A["same_rules"] and dt_same:
        md.append("Spark and the single-machine libraries produce the same output here (same "
                  "frequent itemsets and 1-item-consequent rules; trees within "
                  f"{DT_TOL * 100:.0f} point in accuracy and macro F1), so the speedups compare "
                  "like with like.\n")
    else:
        bad = []
        if not A["same_fi"]:
            bad.append(f"frequent itemsets differ ({A['fi_mismatch_apr']} vs Apriori, "
                       f"{A['fi_mismatch_fpg']} vs FP-growth)")
        if not A["same_rules"]:
            bad.append(f"1-item-consequent rules differ (Spark {A['n_rules_spark']}, Apriori "
                       f"{A['n_rules1_apr']}, FP-growth {A['n_rules1_fpg']})")
        if not dt_same:
            bad.append(f"the trees differ by {d_acc:.2f} accuracy points and {d_f1:.2f} macro-F1 "
                       "points")
        md.append("The output agreement check failed: " + "; ".join(bad) + ". The speedups "
                  "below compare runs whose outputs are not the same.\n")

    # small data: the fixed per-job cost, without repeating the 1x numbers
    first = [f for f in facts if f["sizes"][0] == min(sizes)]
    if first and all(f["first"] < SLOWER for f in first):
        single_1x = med[(med["size"] == min(sizes)) & ~med.engine.isin(SPARK_ENGINE.values())]
        n_below = int((single_1x.median_s < ctx["startup_s"]).sum())
        n_all = len(single_1x)
        if n_below == n_all:
            vs = f"longer than every single-machine run at {min(sizes)}x"
        elif n_below == 0:
            vs = f"shorter than every single-machine run at {min(sizes)}x"
        else:
            vs = (f"longer than {NUM_WORD.get(n_below, n_below)} of the "
                  f"{NUM_WORD.get(n_all, n_all)} single-machine runs at {min(sizes)}x")
        md.append(f"At {min(sizes)}x Spark is slower in all {NUM_WORD.get(len(first), len(first))} "
                  "comparisons. Each Spark job has a fixed cost: building the query plan, "
                  f"scheduling tasks on {ctx['npart']} partitions, moving rows between JVM "
                  "threads and, for the tree, one pass over the data per tree level. mlxtend and "
                  "sklearn work on one in-memory array and skip all of that. A real script also "
                  f"pays the {ctx['startup_s']:.2f} s session start-up, which on its own is {vs}.\n")
    elif first:
        md.append(f"At {min(sizes)}x Spark is " +
                  ", ".join(f"{verdict(f['first'])} than {LABEL[f['single']]}" for f in first) +
                  ".\n")

    for p in paras:
        md.append(p + "\n")

    slow_growth = [f for f in facts if f["g_spark"] < f["g_single"]]
    if slow_growth and len(slow_growth) < len(facts):
        md.append("The Spark time grows more slowly with data size than the single-machine time "
                  "only against " + ", ".join(LABEL[f["single"]] for f in slow_growth) + ".\n")

    with open(os.path.join(outdir, "results.md"), "w") as fh:
        fh.write("\n".join(md))
    plot(med, "arm", os.path.join(outdir, "fig_speedup_arm.png"),
         f"Association rules: Spark local[{cores}] vs single machine", repeats)
    plot(med, "dt", os.path.join(outdir, "fig_speedup_dt.png"),
         f"Decision tree: Spark local[{cores}] vs single machine", repeats)
    return med


if REPORT_ONLY:
    i = sys.argv.index("--report-only")
    src, dst = sys.argv[i + 1], sys.argv[i + 2]
    with open(os.path.join(src, "summary.json")) as fh:
        ctx = json.load(fh)
    write_report(ctx, pd.read_csv(os.path.join(src, "timings.csv")), dst)
    print("wrote", dst)
    sys.exit(0)


# single-machine libraries (imported here so --report-only stays light)
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.feature_extraction.text import CountVectorizer as SkCountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import accuracy_score, f1_score, silhouette_score

os.makedirs(OUT, exist_ok=True)
_log_f = open(os.path.join(OUT, "run_log.txt"), "w")


# ---------------------------------------------------------------------------
# Memory helpers.
# Peak resident memory (VmHWM) is read from /proc for the Python driver and
# for the Spark JVM (a separate process in local mode: it is both the Spark
# driver and the executor). Writing "5" to /proc/<pid>/clear_refs resets the
# peak counter, so each timed run gets its own peak. Falls back to
# resource.getrusage (lifetime peak of this Python process) if /proc fails.
# ---------------------------------------------------------------------------
def free_ram_gb():
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return info["MemTotal"] / 2**20, info["MemAvailable"] / 2**20


def reset_peak(pid):
    try:
        with open(f"/proc/{pid}/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def peak_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    if pid == os.getpid():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return float("nan")


# ---------------------------------------------------------------------------
# 0. Check the machine before sizing anything
# ---------------------------------------------------------------------------
tot, avail = free_ram_gb()
log(f"# p06_pyspark  python {platform.python_version()}  cores={CORES}")
log(f"RAM total {tot:.1f} GB, available {avail:.1f} GB")
# The biggest single-machine object is the dense 20x DT matrix:
# 0.8*34407*20 rows x 1000 cols float32 ~ 2.2 GB, plus Spark's 6 GB driver heap.
if avail < 12 and not QUICK:
    SIZES = [s for s in SIZES if s <= 10]
    log("Less than 12 GB free -> largest replication dropped to 10x")
log("sizes:", SIZES, " repeats:", REPEATS)

# ---------------------------------------------------------------------------
# 1. Load data (pandas) and build the shared pieces
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA)
df["tokens"] = df["tokens"].fillna("").astype(str)
df["title_clean"] = df["title_clean"].fillna("").astype(str)
N = len(df)
log(f"rows {N}, categories {df['category'].nunique()}")

# Transactions for technique A: the distinct title tokens plus the category
# as an extra item, so rules like {bosch, integrated} -> {cat=Dishwashers} can
# appear. FPGrowth requires unique items per basket, hence set().
transactions = [sorted(set(t.split()) | {"cat=" + c})
                for t, c in zip(df["tokens"], df["category"])]
item_count = collections.Counter(i for b in transactions for i in b)
N_ITEMS_ALL = len(item_count)                    # distinct items before the support filter

# One shuffled index gives the SAME 80/20 split to Spark and sklearn
# (Spark's randomSplit and sklearn's train_test_split would not agree even
# with the same seed).
rng = np.random.default_rng(SEED)
perm = rng.permutation(N)
n_train = int(0.8 * N)
is_train = np.zeros(N, dtype=bool)
is_train[perm[:n_train]] = True
train_pd = df.loc[is_train, ["title_clean", "category"]].reset_index(drop=True)
test_pd = df.loc[~is_train, ["title_clean", "category"]].reset_index(drop=True)
log(f"train {len(train_pd)}, test {len(test_pd)}")

# ---------------------------------------------------------------------------
# 2. Start Spark, timing the start-up separately, then warm the JVM up
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.feature import (Tokenizer, CountVectorizer, StringIndexer,
                                IDF, Normalizer)
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

DRIVER_MEM = "6g"
spark = (SparkSession.builder
         .master("local[*]")                                   # all cores
         .appName("p06_pyspark")
         .config("spark.driver.memory", DRIVER_MEM)           # local mode: driver = executor
         .config("spark.sql.shuffle.partitions", str(2 * CORES))
         .config("spark.default.parallelism", str(2 * CORES))
         .config("spark.ui.enabled", "false")                  # no web UI overhead
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
t_startup = time.perf_counter() - t0
JVM_PID = spark.sparkContext._jvm.java.lang.ProcessHandle.current().pid()
PY_PID = os.getpid()
log(f"Spark {spark.version} started in {t_startup:.2f} s "
    f"(defaultParallelism={spark.sparkContext.defaultParallelism}, JVM pid {JVM_PID})")

NPART = 2 * CORES   # partitions used for every cached input DataFrame

# Spark DataFrames for the 1x data (built once; replicated copies are derived
# inside Spark with a cross join so the driver never ships 20x data).
tx_sdf = spark.createDataFrame(pd.DataFrame({"items": transactions}),
                               schema="items array<string>")
train_sdf = spark.createDataFrame(train_pd)
test_sdf = spark.createDataFrame(test_pd).repartition(NPART).cache()
test_sdf.count()


def make_fpgrowth():
    return FPGrowth(itemsCol="items", minSupport=MIN_SUP, minConfidence=MIN_CONF)


def make_dt_pipeline():
    """Spark ML pipeline for technique B. Each stage mirrors the sklearn side."""
    return Pipeline(stages=[
        # split title on whitespace (title_clean is already lower-case, punctuation-free)
        Tokenizer(inputCol="title_clean", outputCol="words"),
        # binary bag of words over the 1000 most frequent terms
        CountVectorizer(inputCol="words", outputCol="features",
                        vocabSize=1000, binary=True),
        # category string -> numeric label (handleInvalid keeps unseen test labels)
        StringIndexer(inputCol="category", outputCol="label", handleInvalid="keep"),
        # maxBins 32 is irrelevant for 0/1 features; maxMemoryInMB raised from 256
        # so each level of the tree is grown in fewer passes over the data;
        # cacheNodeIds avoids re-sending the tree to executors at every level.
        DecisionTreeClassifier(featuresCol="features", labelCol="label",
                               maxDepth=20, impurity="entropy", seed=SEED,
                               maxMemoryInMB=1024, cacheNodeIds=True),
    ])


# Warm-up: one tiny run of every Spark algorithm used later, so JIT
# compilation, class loading and Python worker start-up are paid here and not
# inside the first timed run.
t0 = time.perf_counter()
make_fpgrowth().fit(tx_sdf.limit(2000)).associationRules.count()
make_dt_pipeline().fit(train_sdf.limit(2000)).transform(test_sdf.limit(500)).count()
KMeans(k=3, seed=SEED, featuresCol="features").fit(
    CountVectorizer(inputCol="words", outputCol="features", vocabSize=100)
    .fit(warm_words := Tokenizer(inputCol="title_clean", outputCol="words").transform(train_sdf.limit(2000)))
    .transform(warm_words)
).summary.trainingCost
t_warm = time.perf_counter() - t0
log(f"JVM warm-up runs took {t_warm:.2f} s (excluded from timings)")


def replicate_sdf(sdf, k):
    """k copies of a Spark DataFrame, spread over NPART partitions and cached.
    Built inside Spark (cross join with range(k)); materialised before timing."""
    out = sdf.crossJoin(spark.range(k).withColumnRenamed("id", "_rep")).drop("_rep") \
        if k > 1 else sdf
    out = out.repartition(NPART).cache()
    out.count()
    return out


# ---------------------------------------------------------------------------
# 3. The timed units of work. Each returns its result so the 1x run can also
#    be used for the accuracy / agreement checks.
# ---------------------------------------------------------------------------

# ---- Technique A, single machine ------------------------------------------
def single_arm(tx, algo):
    """mlxtend pipeline on a list of baskets. Timed end-to-end from raw
    baskets, because Spark FPGrowth also starts from raw baskets.
    Step 1 is Apriori's own first pass (L1): count items and keep only those
    with support >= MIN_SUP. Encoding only the ~130 frequent items instead of
    all ~7,000 tokens keeps the one-hot matrix small (at 20x the full matrix
    would be several GB of booleans); it cannot change the result, since no
    frequent itemset contains an infrequent item. results.md states this,
    because p03 times apriori on the full matrix."""
    n = len(tx)
    cnt = collections.Counter(i for b in tx for i in b)
    keep = sorted(i for i, c in cnt.items() if c >= MIN_SUP * n)
    col = {i: j for j, i in enumerate(keep)}
    X = np.zeros((n, len(keep)), dtype=bool)
    for r, b in enumerate(tx):
        for i in b:
            j = col.get(i)
            if j is not None:
                X[r, j] = True
    X = pd.DataFrame(X, columns=keep)
    # low_memory=False is mlxtend's default (fast, memory-hungry); spelled out
    # because the memory guard and results.md rely on it
    kw = dict(low_memory=False) if algo is apriori else {}
    fi = algo(X, min_support=MIN_SUP, use_colnames=True, **kw)
    rules = association_rules(fi, metric="confidence", min_threshold=MIN_CONF)
    return fi, rules


# ---- Technique A, Spark ---------------------------------------------------
def spark_arm(sdf):
    """Distributed FP-growth. fit() mines the frequent itemsets; the rules
    are lazy, so count() forces them. Both counts are part of the timing."""
    model = make_fpgrowth().fit(sdf)
    n_fi = model.freqItemsets.count()
    n_rules = model.associationRules.count()
    return model, n_fi, n_rules


# ---- Technique B, single machine ------------------------------------------
def single_dt(texts, labels):
    """sklearn: binary bag of words (1000 terms) -> entropy tree, depth 20.
    token_pattern \\S+ splits on whitespace exactly like Spark's Tokenizer.
    The matrix is densified (float32) before fitting because sklearn's tree
    splitter is slower on sparse CSR input than on dense input, so dense is the
    stronger single-machine baseline. The densify step is inside the timing."""
    vec = SkCountVectorizer(binary=True, max_features=1000, token_pattern=r"\S+")
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    clf = SkDecisionTree(criterion="entropy", max_depth=20, random_state=SEED)
    clf.fit(X, labels)
    return vec, clf


# ---- Technique B, Spark ---------------------------------------------------
def spark_dt(sdf):
    """Spark ML pipeline fit (vocabulary, label index and tree all learned
    from the training DataFrame)."""
    return make_dt_pipeline().fit(sdf)


def timed(fn, *args):
    """Run fn, return (result, seconds, peak python MB, peak JVM MB)."""
    reset_peak(PY_PID)
    reset_peak(JVM_PID)
    t0 = time.perf_counter()
    res = fn(*args)
    dt = time.perf_counter() - t0
    return res, dt, peak_mb(PY_PID), peak_mb(JVM_PID)


rows = []          # timing rows -> timings.csv
# ctx collects every number the report text needs; it is saved as summary.json
# so results.md can be rebuilt with --report-only without rerunning Spark.
ctx = dict(cores=CORES, ram_total_gb=tot, ram_avail_gb=avail, spark_version=spark.version,
           python=platform.python_version(), pandas=pd.__version__, driver_mem=DRIVER_MEM,
           npart=NPART, startup_s=t_startup, warmup_s=t_warm, min_sup=MIN_SUP,
           min_conf=MIN_CONF, sizes=SIZES, repeats=REPEATS, time_limit=TIME_LIMIT,
           seed=SEED, n_listings=N)

# ---------------------------------------------------------------------------
# 4. Technique A at 1x: results and agreement check
# ---------------------------------------------------------------------------
log("\n=== A. Association rules at 1x ===")
tx1 = replicate_sdf(tx_sdf, 1)
(model1, n_fi_s, n_r_s), t, _, _ = timed(spark_arm, tx1)
log(f"Spark FPGrowth: {n_fi_s} frequent itemsets, {n_r_s} rules, {t:.2f} s")
(fi_a, r_a), t_a, _, _ = timed(single_arm, transactions, apriori)
log(f"mlxtend apriori: {len(fi_a)} itemsets, {len(r_a)} rules, {t_a:.2f} s")
(fi_f, r_f), t_f, _, _ = timed(single_arm, transactions, fpgrowth)
log(f"mlxtend fpgrowth: {len(fi_f)} itemsets, {len(r_f)} rules, {t_f:.2f} s")

# Itemsets as {frozenset: absolute count} for an exact comparison.
spark_fi = {frozenset(r["items"]): r["freq"] for r in model1.freqItemsets.collect()}
apr_fi = {frozenset(s): int(round(v * N)) for s, v in zip(fi_a["itemsets"], fi_a["support"])}
fpg_fi = {frozenset(s): int(round(v * N)) for s, v in zip(fi_f["itemsets"], fi_f["support"])}
same_fi = spark_fi == apr_fi == fpg_fi
mismatch = lambda a, b: sum(1 for k in set(a) | set(b) if a.get(k) != b.get(k))

# Spark only generates rules with ONE item in the consequent; mlxtend
# generates every consequent size. Compare on single-consequent rules.
spark_rules_pd = model1.associationRules.toPandas()
spark_rules = {(frozenset(a), frozenset(c)) for a, c in
               zip(spark_rules_pd["antecedent"], spark_rules_pd["consequent"])}
apr_rules1 = r_a[r_a["consequents"].apply(len) == 1]
apr_rules = {(frozenset(a), frozenset(c)) for a, c in
             zip(apr_rules1["antecedents"], apr_rules1["consequents"])}
fpg_rules1 = r_f[r_f["consequents"].apply(len) == 1]
fpg_rules = {(frozenset(a), frozenset(c)) for a, c in
             zip(fpg_rules1["antecedents"], fpg_rules1["consequents"])}
same_rules = spark_rules == apr_rules == fpg_rules
log(f"itemsets identical across 3 engines: {same_fi}; "
    f"single-consequent rules identical: {same_rules} "
    f"(Spark {len(spark_rules)}, apriori {len(apr_rules)}, fpgrowth {len(fpg_rules)})")

# Lift check: same rule, same lift in Spark and mlxtend.
lift_s = {(frozenset(a), frozenset(c)): l for a, c, l in
          zip(spark_rules_pd["antecedent"], spark_rules_pd["consequent"], spark_rules_pd["lift"])}
lift_m = {(frozenset(a), frozenset(c)): l for a, c, l in
          zip(apr_rules1["antecedents"], apr_rules1["consequents"], apr_rules1["lift"])}
shared = [k for k in lift_s if k in lift_m]
max_lift_diff = max(abs(lift_s[k] - lift_m[k]) for k in shared) if shared else None
log(f"max |lift_spark - lift_mlxtend| over shared rules: {max_lift_diff}")
# Any rule that one engine keeps and the other drops, with its exact counts.
rule_diff = []
for k in spark_rules ^ apr_rules:
    d = dict(who="Spark only" if k in spark_rules else "mlxtend only",
             ante=sorted(k[0]), cons=sorted(k[1]),
             n_ac=spark_fi.get(k[0] | k[1]), n_a=spark_fi.get(k[0]))
    rule_diff.append(d)
    log("  rule difference ->", d)

by_lift = spark_rules_pd.sort_values(["lift", "confidence", "support"], ascending=False)
top = by_lift.head(15)
log("\nTop 15 rules by lift:")
for i, r in enumerate(top.itertuples(), 1):
    log(f"{i:2d}. {fmt(r.antecedent) + ' -> ' + fmt(r.consequent):60s} "
        f"sup={r.support:.4f} conf={r.confidence:.3f} lift={r.lift:.2f}")

is_cat1 = lambda c: len(c) == 1 and next(iter(c)).startswith("cat=")
ctx["arm"] = dict(
    n_items_all=N_ITEMS_ALL,
    n_freq_items=sum(1 for k in spark_fi if len(k) == 1),     # Apriori's L1
    n_fi_spark=n_fi_s, n_rules_spark=n_r_s, t_spark_1x=t,
    n_fi_apr=len(fi_a), n_rules_apr=len(r_a), n_rules1_apr=len(apr_rules), t_apr_1x=t_a,
    n_fi_fpg=len(fi_f), n_rules_fpg=len(r_f), n_rules1_fpg=len(fpg_rules), t_fpg_1x=t_f,
    same_fi=bool(same_fi), same_rules=bool(same_rules),
    fi_mismatch_apr=mismatch(spark_fi, apr_fi), fi_mismatch_fpg=mismatch(spark_fi, fpg_fi),
    max_lift_diff=max_lift_diff, rule_diff=rule_diff,
    n_cat_rules_spark=sum(1 for c in spark_rules_pd["consequent"] if is_cat1(c)),
    n_cat_rules_apr=int(apr_rules1["consequents"].apply(is_cat1).sum()),
    # every rule, sorted by lift: the report dedupes them by itemset, and the top
    # 15 alone are mostly reorderings of one phrase
    rules_by_lift=[dict(ante=sorted(r.antecedent), cons=sorted(r.consequent),
                        support=float(r.support), confidence=float(r.confidence),
                        lift=float(r.lift))
                   for r in by_lift.itertuples()])
log(f"category rules (single cat= consequent): Spark {ctx['arm']['n_cat_rules_spark']}, "
    f"apriori {ctx['arm']['n_cat_rules_apr']}")

# p03's counts at the same thresholds, read from its own output files, so the
# comparison in results.md is traceable (skipped if p03 has not been run).
try:
    sw = pd.read_csv(P03_SWEEP)
    row = sw[np.isclose(sw.min_sup, MIN_SUP) & np.isclose(sw.min_conf, MIN_CONF)].iloc[0]
    cons = pd.read_csv(P03_RULES)["consequents"].astype(str)
    ctx["p03"] = dict(n_itemsets=int(row.n_itemsets), n_rules=int(row.n_rules),
                      n_cat_rules=int((cons.str.startswith("{cat=") & ~cons.str.contains(", ")).sum()))
    log("p03 at the same thresholds:", ctx["p03"])
except Exception as e:
    ctx["p03"] = None
    log("p03 counts not available:", repr(e))
tx1.unpersist()

# ---------------------------------------------------------------------------
# 5. Technique B at 1x: accuracy and macro F1 on the shared test set.
#    Both engines are scored with the same sklearn functions on the same rows;
#    macro F1 (not weighted) because p04 reports macro F1.
# ---------------------------------------------------------------------------
log("\n=== B. Decision tree at 1x ===")
tr1 = replicate_sdf(train_sdf, 1)
dt_model, t_dt_s, _, _ = timed(spark_dt, tr1)
pred_pd = dt_model.transform(test_sdf).select("category", "prediction").toPandas()
labels_s = dt_model.stages[2].labelsArray[0]          # label index -> category name
yhat_s = [labels_s[int(p)] if int(p) < len(labels_s) else "<unseen>" for p in pred_pd["prediction"]]
acc_s = accuracy_score(pred_pd["category"], yhat_s)
f1_s = f1_score(pred_pd["category"], yhat_s, average="macro")
tree_s = dt_model.stages[-1]
log(f"Spark DT : acc {acc_s:.4f}  macro F1 {f1_s:.4f}  train {t_dt_s:.2f} s  "
    f"depth {tree_s.depth} nodes {tree_s.numNodes}")

(vec, clf), t_dt_k, _, _ = timed(single_dt, train_pd["title_clean"].tolist(),
                                  train_pd["category"].values)
yhat = clf.predict(vec.transform(test_pd["title_clean"]).toarray().astype(np.float32))
acc_k = accuracy_score(test_pd["category"], yhat)
f1_k = f1_score(test_pd["category"], yhat, average="macro")
log(f"sklearn DT: acc {acc_k:.4f}  macro F1 {f1_k:.4f}  train {t_dt_k:.2f} s  "
    f"depth {clf.get_depth()} nodes {clf.tree_.node_count}")
vocab_overlap = len(set(dt_model.stages[1].vocabulary) & set(vec.get_feature_names_out()))
log(f"vocabulary overlap (Spark vs sklearn top-1000 terms): {vocab_overlap}/1000")
tr1.unpersist()
ctx["dt"] = dict(n_train=len(train_pd), n_test=len(test_pd), vocab_overlap=vocab_overlap,
                 acc_spark=acc_s, f1_spark=f1_s, t_spark_1x=t_dt_s,
                 depth_spark=tree_s.depth, nodes_spark=tree_s.numNodes,
                 acc_sk=acc_k, f1_sk=f1_k, t_sk_1x=t_dt_k,
                 depth_sk=int(clf.get_depth()), nodes_sk=int(clf.tree_.node_count))

# ---------------------------------------------------------------------------
# 6. Optional extra: K-Means k=10 on TF-IDF title features (1x only)
# ---------------------------------------------------------------------------
log("\n=== K-Means k=10 on TF-IDF (1x, optional) ===")
try:
    all_sdf = spark.createDataFrame(df[["title_clean"]]).repartition(NPART).cache()
    all_sdf.count()

    def spark_kmeans():
        words = Tokenizer(inputCol="title_clean", outputCol="words").transform(all_sdf)
        tf = CountVectorizer(inputCol="words", outputCol="tf", vocabSize=1000).fit(words).transform(words)
        tfidf = IDF(inputCol="tf", outputCol="tfidf").fit(tf).transform(tf)
        # L2-normalise so Euclidean k-means behaves like cosine (same as sklearn's TfidfVectorizer)
        feats = Normalizer(inputCol="tfidf", outputCol="features", p=2.0).transform(tfidf).cache()
        m = KMeans(k=10, seed=SEED, maxIter=20, initMode="k-means||").fit(feats)
        return m, feats

    (km_s, feats), t_km_s, _, _ = timed(spark_kmeans)
    sil_s = ClusteringEvaluator(featuresCol="features", predictionCol="prediction") \
        .evaluate(km_s.transform(feats))

    def sk_kmeans():
        X = TfidfVectorizer(max_features=1000, token_pattern=r"\S+").fit_transform(df["title_clean"])
        m = SkKMeans(n_clusters=10, n_init=1, max_iter=20, random_state=SEED).fit(X)
        return m, X

    (km_k, Xk), t_km_k, _, _ = timed(sk_kmeans)
    # silhouette on a 5000-row sample (full silhouette is O(n^2));
    # Spark's evaluator uses the squared-Euclidean "simplified" silhouette on all rows.
    sil_k = silhouette_score(Xk, km_k.labels_, sample_size=5000, random_state=SEED)
    log(f"Spark KMeans  : {t_km_s:.2f} s, silhouette (sq. Euclidean, all rows) {sil_s:.3f}")
    log(f"sklearn KMeans: {t_km_k:.2f} s, silhouette (Euclidean, 5000 sample) {sil_k:.3f}")
    ctx["kmeans"] = dict(ok=True, t_spark=t_km_s, sil_spark=sil_s, t_sk=t_km_k, sil_sk=float(sil_k))
    feats.unpersist()
    all_sdf.unpersist()
except Exception as e:                       # optional part must not kill the run
    log("K-Means part FAILED:", repr(e))
    ctx["kmeans"] = dict(ok=False, error=repr(e))

# ---------------------------------------------------------------------------
# 7. Scalability study: replicate 1x..20x, median of REPEATS runs per point.
#    A series (technique, engine) stops growing once one run exceeds 5 min.
# ---------------------------------------------------------------------------
log("\n=== Scalability study ===")
stopped = {}      # series -> size at which it was stopped
mem_hist = {}     # series -> [(rows, peak Python MB)] of the sizes run so far
mem_skipped = {}  # series -> list of sizes skipped for memory


def run_series(tech, engine, size, fn, *args, n_rows):
    key = (tech, engine)
    if key in stopped:
        log(f"  {tech}/{engine} {size}x skipped (stopped after a run > {TIME_LIMIT:.0f} s at {stopped[key]}x)")
        return
    # Memory guard: mlxtend's apriori (low_memory=False) materialises a
    # rows x candidates boolean array per level, and the dense sklearn matrix
    # also grows with the rows. The peak of the next size is extrapolated
    # linearly from the last two measured sizes (peak = base + slope*rows), and
    # a run that would need more than 60% of the RAM free right now is skipped
    # rather than letting the kernel OOM-kill the whole study.
    hist = mem_hist.get(key, [])
    if len(hist) >= 2:
        (r0, m0), (r1, m1) = hist[-2], hist[-1]
        proj_gb = (m1 + (m1 - m0) / (r1 - r0) * (n_rows - r1)) / 1024
        free_gb = free_ram_gb()[1]
        if proj_gb > 0.6 * free_gb:
            mem_skipped.setdefault(key, []).append(size)
            log(f"  {tech}/{engine} {size}x skipped: projected peak {proj_gb:.1f} GB "
                f"> 60% of {free_gb:.1f} GB free")
            return
    for rep in range(REPEATS):
        _, t, py_mb, jvm_mb = timed(fn, *args)
        rows.append(dict(technique=tech, engine=engine, size=size, rows=n_rows,
                         repeat=rep + 1, seconds=round(t, 4),
                         peak_py_mb=round(py_mb, 1), peak_jvm_mb=round(jvm_mb, 1)))
        log(f"  {tech}/{engine:16s} {size:2d}x rep{rep+1}: {t:8.2f} s  "
            f"peak py {py_mb:7.0f} MB  peak JVM {jvm_mb:7.0f} MB")
        if rep == 0:
            mem_hist.setdefault(key, []).append((n_rows, py_mb))
        if t > TIME_LIMIT:
            stopped[key] = size
            log(f"  -> {tech}/{engine} exceeded {TIME_LIMIT:.0f} s; no larger sizes for it")
            break
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "timings.csv"), index=False)  # save as we go


for k in SIZES:
    log(f"\n-- replication {k}x --")
    # Technique A
    tx_k = transactions * k
    sdf_k = replicate_sdf(tx_sdf, k)
    run_series("arm", "spark_fpgrowth", k, spark_arm, sdf_k, n_rows=N * k)
    run_series("arm", "mlxtend_apriori", k, single_arm, tx_k, apriori, n_rows=N * k)
    run_series("arm", "mlxtend_fpgrowth", k, single_arm, tx_k, fpgrowth, n_rows=N * k)
    sdf_k.unpersist()
    del tx_k
    # Technique B (training set replicated; test set unchanged)
    sdf_k = replicate_sdf(train_sdf, k)
    texts_k = train_pd["title_clean"].tolist() * k
    labels_k = np.tile(train_pd["category"].values, k)
    run_series("dt", "spark_dt", k, spark_dt, sdf_k, n_rows=len(train_pd) * k)
    run_series("dt", "sklearn_dt", k, single_dt, texts_k, labels_k, n_rows=len(train_pd) * k)
    sdf_k.unpersist()
    del texts_k, labels_k

spark.stop()

# ---------------------------------------------------------------------------
# 8. Save timings + summary, then generate results.md and the figures
# ---------------------------------------------------------------------------
tim = pd.DataFrame(rows)
tim.to_csv(os.path.join(OUT, "timings.csv"), index=False)
ctx["stopped"] = {f"{a}/{b}": s for (a, b), s in stopped.items()}
ctx["mem_skipped"] = {f"{a}/{b}": v for (a, b), v in mem_skipped.items()}
# summary.json first: if the report step fails it can be redone with --report-only
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(ctx, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

med = write_report(ctx, tim, OUT)
log("\nMedian timings:\n" + med.to_string(index=False))
for tech in ("arm", "dt"):
    log(f"\nSpeedups, {tech}:\n" + speedup_table(med, tech).to_string())
log("\nwrote", OUT)
