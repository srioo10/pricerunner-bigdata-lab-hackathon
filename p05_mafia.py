"""
Step 5 - Maximal Frequent Itemsets (MFIs) with MAFIA (Burdick, Calimlim, Gehrke, ICDE 2001).
Binary: MAFIA 1.4 C++ source (FIMI repository), compiled to exam/bin/mafia.
  MAFIA = depth-first search over the itemset lattice using vertical bitmaps, with
  Parent-Equivalence Pruning (PEP), Frequent Head Union Tail (FHUT) and HUTMFI superset pruning.
Pipeline:
  1. encode each listing (token set + 'cat=<category>') as integer items -> FIMI .dat file
  2. run MAFIA -mfi (maximal), and -fci / -fi at the same support to show the compression FI >= CFI >= MFI
  3. cross-check the MFIs against mlxtend.fpmax (independent implementation)
  4. derive 'maximal association rules' X -> (M \\ X) from every MFI M (confidence = sup(M)/sup(X)), keeping per MFI
     only the rules with the smallest antecedent. This is a chosen subset of the confident rules, not every association rule.
Assumed parameters: min_sup in {5%, 2%, 1%, 0.5%}; rules from MFIs at min_sup 1%, min_conf 80%.
Outputs: out/mafia/*.txt|csv, out/mafia/mafia_summary.md
"""
import subprocess, time, math
from itertools import combinations
import pandas as pd
from pathlib import Path
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpmax, apriori, fpgrowth
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MAFIA = ROOT / "bin/mafia"     # MAFIA 1.4 compiled from the FIMI repository source (g++ -O3 src/*.cpp)
OUT = ROOT / "out/mafia"; OUT.mkdir(parents=True, exist_ok=True)
md = []
def say(s=""):
    """Print a line and append it to mafia_summary.md (the report includes that file)."""
    print(s); md.append(s)

def run_mafia(mode, dat, frac):
    """mode: -mfi | -fci | -fi ; frac: relative support (MAFIA uses ceil(frac*N))."""
    res = OUT / f"mafia{mode}_{frac}.txt"
    res.unlink(missing_ok=True)                        # MAFIA may write nothing when no itemset is frequent
    t0 = time.perf_counter()
    subprocess.run([str(MAFIA), mode, str(frac), "-ascii", str(dat), str(res)], check=True, capture_output=True)
    t = time.perf_counter() - t0
    sets = []
    for line in (res.read_text().splitlines() if res.exists() else []):          # format: 'i1 i2 ... (support)'
        if "(" not in line: continue
        items, sup = line.rsplit("(", 1)
        if items.split():
            sets.append((frozenset(map(int, items.split())), int(sup.strip(") "))))
    return sets, t

def main():
    if not MAFIA.is_file():
        raise SystemExit(f"MAFIA binary not found at {MAFIA}. Build it from the FIMI source: "
                         "g++ -O3 -o bin/mafia src/*.cpp (see README).")
    df = pd.read_csv(ROOT / "out/clean.csv")
    T = [frozenset(t.split()) | {f"cat={c}"} for t, c in zip(df["tokens"], df["category"])]
    N = len(T)
    items = sorted({i for t in T for i in t})
    code = {it: k + 1 for k, it in enumerate(items)}; name = {v: k for k, v in code.items()}
    dat = OUT / "products.dat"
    dat.write_text("\n".join(" ".join(str(code[i]) for i in sorted(t, key=code.get)) for t in T) + "\n")
    say(f"# MAFIA maximal frequent itemsets\n\nFIMI file: {N} transactions, {len(items)} items -> {dat.name}\n")

    # ---- 1. support sweep: |FI| vs |CFI| vs |MFI| ----
    say("| min_sup | count | #FI | #CFI | #MFI | MAFIA time MFI (s) | longest MFI |\n|---|---|---|---|---|---|---|")
    sweep, mfi_at = [], {}
    for frac in (0.05, 0.02, 0.01, 0.005):
        mfi, t_m = run_mafia("-mfi", dat, frac)
        mfi_at[frac] = (mfi, t_m)                      # reused below for the 1% run, so MAFIA is not run twice
        cfi, _ = run_mafia("-fci", dat, frac)
        fi, _ = run_mafia("-fi", dat, frac)
        longest = max((len(s) for s, _ in mfi), default=0)
        sweep.append(dict(min_sup=frac, count=math.ceil(frac * N), FI=len(fi), CFI=len(cfi), MFI=len(mfi), t_mfi=t_m, longest=longest))
        say(f"| {frac:.1%} | {math.ceil(frac*N)} | {len(fi)} | {len(cfi)} | {len(mfi)} | {t_m:.3f} | {longest} |")
    s1 = next(r for r in sweep if r["min_sup"] == 0.01)
    say(f"\nAt 1% support {s1['FI']} frequent itemsets reduce to {s1['CFI']} closed and {s1['MFI']} maximal ones. "
        "Every FI is a subset of some MFI, so the MFIs mark the border of the frequent region, but (unlike closed "
        "itemsets) they do not keep the support of the subsets.\n")

    # ---- 2. main run at 1% + cross-check with fpmax ----
    frac = 0.01
    mfi, t_m = mfi_at[frac]
    te = TransactionEncoder(); X = pd.DataFrame(te.fit([list(t) for t in T]).transform([list(t) for t in T]), columns=te.columns_)
    t0 = time.perf_counter(); fp = fpmax(X, min_support=math.ceil(frac * N) / N - 1e-12, use_colnames=True); t_fp = time.perf_counter() - t0
    m_set = {frozenset(name[i] for i in s): c for s, c in mfi}
    f_set = {frozenset(s): round(v * N) for s, v in zip(fp.itemsets, fp.support)}
    say(f"## min_sup = 1% ({math.ceil(frac*N)} listings)\nMAFIA: {len(m_set)} MFIs in {t_m:.3f}s; "
        f"mlxtend fpmax: {len(f_set)} MFIs in {t_fp:.2f}s; identical sets and supports: **{m_set == f_set}**\n")
    # ---- run time vs support: Apriori (level-wise), FP-growth (pattern growth), MAFIA (depth-first, maximal only)
    sw = pd.DataFrame(sweep); ta, tf_ = [], []
    for fr in sw.min_sup:
        t0 = time.perf_counter(); apriori(X, min_support=fr, use_colnames=True, low_memory=True); ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); fpgrowth(X, min_support=fr, use_colnames=True); tf_.append(time.perf_counter() - t0)
    sw["t_apriori"], sw["t_fpgrowth"] = ta, tf_
    sw.to_csv(OUT / "sweep.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = [f"{v:.1%}" for v in sw.min_sup]
    ax[0].plot(xs, sw.t_apriori, "o-", label="Apriori (mlxtend, all FIs)")
    ax[0].plot(xs, sw.t_fpgrowth, "s-", label="FP-growth (mlxtend, all FIs)")
    ax[0].plot(xs, sw.t_mfi, "^-", label="MAFIA (C++, MFIs only)")
    ax[0].set(yscale="log", xlabel="minimum support", ylabel="run time (s, log scale)", title="Run time vs minimum support")
    ax[0].legend(fontsize=8)
    w = 0.27; ix = range(len(sw))
    ax[1].bar([i - w for i in ix], sw.FI, w, label="frequent (FI)")
    ax[1].bar(list(ix), sw.CFI, w, label="closed (CFI)")
    ax[1].bar([i + w for i in ix], sw.MFI, w, label="maximal (MFI)")
    ax[1].set_xticks(list(ix), xs); ax[1].set(yscale="log", xlabel="minimum support", ylabel="itemsets (log scale)",
                                             title="FI vs CFI vs MFI counts"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_mafia_sweep.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    say("### Run time vs support\n\n| min_sup | Apriori (s) | FP-growth (s) | MAFIA (s) |\n|---|---|---|---|")
    for r in sw.itertuples():
        say(f"| {r.min_sup:.1%} | {r.t_apriori:.2f} | {r.t_fpgrowth:.2f} | {r.t_mfi:.3f} |")
    say(f"\nApriori time grows {sw.t_apriori.iloc[-1]/sw.t_apriori.iloc[0]:.1f}x from {sw.min_sup.iloc[0]:.0%} to "
        f"{sw.min_sup.iloc[-1]:.1%} support (candidate generation over more levels); FP-growth grows "
        f"{sw.t_fpgrowth.iloc[-1]/sw.t_fpgrowth.iloc[0]:.1f}x. MAFIA mines only the maximal border, so its output grows "
        f"from {sw.MFI.iloc[0]} to {sw.MFI.iloc[-1]} itemsets while FIs grow from {sw.FI.iloc[0]} to {sw.FI.iloc[-1]}. "
        "Times compare implementations (Python libraries vs compiled C++), not only algorithms.\n")

    rows = sorted(m_set.items(), key=lambda kv: (-len(kv[0]), -kv[1]))
    pd.DataFrame([dict(mfi=" ".join(sorted(s)), size=len(s), support_count=c, support=c / N) for s, c in rows]) \
        .to_csv(OUT / "mfi_1pct.csv", index=False)
    say("### Longest MFIs\n\n| # | maximal itemset | size | support |\n|---|---|---|---|")
    for k, (s, c) in enumerate(rows[:15], 1):
        say(f"| {k} | {{{', '.join(sorted(s))}}} | {len(s)} | {c/N:.4f} ({c}) |")
    by_cat = pd.Series([next((x[4:] for x in s if x.startswith("cat=")), "(no category item)") for s in m_set]).value_counts()
    say("\nMFIs per category item: " + ", ".join(f"{k}: {v}" for k, v in by_cat.items()) + "\n")

    # ---- 3. maximal association rules ----
    # support of any subset X of an MFI is counted directly (MFIs alone don't store it)
    tid = {}
    for k, t in enumerate(T):
        for i in t: tid.setdefault(i, set()).add(k)
    def sup(X): return len(set.intersection(*(tid[i] for i in X)))
    rules = []
    for M, cM in m_set.items():
        if len(M) < 2: continue
        for r in range(1, len(M)):
            for Xa in combinations(sorted(M), r):
                X = frozenset(Xa); cX = sup(X); conf = cM / cX
                if conf >= 0.8:
                    Y = M - X
                    lift = conf / (sup(Y) / N)
                    rules.append(dict(antecedent=", ".join(sorted(X)), consequent=", ".join(sorted(Y)),
                                      support=cM / N, confidence=conf, lift=lift, len_X=len(X)))
    if not rules:
        say("No rule reaches confidence 0.8 at 1% support.")
        (OUT / "mafia_summary.md").write_text("\n".join(md)); return
    R = pd.DataFrame(rules)
    # rule policy: from each MFI keep only the rules with the SMALLEST antecedent that reach min_conf
    # (the most general IF part); longer antecedents for the same MFI are redundant variants
    R["mfi"] = [" ".join(sorted(set(a.split(", ")) | set(c.split(", ")))) for a, c in zip(R.antecedent, R.consequent)]
    n_all = len(R)
    R = R[R.len_X == R.groupby("mfi").len_X.transform("min")]
    R = R.sort_values(["len_X", "confidence", "lift"], ascending=[True, False, False])
    R.to_csv(OUT / "maximal_rules_1pct_c80.csv", index=False)
    say(f"## Maximal association rules (X -> MFI minus X, min_conf 80%)\n\nThese rules are derived from the MFIs only, "
        "so they are a subset of all confident association rules, not the full set. Rule policy: for every MFI M and every non-empty "
        f"X subset of M, X -> M minus X is kept if sup(M)/sup(X) >= 0.8 ({n_all} rules); then, per MFI, only the rules with the "
        f"smallest antecedent are kept ({len(R)} rules from {R.mfi.nunique()} MFIs).\n")
    say("| # | IF | THEN | support | confidence | lift |\n|---|---|---|---|---|---|")
    best = R.sort_values(["support", "confidence"], ascending=False).drop_duplicates("consequent").head(20)
    for k, r in enumerate(best.itertuples(), 1):
        say(f"| {k} | {{{r.antecedent}}} | {{{r.consequent}}} | {r.support:.4f} | {r.confidence:.3f} | {r.lift:.2f} |")
    say("\nEach rule's IF+THEN is a whole maximal itemset, so these rules describe the largest product "
        "descriptions that still occur in >= 1% of listings.")
    (OUT / "mafia_summary.md").write_text("\n".join(md))

if __name__ == "__main__":
    main()
