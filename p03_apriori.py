"""
Step 3 - Association Rule Mining with Apriori.
Transaction = one product listing = set of its cleaned title tokens + one item 'cat=<category>'.
Assumed parameters (stated, as the exam allows):
  Global DB (34k listings, 7k items, very sparse): min_sup in {1%, 2%, 5%}, min_conf in {70%, 80%, 90%}
  Per-category DBs (denser): (min_sup, min_conf) in {(20%, 80%), (10%, 80%), (5%, 70%)}
Library: mlxtend.apriori (+ association_rules). Cross-check: own from-scratch Apriori (own_apriori.py).
Outputs: out/arm/*.csv, out/arm/arm_summary.md
"""
import sys, time
import pandas as pd
from pathlib import Path
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import own_apriori   # from-scratch Apriori, used as a correctness check

OUT = ROOT / "out/arm"; OUT.mkdir(parents=True, exist_ok=True)
md = []
def say(s=""):
    print(s); md.append(s)

def load_transactions(df):
    return [set(t.split()) | {f"cat={c}"} for t, c in zip(df["tokens"], df["category"])]

def onehot(T):
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(T).transform(T, sparse=False), columns=te.columns_)

def timed(fn, reps=3):
    """Run fn reps times; return its result and the median wall time (one run moves by a few tenths of a second)."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, sorted(ts)[len(ts) // 2]

def mine(X, sup, conf):
    fi, t_fi = timed(lambda: apriori(X, min_support=sup, use_colnames=True, low_memory=True))
    t0 = time.perf_counter()
    rules = association_rules(fi, metric="confidence", min_threshold=conf) if len(fi) else pd.DataFrame()
    if len(rules):   # fixed alphabetical order first, so ties in later sorts always break the same way
        key = rules.antecedents.map(fmt) + rules.consequents.map(fmt)
        rules = rules.loc[key.sort_values(kind="mergesort").index].reset_index(drop=True)
    return fi, rules, t_fi, t_fi + time.perf_counter() - t0

def distinct(rules):
    """Keep a rule only if it shares no item with a rule ranked above it, so one phrase cannot fill the table."""
    kept, seen = [], set()
    for i, (a, c) in zip(rules.index, zip(rules.antecedents, rules.consequents)):
        if not (a | c) & seen:
            kept.append(i); seen |= a | c
    return rules.loc[kept]

def fmt(s): return "{" + ", ".join(sorted(s)) + "}"

def rules_table(rules, n=15):
    rows = ["| # | Rule (IF -> THEN) | support | confidence | lift |", "|---|---|---|---|---|"]
    for i, r in enumerate(rules.head(n).itertuples(), 1):
        rows.append(f"| {i} | {fmt(r.antecedents)} -> {fmt(r.consequents)} | {r.support:.4f} | {r.confidence:.3f} | {r.lift:.2f} |")
    return "\n".join(rows)

def main():
    df = pd.read_csv(ROOT / "out/clean.csv")
    T = load_transactions(df)
    X = onehot(T)
    say(f"# Apriori association rules\n\nTransactions: {len(T)}, distinct items: {X.shape[1]}, "
        f"avg items/transaction: {X.sum(1).mean():.2f}, density: {X.values.mean():.5f}\n")

    # ---- 1. parameter sweep on the global DB ----
    say("## Parameter sweep (global DB)\n\n| min_sup | min_conf | #frequent itemsets | max size | #rules | time (s, median of 3) |\n|---|---|---|---|---|---|")
    sweep = []
    for sup in (0.05, 0.02, 0.01):
        for conf in (0.9, 0.8, 0.7):
            fi, rules, tfi, tall = mine(X, sup, conf)
            mx = fi["itemsets"].map(len).max()
            sweep.append(dict(min_sup=sup, min_conf=conf, n_itemsets=len(fi), max_len=mx, n_rules=len(rules), time_s=tall))
            say(f"| {sup:.0%} | {conf:.0%} | {len(fi)} | {mx} | {len(rules)} | {tall:.2f} |")
    pd.DataFrame(sweep).to_csv(OUT / "sweep_global.csv", index=False)
    say("\nLower support -> many more itemsets and rules and longer runtime (the candidate space grows); "
        "higher confidence keeps only the near-deterministic rules.\n")

    # ---- 2. chosen setting: sup 1%, conf 80% ----
    fi, rules, _, _ = mine(X, 0.01, 0.8)
    rules = rules.sort_values(["lift", "confidence"], ascending=False, kind="mergesort")
    rules.assign(antecedents=rules.antecedents.map(fmt), consequents=rules.consequents.map(fmt)) \
         .to_csv(OUT / "rules_global_s1_c80.csv", index=False)
    say(f"## Chosen setting min_sup=1%, min_conf=80% -> {len(fi)} frequent itemsets, {len(rules)} rules\n")
    # many rules are permutations of one itemset (e.g. 'hewlett packard enterprise'): keep one
    # single-consequent rule per itemset so the top list shows distinct patterns
    nr = rules[rules.consequents.map(len) == 1].copy()
    nr["itemset"] = [a | c for a, c in zip(nr.antecedents, nr.consequents)]
    nr = nr.sort_values(["lift", "confidence"], ascending=False, kind="mergesort").drop_duplicates("itemset")
    nr = nr[~nr.itemset.map(lambda s: any(x.startswith("cat=") for x in s))]
    # many top rules are re-orderings of one phrase ('hewlett packard enterprise'); to show different patterns,
    # a rule is skipped if it shares any item with a rule already listed
    nr = distinct(nr.sort_values(["lift", "confidence"], ascending=False, kind="mergesort"))
    say("### Top 15 rules by lift, no item repeated (a rule sharing any item with a higher-ranked rule is skipped; category item excluded)\n"); say(rules_table(nr))
    cls = rules[rules.consequents.map(lambda c: len(c) == 1 and next(iter(c)).startswith("cat="))]
    cls = cls.sort_values(["confidence", "support"], ascending=False, kind="mergesort")
    say(f"\n### Class-association rules (THEN part = category): {len(cls)} rules, top 15 by confidence\n")
    say(rules_table(cls))
    say("\nOn this corpus, one or two title words co-occur with a single category with ~100% confidence. These are "
        "descriptive association rules mined from all listings, not a validated classifier; held-out classification "
        "performance is measured separately with the decision tree.\n")

    # ---- 2b. same mining WITHOUT the inserted category item: only word-word associations ----
    Xw = X[[c for c in X.columns if not c.startswith("cat=")]]
    fi_w, rules_w, _, t_w = mine(Xw, 0.01, 0.8)
    say(f"## Words only (no category item), min_sup 1%, min_conf 80%\n\n{len(fi_w)} frequent itemsets and {len(rules_w)} "
        f"rules in {t_w:.2f}s (vs {len(fi)} / {len(rules)} with the category item). These rules are pure title-word "
        "associations, free of the inserted label. Top 10 single-consequent rules by confidence then support, "
        "again with no item repeated:\n")
    rw = distinct(rules_w[rules_w.consequents.map(len) == 1].sort_values(["confidence", "support"], ascending=False, kind="mergesort")).head(10)
    say(rules_table(rw, 10) + "\n")
    # ---- 3. per-category mining: denser databases, so higher supports (20/10/5%) ----
    say("## Per-category Apriori (min_sup 20%, 10%, 5%)\n\nInside one category the database is much denser, so higher supports are possible; 20% finds few rules in most categories, so 10% and 5% are also shown.\n")
    say("| category | min_sup | min_conf | #itemsets | #rules | top rule (by lift) |\n|---|---|---|---|---|---|")
    per = []
    for cat in sorted(df["category"].unique()):
        sub = df[df.category == cat]
        Xc = onehot([set(t.split()) for t in sub["tokens"]])
        for sup, conf in ((0.2, 0.8), (0.1, 0.8), (0.05, 0.7)):
            fi_c, r_c, _, t = mine(Xc, sup, conf)
            top = r_c.sort_values("lift", ascending=False, kind="mergesort").iloc[0] if len(r_c) else None
            tr = (f"{fmt(top.antecedents)} -> {fmt(top.consequents)} (c={top.confidence:.2f}, L={top.lift:.2f})"
                  if top is not None else "-")
            per.append(dict(category=cat, min_sup=sup, min_conf=conf, n_itemsets=len(fi_c), n_rules=len(r_c), top_rule=tr))
            say(f"| {cat} | {sup:.0%} | {conf:.0%} | {len(fi_c)} | {len(r_c)} | {tr} |")
    pd.DataFrame(per).to_csv(OUT / "per_category.csv", index=False)

    # ---- 4. correctness check vs from-scratch Apriori (TVs at 10%: deep enough that join + prune really run) ----
    sub = df[df.category == "TVs"]
    Tm = [frozenset(t.split()) for t in sub["tokens"]]
    t0 = time.perf_counter(); own = own_apriori.apriori(Tm, 0.1); t_own = time.perf_counter() - t0
    fi_m, _, t_lib, _ = mine(onehot([set(t) for t in Tm]), 0.1, 0.7)
    lib = {frozenset(s): round(v * len(Tm)) for s, v in zip(fi_m.itemsets, fi_m.support)}
    same = lib == own
    say(f"\n## Cross-check (TVs, min_sup 10%)\nmlxtend: {len(lib)} itemsets (longest {max(map(len, lib))} items) in {t_lib:.3f}s; "
        f"own Apriori: {len(own)} itemsets in {t_own:.3f}s; identical itemsets and support counts: {'yes' if same else 'NO'}")
    (OUT / "arm_summary.md").write_text("\n".join(md))

if __name__ == "__main__":
    main()
