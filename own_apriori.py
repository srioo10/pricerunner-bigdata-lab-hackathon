"""From-scratch Apriori (level-wise join + prune), used in p03 to check the mlxtend results."""
import math
from itertools import combinations


def apriori(transactions, min_sup):
    """Return {frozenset(itemset): support count} for every itemset with support >= min_sup.

    min_sup is a fraction of the transactions (e.g. 0.1) or an absolute count (int).
    """
    T = [frozenset(t) for t in transactions]
    ms = min_sup if isinstance(min_sup, int) else math.ceil(min_sup * len(T))
    Ck = [frozenset([i]) for i in sorted({i for t in T for i in t})]   # C1: every single item
    frequent, k = {}, 1
    while Ck:
        counts = {c: sum(1 for t in T if c <= t) for c in Ck}          # one pass over the data per level
        Lk = {c: n for c, n in counts.items() if n >= ms}
        frequent.update(Lk)
        # join: two frequent k-itemsets that share their first k-1 items (in sorted order) form a (k+1)-candidate
        Ls = sorted(sorted(c) for c in Lk)
        cand = {frozenset(a) | frozenset(b) for a, b in combinations(Ls, 2) if a[:-1] == b[:-1]}
        # prune: a candidate with any infrequent k-subset cannot be frequent (Apriori property)
        Ck = [c for c in cand if all(frozenset(s) in Lk for s in combinations(c, k))]
        k += 1
    return frequent
