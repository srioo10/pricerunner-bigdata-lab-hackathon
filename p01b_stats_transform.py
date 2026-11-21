"""
Step 1b - Descriptive statistics + the remaining course preprocessing techniques on the numeric attributes.
Numeric attributes (derived in p01): n_chars (cleaned title length), n_tokens (informative words per title),
n_codes (SKU/model codes per title), listings_per_product (listings sharing a cluster_id),
listings_per_merchant (listings of the same merchant).
  - descriptive statistics: count, mean, median, mode, std, min, Q1, Q3, IQR, max, range, skewness, kurtosis
  - normalisation: min-max [0,1], z-score, decimal scaling (n_chars)
  - discretisation: equal-frequency bins on n_chars + smoothing by bin means / medians / boundaries (4 and 6 bins)
  - outliers: IQR fence vs |z| > 3 (counts compared)
  - skew handling: log1p, sqrt, Box-Cox on the right-skewed listings_per_product
  - encoding: integer (label) encoding and one-hot encoding of category
Output: out/stats_transform.md, out/fig_skew_transform.png
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17, "xtick.labelsize": 15,
                     "ytick.labelsize": 15, "legend.fontsize": 14})   # figures are shrunk to page width, so start large
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
md = []
def say(s=""):
    """Print a line and append it to stats_transform.md (the report includes that file)."""
    print(s); md.append(s)

def main():
    # the two count attributes are not in clean.csv; derive them per listing from the group sizes
    df = pd.read_csv(OUT / "clean.csv")
    df["listings_per_product"] = df.groupby("cluster_id")["cluster_id"].transform("size")
    df["listings_per_merchant"] = df.groupby("merchant_id")["merchant_id"].transform("size")
    num = ["n_chars", "n_tokens", "n_codes", "listings_per_product", "listings_per_merchant"]

    # ---- descriptive statistics ----
    say("# Descriptive statistics and data transformation\n")
    say("## Descriptive statistics of the numeric attributes\n\nlistings_per_product and listings_per_merchant are listing-weighted: each listing carries the size of its product group / merchant.\n")
    say("| attribute | count | mean | median | mode | std | min | Q1 | Q3 | IQR | max | range | skewness | kurtosis |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in num:
        x = df[c]; q1, q3 = x.quantile([.25, .75])
        say(f"| {c} | {len(x)} | {x.mean():.2f} | {x.median():.1f} | {x.mode().iloc[0]} | {x.std():.2f} | {x.min()} | "
            f"{q1:.1f} | {q3:.1f} | {q3-q1:.1f} | {x.max()} | {x.max()-x.min()} | {stats.skew(x):.3f} | {stats.kurtosis(x):.3f} |")
    sk = {c: stats.skew(df[c]) for c in num}
    sym = [c for c in num if abs(sk[c]) < 0.5]; mod = [c for c in num if 0.5 <= abs(sk[c]) < 1]; hi = [c for c in num if abs(sk[c]) >= 1]
    say(f"\nRoughly symmetric (|skew| < 0.5): {', '.join(sym) or '-'}. Moderately skewed (0.5-1): {', '.join(mod) or '-'}. "
        f"Highly right-skewed (> 1): {', '.join(hi) or '-'}; for these the median is a better centre than the mean.\n")
    say("Category is nominal: its mode is " + df.category.mode().iloc[0] + " with "
        f"{df.category.value_counts().iloc[0]} listings.\n")
    corr = df[num].corr().round(3)
    say("Pearson correlation matrix:\n\n| | " + " | ".join(num) + " |\n|---|" + "---|" * len(num))
    for c in num: say(f"| {c} | " + " | ".join(f"{v:.3f}" for v in corr.loc[c]) + " |")

    # ---- normalisation of n_chars ----
    x = df["n_chars"].astype(float)
    mm = (x - x.min()) / (x.max() - x.min())
    z = (x - x.mean()) / x.std()
    j = int(np.ceil(np.log10(x.abs().max() + 1)))          # smallest j with max(|x|)/10^j < 1
    ds = x / 10 ** j
    say("\n## Normalisation of n_chars\n")
    say(f"- Min-max: v' = (v - {x.min():.0f}) / ({x.max():.0f} - {x.min():.0f}) -> range [{mm.min():.2f}, {mm.max():.2f}], mean {mm.mean():.3f}")
    say(f"- Z-score: v' = (v - {x.mean():.2f}) / {x.std():.2f} -> mean {z.mean():.3f}, std {z.std():.3f}, range [{z.min():.2f}, {z.max():.2f}]")
    say(f"- Decimal scaling: j = {j}, v' = v / 10^{j} -> range [{ds.min():.3f}, {ds.max():.3f}] (all |v'| < 1)")
    ex = x.iloc[:3].tolist()
    say(f"- Example (first 3 listings): v = {ex} -> min-max {[round(v,3) for v in mm.iloc[:3]]}, "
        f"z {[round(v,3) for v in z.iloc[:3]]}, decimal {[round(v,3) for v in ds.iloc[:3]]}\n")

    # ---- equal-frequency binning + smoothing ----
    say("## Equal-frequency binning and smoothing of n_chars\n")
    changes = {}
    for nb in (4, 6):
        b = pd.qcut(x, nb, duplicates="drop")
        g = x.groupby(b, observed=True)
        say(f"{nb} bins (equal frequency):\n\n| bin | count | min | max | smoothed by mean | smoothed by median |\n|---|---|---|---|---|---|")
        for iv, s in g:
            say(f"| {iv} | {len(s)} | {s.min():.0f} | {s.max():.0f} | {s.mean():.2f} | {s.median():.1f} |")
        bmean = g.transform("mean"); bmed = g.transform("median")
        lo, hi = g.transform("min"), g.transform("max")
        bbound = np.where((x - lo) <= (hi - x), lo, hi)        # each value -> nearer bin boundary
        ch = {"means": np.abs(x - bmean).mean(), "medians": np.abs(x - bmed).mean(), "boundaries": np.abs(x - bbound).mean()}
        say(f"\nMean absolute change after smoothing - by means {ch['means']:.2f}, by medians {ch['medians']:.2f}, "
            f"by boundaries {ch['boundaries']:.2f} characters (least: {min(ch, key=ch.get)}, most: {max(ch, key=ch.get)}).\n")
        changes[nb] = ch
    if all(changes[6][k] < changes[4][k] for k in changes[4]):
        say("With 6 bins instead of 4 every smoothing method moves values less, because each bin is narrower.\n")
    else:
        say("Going from 4 to 6 bins did not reduce the change for every method; see the numbers above.\n")

    # ---- outliers: IQR vs z ----
    say("## Outlier detection: IQR fence vs z-score\n\n| attribute | IQR fence | #outliers (IQR) | #outliers (abs z > 3) |\n|---|---|---|---|")
    more_iqr = []
    for c in num:
        v = df[c]; q1, q3 = v.quantile([.25, .75]); i = q3 - q1
        lo, hi = q1 - 1.5 * i, q3 + 1.5 * i
        n_iqr, n_z = ((v < lo) | (v > hi)).sum(), (np.abs(stats.zscore(v)) > 3).sum()
        if n_iqr > n_z: more_iqr.append(f"{c} ({n_iqr} vs {n_z})")
        say(f"| {c} | [{lo:.1f}, {hi:.1f}] | {n_iqr} | {n_z} |")
    # the z rule uses mean and std, which the extreme values themselves inflate; the IQR fence uses quartiles only
    say(f"\nThe IQR fence flags more points than abs(z) > 3 on: {', '.join(more_iqr) or 'none'}. "
        "The z rule is computed from the mean and std, which the extreme values pull up, so it flags fewer of them. "
        "These fences are recomputed on the cleaned data, so they differ from the p01 cleaning fence (computed on the raw "
        "titles) and the titles they flag are kept. No rows are dropped here; large count values are real "
        "popular products or merchants.\n")

    # ---- skew transforms ----
    v = df["listings_per_product"].astype(float)
    lg, sq = np.log1p(v), np.sqrt(v)
    bc, lam = stats.boxcox(v)
    say("## Skew reduction on listings_per_product\n\n| transform | skewness |\n|---|---|")
    for n_, t in (("none", v), ("log(1+x)", lg), ("sqrt(x)", sq), (f"Box-Cox (lambda={lam:.3f})", bc)):
        say(f"| {n_} | {stats.skew(t):.3f} |")
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for a, (n_, t) in zip(ax, (("original", v), ("log(1+x)", lg), ("sqrt(x)", sq), (f"Box-Cox lambda={lam:.2f}", bc))):
        a.hist(t, bins=30, color="#4C72B0"); a.set_title(f"{n_}\nskew={stats.skew(t):.2f}", fontsize=9)
    fig.suptitle("Skew reduction of listings per product"); fig.tight_layout()
    fig.savefig(OUT / "fig_skew_transform.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- encoding ----
    oh = pd.get_dummies(df["category"], prefix="cat", dtype=int)
    say(f"\n## Encoding of category\n\n- Integer (label) encoding: {dict(zip(sorted(df.category.unique()), range(10)))}")
    say(f"- One-hot encoding: {oh.shape[1]} binary columns ({', '.join(oh.columns[:3])}, ...); every row sums to "
        f"{int(oh.sum(axis=1).min())}. This frame is built only to show the encoding. The integer codes are the class label "
        "for the decision tree; the ARM transactions (p03, p05) carry the category as one item 'cat=<category>', "
        "which is the same one-of-ten idea written as an item.")
    (OUT / "stats_transform.md").write_text("\n".join(md))

if __name__ == "__main__":
    main()
