"""
Step 2 - Exploratory visualisation of the cleaned PriceRunner listings (UCI #837).
16 figures (14 distinct plot types; bar and histogram are each used twice, for different attributes), saved as out/figs/figNN_*.png (150 dpi, tight bbox),
plus the numeric inferences they support in out/plot_inferences.md.

Run: ../.venv/bin/python p02_plots.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless backend: no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "out/figs"
FIG.mkdir(parents=True, exist_ok=True)
MD = ROOT / "out/plot_inferences.md"

# setup
df = pd.read_csv(ROOT / "out/clean.csv")
N = f"{len(df):,}"
CATS = df.category.value_counts().index.tolist()          # categories, largest first
PAL = dict(zip(CATS, sns.color_palette("tab10", len(CATS)).as_hex()))  # stable colour per category
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.autolayout": False})
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 11})   # figures are shrunk to page width, so start large

INF = []                                                    # (fig id, title, caption, [bullets]) -> markdown


def save(fig, name, title, shrink=0.8, layout=True):
    """Save the PNG at 150 dpi on a canvas sized for the PDF."""
    fig.set_size_inches(fig.get_size_inches() * shrink)   # the PDF shows plots ~4 in wide; a smaller canvas keeps text legible there
    if layout: fig.tight_layout()   # title is the report heading, so it is not repeated on the image
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def note(fig_id, title, caption, bullets):
    """Store the caption (what is plotted) and the inferences, and print them."""
    INF.append((fig_id, title, caption, bullets))
    print(f"\n## {fig_id} - {title}\n{caption}")
    for b in bullets:
        print(f"- {b}")


def pct(x):
    """Fraction -> '12.3%' (one format for every bullet)."""
    return f"{100 * x:.1f}%"


def gini(v):
    """Gini coefficient of a non-negative vector (0 = equal, ->1 = concentrated)."""
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


# fig01 bar
counts = df.category.value_counts()
fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.bar(counts.index, counts.values, color=[PAL[c] for c in counts.index],
              edgecolor="black", linewidth=0.5)
ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], padding=2, fontsize=10, rotation=90)
ax.set_xlabel("Category")
ax.set_ylabel("Number of listings")
ax.set_title("Listing count per product category (n = %s)" % f"{len(df):,}")
ax.tick_params(axis="x", rotation=40); plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
ax.margins(y=0.12)
save(fig, "fig01_bar_category_counts", "Listings per category")
fair = len(df) / len(counts)                                 # size of each class if all were equal
near = counts[(counts - fair).abs() <= 0.2 * fair]
note("fig01", "Listings per category", f"Listings per category (bar chart, n = {N})", [
    f"{counts.index[0]} is the largest category with {counts.iloc[0]:,} listings and {counts.index[-1]} the smallest "
    f"with {counts.iloc[-1]:,}, a largest/smallest ratio of {counts.iloc[0] / counts.iloc[-1]:.2f}x.",
    f"With equal classes each would have {fair:,.0f} listings; {len(near)} of the {len(counts)} categories are within "
    f"20% of that, and the median category has {counts.median():,.0f} listings.",
])

# fig02 doughnut
fig, ax = plt.subplots(figsize=(7.2, 6))
wedges, _, autot = ax.pie(counts.values, colors=[PAL[c] for c in counts.index],
                          startangle=90, counterclock=False,
                          autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
                          pctdistance=0.78,
                          wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
                          textprops=dict(fontsize=10))
ax.legend(wedges, counts.index, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=False)
ax.text(0, 0, f"{len(df):,}\nlistings", ha="center", va="center", fontsize=11, fontweight="bold")
ax.set_title("Share of listings by category (labels shown for shares >= 5%)")
save(fig, "fig02_doughnut_category_share", "Category share of listings")
shares = counts / len(df)
note("fig02", "Category share of listings", f"Share of listings per category (doughnut chart, n = {N})", [
    f"The two largest categories ({counts.index[0]}, {counts.index[1]}) together are {pct(shares.iloc[:2].sum())} of "
    f"the corpus and the top three {pct(shares.iloc[:3].sum())}; the smallest, {counts.index[-1]}, still has "
    f"{pct(shares.iloc[-1])}.",
    f"The Herfindahl index of the class shares is {float((shares ** 2).sum()):.3f} (1/10 = 0.100 for a perfectly "
    f"balanced 10-class problem), so the imbalance is mild.",
])

# fig03 histogram n_chars
x = df.n_chars
mu, med = x.mean(), x.median()
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.hist(x, bins=40, color="#4c72b0", edgecolor="white", alpha=0.9)
ax.axvline(mu, color="#c44e52", lw=2, label=f"mean = {mu:.1f}")
ax.axvline(med, color="#dd8452", lw=2, ls="--", label=f"median = {med:.0f}")
ax.set_xlabel("Title length (characters, cleaned)")
ax.set_ylabel("Number of listings")
ax.set_title("Distribution of cleaned title length")
ax.legend(frameon=False)
save(fig, "fig03_hist_nchars", "Histogram of title length")
q1, q3 = x.quantile([0.25, 0.75])
p10, p90 = x.quantile([0.10, 0.90])
sk = stats.skew(x)
shape = ("nearly symmetric" if abs(sk) < 0.5 else "right-skewed" if sk > 0 else "left-skewed")
note("fig03", "Histogram of title length (n_chars)",
     f"Cleaned title length in characters (histogram, 40 bins, n = {N})", [
    f"Mean {mu:.1f} and median {med:.0f} characters, skewness {sk:.2f}: title length is {shape}. "
    f"The middle 50% lies between {q1:.0f} and {q3:.0f} characters (IQR = {q3 - q1:.0f}).",
    f"The most common exact length is {x.value_counts().idxmax()} characters ({x.value_counts().max():,} listings, "
    f"{pct(x.value_counts().max() / len(x))}), and 80% of titles fall between {p10:.0f} and {p90:.0f} characters.",
])

# fig04 box n_tokens by category
order = df.groupby("category").n_tokens.median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="category", y="n_tokens", order=order, hue="category",
            palette=PAL, legend=False, width=0.6, fliersize=2, ax=ax)
ax.set_xlabel("Category (ordered by median)")
ax.set_ylabel("Tokens per title")
ax.set_title("Token-count spread per category")
ax.tick_params(axis="x", rotation=40); plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
save(fig, "fig04_box_ntokens_category", "Box plot of tokens by category")
med_by = df.groupby("category").n_tokens.median()
iqr_by = df.groupby("category").n_tokens.quantile(0.75) - df.groupby("category").n_tokens.quantile(0.25)
note("fig04", "Box plot of tokens per category",
     f"Tokens per title by category, ordered by median (box plot, n = {N})", [
    f"{med_by.idxmax()} has the longest titles by median ({med_by.max():.0f} tokens); "
    f"{med_by.idxmin()} the shortest ({med_by.min():.0f}).",
    f"The widest token spread (IQR = {iqr_by.max():.0f} tokens) is shared by {', '.join(iqr_by[iqr_by == iqr_by.max()].index)}, against "
    f"{iqr_by.min():.0f} in the tightest categories ({', '.join(iqr_by[iqr_by == iqr_by.min()].index)}).",
    f"Over all categories the median title has {df.n_tokens.median():.0f} tokens and 75% of titles have at most "
    f"{df.n_tokens.quantile(0.75):.0f}.",
])

# fig05 violin n_chars by category
order = df.groupby("category").n_chars.median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(10, 5.2))
sns.violinplot(data=df, x="category", y="n_chars", order=order, hue="category",
               palette=PAL, legend=False, cut=0, inner="quartile", linewidth=0.8, ax=ax)
ax.set_xlabel("Category (ordered by median)")
ax.set_ylabel("Title length (characters)")
ax.set_title("Title-length density per category")
ax.tick_params(axis="x", rotation=40); plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
save(fig, "fig05_violin_nchars_category", "Violin plot of title length by category")
cm = df.groupby("category").n_chars.median()
ciqr = df.groupby("category").n_chars.quantile(.75) - df.groupby("category").n_chars.quantile(.25)
oq1, oq3 = df.n_chars.quantile([0.25, 0.75])
note("fig05", "Violin plot of title length by category",
     f"Title length in characters by category, with quartile lines (violin plot, n = {N})", [
    f"Median title length ranges from {cm.min():.0f} ({cm.idxmin()}) to {cm.max():.0f} characters "
    f"({cm.idxmax()}). The gap of {cm.max() - cm.min():.0f} characters is close to the IQR of all titles "
    f"({oq3 - oq1:.0f}, from {oq1:.0f} to {oq3:.0f}).",
    f"The IQR of title length per category runs from {ciqr.min():.0f} ({ciqr.idxmin()}) to {ciqr.max():.0f} "
    f"characters ({ciqr.idxmax()}).",
])

# fig06 pareto top-25 brands
bc = df.brand.value_counts()
top25 = bc.head(25)
cum_all = bc.cumsum() / len(df)
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(range(1, 26), top25.values, color="#4c72b0", edgecolor="black", linewidth=0.4)
ax.set_xlabel("Brand (ranked by number of listings)")
ax.set_ylabel("Listings")
ax.set_xticks(range(1, 26), top25.index, rotation=60, ha="right", rotation_mode="anchor", fontsize=10)   # brand names, readable
ax.set_ylim(0, top25.max() * 1.15)
ax2 = ax.twinx()
ax2.plot(range(1, 26), 100 * cum_all.head(25).values, color="#c44e52", marker="o", ms=3, lw=1.6,
         label="cumulative % of listings")
ax2.axhline(80, color="grey", ls="--", lw=1.2, label="80% line")
ax2.set_ylabel("Cumulative share of listings (%)")
ax2.set_ylim(0, 105)
h1, l1 = ax2.get_legend_handles_labels()
ax.legend(h1, l1, loc="center right", frameon=False, fontsize=10)
ax.set_title("Bars = listings per brand, line = cumulative share")
save(fig, "fig06_pareto_brands", "Pareto chart of top-25 brands")
n80 = int(np.searchsorted(cum_all.values, 0.80) + 1)
small = int((bc < 10).sum())
note("fig06", "Pareto chart of top-25 brands",
     f"Listings of the 25 largest brands with the cumulative share of all {N} listings "
     f"(Pareto chart, {bc.size:,} brand tokens)", [
    f"The top 25 brands cover {pct(cum_all.iloc[24])} of all listings. Of the {bc.size:,} brand tokens (brand = first "
    f"informative word of the title, a proxy), {n80} are needed to cover 80% of listings.",
    f"The single largest brand, '{bc.index[0]}', appears in {bc.iloc[0]:,} listings ({pct(bc.iloc[0] / len(df))}), "
    f"about {bc.iloc[0] / bc.iloc[1]:.1f}x the runner-up '{bc.index[1]}' ({bc.iloc[1]:,}).",
    f"{small} of the {bc.size:,} brand tokens ({pct(small / bc.size)}) have fewer than 10 listings and "
    f"{int((bc == 1).sum())} appear in only one.",
])

# fig07 heatmap category x top-15 brands
top15 = bc.head(15).index.tolist()
ct = pd.crosstab(df.category, df.brand).reindex(columns=top15).fillna(0).astype(int)
ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(11, 5.6))
sns.heatmap(np.log1p(ct), annot=ct.values, fmt="d", cmap="YlGnBu", linewidths=0.5,
            cbar_kws={"label": "log1p(listings)"}, annot_kws={"fontsize": 9}, ax=ax)
ax.set_xlabel("Brand (top 15 by listings)")
ax.set_ylabel("Category")
ax.set_title("Brand presence per category (counts annotated)")
ax.tick_params(axis="x", rotation=40); plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
save(fig, "fig07_heatmap_category_brand", "Category x brand heatmap", shrink=1.0)
cell = ct.stack().idxmax()
cov = (ct > 0).sum(axis=0)                                  # in how many categories each brand appears
npres = (ct > 0).sum(axis=1)                                # how many of the top-15 brands each category has
topshare = df.groupby("category").brand.agg(lambda b: b.value_counts().iloc[0] / len(b))
note("fig07", "Category x top-15 brand heatmap",
     "Listings per category for the 15 largest brands (heatmap, colour = log(1 + count), counts annotated)", [
    f"The densest brand-category cell is {cell[0]} x {cell[1]} with {ct.stack().max():,} listings; the most "
    f"widespread brand is '{cov.idxmax()}', present in {cov.max()} of the {len(CATS)} categories.",
    f"Listings of the top-15 brands make up {pct(ct.values.sum() / len(df))} of the corpus. Within a category the "
    f"largest brand's share ranges from {pct(topshare.min())} ({topshare.idxmin()}) to "
    f"{pct(topshare.max())} ({topshare.idxmax()}).",
    f"{' and '.join(npres[npres == npres.max()].index)} "
    f"{'contains' if (npres == npres.max()).sum() == 1 else 'each contain'} {npres.max()} of the top-15 brands; "
    f"{' and '.join(npres[npres == npres.min()].index)} "
    f"{'contains' if (npres == npres.min()).sum() == 1 else 'each contain'} only {npres.min()}.",
])

# fig08 scatter n_tokens vs n_chars
samp = df.sample(5000, random_state=42)
rng = np.random.default_rng(0)
xj = samp.n_tokens + rng.uniform(-0.28, 0.28, len(samp))     # integer jitter so points separate
yj = samp.n_chars + rng.uniform(-1.6, 1.6, len(samp))
fig, ax = plt.subplots(figsize=(9, 5.6))
for c in CATS:
    m = (samp.category == c).values
    ax.scatter(xj[m], yj[m], s=12, alpha=0.45, color=PAL[c], label=c, linewidths=0)
ax.set_xlabel("Tokens per title (jittered)")
ax.set_ylabel("Characters per title (jittered)")
ax.set_title("Title length in tokens vs characters (random sample of 5,000)")
ax.legend(fontsize=10, ncol=2, frameon=True, loc="lower right")
save(fig, "fig08_scatter_tokens_chars", "Scatter of tokens vs characters")
r_all = stats.pearsonr(df.n_tokens, df.n_chars)[0]
by_k = df.groupby("n_tokens").n_chars
sd = by_k.std()[by_k.size() >= 100]                          # skip token counts too rare for a stable std
r_cat = df.groupby("category").apply(lambda g: stats.pearsonr(g.n_tokens, g.n_chars)[0], include_groups=False)
note("fig08", "Scatter of tokens vs characters",
     f"Characters against tokens per title, coloured by category (scatter plot, jittered, random sample of "
     f"{len(samp):,} of {N} listings)", [
    f"Token count and character length are strongly correlated (Pearson r = {r_all:.2f} over all {len(df):,} "
    f"listings; per category between {r_cat.min():.2f} and {r_cat.max():.2f}). r stays below 1 because words differ in "
    f"length, and n_chars also counts numbers and codes that n_tokens leaves out.",
    f"Spread of title length at a fixed token count (std, token counts with >= 100 listings): "
    f"{sd.iloc[0]:.1f} characters at {sd.index[0]} token{'s' if sd.index[0] != 1 else ''}, rising to {sd.max():.1f} at {sd.idxmax()} tokens, "
    f"then {sd.iloc[-1]:.1f} at {sd.index[-1]} tokens.",
])

# fig09 Zipf
# tokens are deduplicated per listing in p01, so a count is the number of listings that contain the token
freq = Counter(w for t in df.tokens for w in t.split())
ranks = np.arange(1, len(freq) + 1)
f = np.array([c for _, c in freq.most_common()])
# A straight line does not hold over all ranks: the top ~100 tokens are flatter, and the tokens seen in one
# listing form a flat step at the end. The fit therefore uses ranks 100 .. last token seen in >= 2 listings.
lo, hi = 100, int((f >= 2).sum())
lx, ly = np.log10(ranks[lo - 1:hi]), np.log10(f[lo - 1:hi])
slope, intercept = np.polyfit(lx, ly, 1)
r2 = 1 - ((ly - (slope * lx + intercept)) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
head_slope = np.polyfit(np.log10(ranks[:lo]), np.log10(f[:lo]), 1)[0]
pred1 = 10 ** intercept                                     # the fitted line extended back to rank 1
fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(ranks, f, color="#4c72b0", lw=1.4, label="observed token frequency")
ax.loglog(ranks[lo - 1:hi], 10 ** (slope * lx + intercept), color="#c44e52", ls="--", lw=1.6,
          label=f"fit on ranks {lo}-{hi:,}: slope = {slope:.2f}, $R^2$ = {r2:.3f}")
ax.loglog(ranks[:lo], 10 ** (slope * np.log10(ranks[:lo]) + intercept), color="#c44e52", ls=":", lw=1.2,
          label="same line extended to rank 1")
ax.set_xlabel("Token rank (log scale)")
ax.set_ylabel("Frequency (log scale)")
ax.set_title(f"Zipf plot of the vocabulary ({len(freq):,} distinct tokens)")
ax.legend(frameon=False)
save(fig, "fig09_zipf_tokens", "Zipf plot of token frequency")
cumf = np.cumsum(f)
half_rank = int(np.searchsorted(cumf, 0.5 * cumf[-1]) + 1)
hapax = int((f == 1).sum())
rare = int((f < 5).sum())
note("fig09", "Zipf plot of token frequency",
     f"Listings containing each token against its frequency rank (Zipf plot, log-log, {len(freq):,} distinct tokens)", [
    f"Between ranks {lo} and {hi:,} (every token seen in at least 2 listings) the log-log fit has slope "
    f"{slope:.2f} and R^2 = {r2:.3f}: frequency falls roughly as rank^{slope:.2f}.",
    f"The top {lo} ranks are flatter (slope {head_slope:.2f} on their own). Extended to rank 1, the line predicts "
    f"{pred1 / f[0]:.0f} times as many listings as the {f[0]:,} the top token actually has.",
    f"Just {half_rank:,} of {len(freq):,} token types ({pct(half_rank / len(freq))}) account for half of all "
    f"token occurrences.",
    f"{hapax:,} token types ({pct(hapax / len(freq))}) occur in exactly one listing and {rare:,} "
    f"({pct(rare / len(freq))}) in fewer than 5. The decision tree in p04 drops words seen in fewer than "
    f"5 training titles (min_df=5).",
])

# fig10 stacked area merchants
mc = pd.crosstab(df.merchant_id, df.category)
mc = mc.loc[mc.sum(axis=1).sort_values(ascending=False).index]
cum = mc.cumsum(axis=0)
xr = np.arange(1, len(cum) + 1)
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.stackplot(xr, [cum[c].values for c in CATS], labels=CATS, colors=[PAL[c] for c in CATS], alpha=0.92)
ax.set_xlabel("Merchant rank (sorted by total listings, 1 = largest)")
ax.set_ylabel("Cumulative listings")
ax.set_title(f"Category mix accumulated across {len(mc):,} merchants")
ax.legend(loc="upper left", fontsize=10, bbox_to_anchor=(1.01, 1.0), frameon=True)
ax.margins(x=0)
save(fig, "fig10_stacked_area_merchants", "Stacked area of merchants by size")
msize = mc.sum(axis=1)
top10 = msize.head(10).sum() / len(df)
n50 = int(np.searchsorted(msize.cumsum().values, 0.5 * len(df)) + 1)
top5share = mc.T.apply(lambda r: r.sort_values(ascending=False).head(5).sum() / r.sum(), axis=1)
note("fig10", "Stacked area of merchants by size",
     f"Cumulative listings per category over merchants ranked by size (stacked area chart, {len(mc):,} merchants, "
     f"n = {N})", [
    f"Listings are concentrated among merchants: the 10 largest of {len(mc):,} merchants hold {pct(top10)} "
    f"of the corpus, and {n50} merchants are needed to reach 50%.",
    f"The merchant-size Gini coefficient is {gini(msize):.2f} (0 = equal, 1 = one merchant holds everything), "
    f"with a median of {msize.median():.0f} listings per merchant versus {msize.max():,} for the largest.",
    f"Within a category the top 5 merchants supply between {pct(top5share.min())} ({top5share.idxmin()}) and "
    f"{pct(top5share.max())} ({top5share.idxmax()}) of listings.",
])

# fig11 radar
g = df.groupby("category")
rad = pd.DataFrame({
    "mean n_tokens": g.n_tokens.mean(),
    "mean n_chars": g.n_chars.mean(),
    "mean n_codes": g.n_codes.mean(),
    "distinct brands": g.brand.nunique(),
    "distinct merchants": g.merchant_id.nunique(),
    "listings": g.size(),
}).loc[CATS]
radn = (rad - rad.min()) / (rad.max() - rad.min())        # min-max per axis across categories
angles = np.linspace(0, 2 * np.pi, rad.shape[1], endpoint=False)
# one small radar per category: ten overlaid polygons were unreadable
short_ax = ["tokens", "chars", "codes", "brands", "merchants", "listings"]
fig, axs = plt.subplots(2, 5, figsize=(21, 9), subplot_kw=dict(polar=True))
for ax, c in zip(axs.flat, CATS):
    v = radn.loc[c].values
    ax.plot(np.r_[angles, angles[0]], np.r_[v, v[0]], lw=1.8, color=PAL[c])
    ax.fill(np.r_[angles, angles[0]], np.r_[v, v[0]], color=PAL[c], alpha=0.25)
    ax.set_xticks(angles); ax.set_xticklabels(short_ax, fontsize=14)
    ax.set_ylim(0, 1); ax.set_yticks([0.5, 1]); ax.set_yticklabels([], fontsize=10)
    ax.set_title(c, fontsize=17, fontweight="bold", pad=34)   # clear of the codes/chars labels
fig.subplots_adjust(wspace=0.85, hspace=0.55)
save(fig, "fig11_radar_category_profile", "Radar profile per category", shrink=1.0, layout=False)
note("fig11", "Radar profile per category",
     f"Six indicators per category, each min-max scaled across categories (radar charts, {len(CATS)} categories)", [
    f"{rad['mean n_codes'].idxmax()} titles carry the most SKU/model codes (mean {rad['mean n_codes'].max():.2f} per "
    f"title) and {rad['mean n_codes'].idxmin()} the fewest ({rad['mean n_codes'].min():.2f}).",
    f"Merchant reach is widest for {rad['distinct merchants'].idxmax()} "
    f"({rad['distinct merchants'].max():,} merchants) and narrowest for {rad['distinct merchants'].idxmin()} "
    f"({rad['distinct merchants'].min():,}).",
    f"{rad['distinct brands'].idxmax()} carries the most distinct first-word brand tokens ({rad['distinct brands'].max():,}) from "
    f"{rad['listings'][rad['distinct brands'].idxmax()]:,} listings, while the largest category "
    f"{rad['listings'].idxmax()} has {rad['distinct brands'][rad['listings'].idxmax()]:,} brands from "
    f"{rad['listings'].max():,} listings.",
])

# fig12 quantile plot (Han)
xs = np.sort(df.n_chars.values)
n = len(xs)
fq = (np.arange(1, n + 1) - 0.5) / n                       # f = (i - 0.5) / n
fig, ax = plt.subplots(figsize=(8.6, 5))
ax.plot(fq, xs, color="#4c72b0", lw=1.8, label="empirical quantile curve")
for q, lab in [(0.25, "Q1"), (0.5, "median"), (0.75, "Q3")]:
    xv = np.quantile(xs, q)
    ax.axhline(xv, color="grey", ls=":", lw=1)
    ax.axvline(q, color="grey", ls=":", lw=1)
    ax.annotate(f"{lab} = {xv:.0f}", (q, xv), xytext=(-8, 12), textcoords="offset points", ha="right", fontsize=11, color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))   # up-left of the point, where the curve is lower
ax.set_xlabel("f = (i - 0.5) / n")
ax.set_ylabel("Sorted title length (characters)")
ax.set_title(f"Quantile plot of title length (n = {n:,})")
ax.set_xlim(0, 1)
save(fig, "fig12_quantile_nchars", "Quantile plot of title length")
p95 = np.quantile(xs, 0.95)
qs = np.quantile(xs, [0.25, 0.5, 0.75])
lstep, ustep = qs[1] - qs[0], qs[2] - qs[1]
note("fig12", "Quantile plot of title length",
     f"Sorted title length against f = (i - 0.5)/n, with Q1, median and Q3 marked (quantile plot, n = {N})", [
    f"Q1 = {qs[0]:.0f}, median = {qs[1]:.0f}, Q3 = {qs[2]:.0f}. "
    + (f"The median is exactly halfway between Q1 and Q3 ({lstep:.0f} characters on each side)."
       if lstep == ustep else
       f"The step from Q1 to the median is {lstep:.0f} characters and from the median to Q3 {ustep:.0f}."),
    f"The tails reach {np.quantile(xs, 0.01):.0f} characters at the 1st percentile and {np.quantile(xs, 0.99):.0f} "
    f"at the 99th. The top decile stretches from {np.quantile(xs, 0.9):.0f} to {xs.max():.0f} characters, and 95% "
    f"of titles are at most {p95:.0f}.",
])

# fig13 Q-Q Mobile Phones vs TVs
a = np.sort(df.loc[df.category == "Mobile Phones", "n_chars"].values)
b = np.sort(df.loc[df.category == "TVs", "n_chars"].values)
m = min(len(a), len(b))
p = (np.arange(1, m + 1) - 0.5) / m
qa, qb = np.quantile(a, p), np.quantile(b, p)
lo, hi = min(qa.min(), qb.min()), max(qa.max(), qb.max())
iqr_a, iqr_b = np.subtract(*np.quantile(a, [0.75, 0.25])), np.subtract(*np.quantile(b, [0.75, 0.25]))
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(qb, qa, s=10, alpha=0.6, color="#4c72b0", label="quantile pairs")
ax.plot([lo, hi], [lo, hi], color="#c44e52", ls="--", lw=1.3, label="y = x (equal distributions)")
ax.set_xlabel("TVs title length quantiles (characters)")
ax.set_ylabel("Mobile Phones title length quantiles (characters)")
ax.set_title(f"Q-Q plot: Mobile Phones (n = {len(a):,}) vs TVs (n = {len(b):,})")
ax.legend(frameon=False, fontsize=10)
save(fig, "fig13_qq_mobile_vs_tv", "Q-Q plot Mobile Phones vs TVs")
dmed = np.median(b) - np.median(a)
note("fig13", "Q-Q plot Mobile Phones vs TVs",
     f"Title-length quantiles of Mobile Phones (n = {len(a):,}) against TVs (n = {len(b):,}) (Q-Q plot)", [
    f"The median Mobile Phones title has {np.median(a):.0f} characters and the median TVs title "
    f"{np.median(b):.0f}"
    + (f", {abs(dmed):.0f} {'more' if dmed > 0 else 'fewer'}." if dmed else "; the medians are equal."),
    f"By IQR the phone distribution is {'wider' if iqr_a > iqr_b else 'narrower' if iqr_a < iqr_b else 'as wide'} "
    f"({iqr_a:.0f} vs {iqr_b:.0f} characters; standard deviations {a.std():.1f} vs {b.std():.1f}).",
    f"{100 * (qa < qb).mean():.2f}% of the quantile pairs fall below the y = x line"
    + (", so phone titles are shorter than TV titles at almost every quantile" if (qa < qb).mean() > 0.95 else "")
    + f" (99th percentile {np.quantile(a, 0.99):.0f} vs {np.quantile(b, 0.99):.0f}).",
])

# fig14 top-20 tokens
tf = freq                                                    # same per-listing token counts as fig09
top20 = pd.Series(dict(tf.most_common(20))).sort_values()
fig, ax = plt.subplots(figsize=(8.4, 6))
bars = ax.barh(top20.index, top20.values, color=sns.color_palette("viridis", len(top20)), edgecolor="black",
               linewidth=0.4)
ax.bar_label(bars, labels=[f"{v:,}" for v in top20.values], padding=2, fontsize=10)
ax.set_xlabel("Listings containing the token")
ax.set_ylabel("Token")
ax.set_title("20 most frequent cleaned tokens")
ax.margins(x=0.12)
save(fig, "fig14_top_tokens", "Top-20 tokens", shrink=1.0)
cat_words = {w.rstrip("s") for c in CATS for w in c.lower().split()} | {w for c in CATS for w in c.lower().split()}
top5_occ = sum(v for _, v in tf.most_common(5)) / sum(tf.values())
in_cat = [w for w in top20.index[::-1] if w in cat_words]
in_brand = [w for w in top20.index[::-1] if w in set(bc.head(25).index) and w not in cat_words]
others = [w for w in top20.index[::-1] if w not in in_cat and w not in in_brand]
note("fig14", "Top-20 tokens",
     f"Listings containing each of the 20 most frequent tokens (horizontal bar chart, n = {N})", [
    f"'{top20.index[-1]}' is the most frequent token, present in {top20.iloc[-1]:,} listings "
    f"({pct(top20.iloc[-1] / len(df))})"
    + ("; it is a word from a category name, not a brand." if top20.index[-1] in cat_words else "."),
    f"The five most frequent tokens account for {pct(top5_occ)} of all token occurrences, and at least one of them "
    f"appears in {pct(df.tokens.str.split().apply(lambda s: any(t in s for t in top20.index[-5:])).mean())} of "
    f"listings.",
    f"{len(in_cat)} of the 20 are words from category names ({', '.join(in_cat)}), {len(in_brand)} are among the "
    f"25 largest brands ({', '.join(in_brand)}), and the other {len(others)} are descriptive words such as "
    f"{', '.join(others[:4])}.",
])

# fig15 cluster size histogram
cs = df.cluster_id.value_counts()
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.hist(cs.values, bins=np.arange(0.5, cs.max() + 1.5, 1), color="#55a868", edgecolor="white")
ax.set_yscale("log")
ax.set_xlabel("Listings per cluster_id")
ax.set_ylabel("Number of clusters (log scale)")
ax.set_title(f"Cluster-size distribution ({len(cs):,} clusters)")
save(fig, "fig15_cluster_sizes", "Histogram of listings per cluster")
sing = int((cs == 1).sum())
note("fig15", "Histogram of listings per cluster",
     f"Number of clusters by listings per cluster_id (histogram, log y-axis, {len(cs):,} clusters)", [
    f"Clusters are tiny: {sing:,} of {len(cs):,} ({pct(sing / len(cs))}) contain a single listing and the "
    f"largest holds only {cs.max()} listings.",
    f"The median cluster has {cs.median():.0f} listings while the mean is {cs.mean():.2f}, and the size Gini is "
    f"{gini(cs):.2f}.",
    f"{pct(cs.head(10).sum() / len(df))} of listings sit in the 10 largest clusters, so cluster_id is a fine-grained "
    f"near-identifier rather than a coarse grouping.",
])

# fig16 KDE n_tokens
fig, ax = plt.subplots(figsize=(9.5, 5.2))
sns.kdeplot(data=df, x="n_tokens", hue="category", palette=PAL, common_norm=False,
            clip=(0.5, 19.5), linewidth=1.5, ax=ax, legend=True)
ax.set_xlabel("Tokens per title")
ax.set_ylabel("Density (per category, separately normalised)")
ax.set_title("Token-count density per category")
sns.move_legend(ax, "upper right", fontsize=10, ncol=2, frameon=True, title=None)
save(fig, "fig16_kde_ntokens", "KDE of tokens per category")
modes = pd.Series({c: df.loc[df.category == c, "n_tokens"].value_counts().idxmax() for c in CATS})
top_mode = modes.mode().iloc[0]
tm = g.n_tokens.mean()
mid = tm[(tm - df.n_tokens.mean()).abs() <= 1]               # categories whose mean is within 1 token of the overall mean
note("fig16", "KDE of tokens per category",
     f"Density of tokens per title for each category, normalised per category (kernel density plot, n = {N})", [
    f"Category means run from {tm.min():.2f} tokens ({tm.idxmin()}) to {tm.max():.2f} ({tm.idxmax()}); the gap of "
    f"{tm.max() - tm.min():.2f} tokens is {(tm.max() - tm.min()) / df.n_tokens.std():.1f} times the overall std "
    f"({df.n_tokens.std():.2f}).",
    f"The peaks span {modes.min()}-{modes.max()} tokens; {(modes == top_mode).sum()} of {len(modes)} categories peak at "
    f"{top_mode}, and {len(mid)} categories have a mean within 1 token of the overall mean ({df.n_tokens.mean():.2f}). "
    f"Their curves overlap, so token count alone cannot tell those {len(mid)} apart.",
])

# markdown
lines = [
    "# Plot inferences - PriceRunner listings (UCI #837)",
    "",
    f"Generated by `p02_plots.py` from `out/clean.csv`: {len(df):,} listings, {len(CATS)} categories, "
    f"{df.merchant_id.nunique():,} merchants, {df.brand.nunique():,} brands. "
    f"Figures in `out/figs/`.",
    "",
]
for fid, title, caption, bullets in INF:
    lines.append(f"## {fid} - {title}")
    lines.append(f"Plotted: {caption}.")
    lines.extend(f"- {b}" for b in bullets)
    lines.append("")
MD.write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved {len(INF)} figures to {FIG} and inferences to {MD}")
