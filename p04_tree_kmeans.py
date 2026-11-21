"""
Step 4 - Predictive analytics: Decision Tree classification + K-Means clustering (+ dendrograms).
Task: predict the product category (10 classes) from the cleaned title; cluster titles without labels.

Decision tree
  features : binary bag-of-words of cleaned tokens (min_df=5), chi2 SelectKBest, k in {250, 500, 1000, all} chosen by 5-fold CV
  split    : 80/20 stratified (seed 42)  AND  a grouped split by cluster_id (the same product never in
             both train and test) so the score is not inflated by near-identical listings of one product
  tuning   : criterion {gini, entropy} x max_depth {5,10,20,40,None} x min_samples_leaf {1,5}
  outputs  : accuracy / precision / recall / F1, confusion matrix, feature importances, IF-THEN rules
K-Means
  features : TF-IDF -> TruncatedSVD(100) (LSA) -> L2 normalise (cosine geometry)
  k choice : elbow (SSE) + silhouette for k=2..24; final k=10 compared with true categories (purity, ARI, NMI)
  dendrograms: agglomerative: Ward on the 10 category centroids; average link + cosine on a 60-listing sample
Outputs: out/model/*.png, out/model/model_summary.md
"""
import re
import time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17, "xtick.labelsize": 15,
                     "ytick.labelsize": 15, "legend.fontsize": 14})   # figures are shrunk to page width, so start large
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import squareform
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out/model"; OUT.mkdir(parents=True, exist_ok=True)
md = []
def say(s=""):
    print(s); md.append(s)

def metrics(y, p):
    P, R, F, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return accuracy_score(y, p), P, R, F

def boot_ci(yt, p, B=1000, seed=0):
    """95% bootstrap interval of accuracy and macro F1: resample the test rows with replacement B times."""
    rng = np.random.default_rng(seed); n = len(yt); acc, f1 = [], []
    for _ in range(B):
        i = rng.integers(0, n, n); a, _, _, f = metrics(yt[i], p[i]); acc.append(a); f1.append(f)
    return np.percentile(acc, [2.5, 97.5]), np.percentile(f1, [2.5, 97.5])

def mcnemar(yt, p1, p2):
    """McNemar test (continuity-corrected): do two classifiers err on different test rows more often than chance?"""
    b = int(((p1 == yt) & (p2 != yt)).sum()); c = int(((p1 != yt) & (p2 == yt)).sum())
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return b, c, stats.chi2.sf(stat, 1)

MIN_PURITY = 0.9      # a rule's leaf must be at least this pure
BRANDS = ("intel", "amd", "samsung", "apple", "bosch", "canon", "nikon")
PRODUCT_LINE = {"bosch": "serie"}   # brand -> its product-line word in these titles

def tree_rules(clf, names, classes, max_rules=25, min_samples=100):
    """Walk the fitted tree and turn every leaf into an IF-THEN rule (conditions on word presence)."""
    t = clf.tree_; rules = []
    def walk(node, conds):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            counts = t.value[node][0]; n = int(t.n_node_samples[node])
            k = int(np.argmax(counts)); purity = counts[k] / counts.sum()
            rules.append((n, purity, conds, classes[k], node)); return
        w = names[t.feature[node]]          # binary feature: <=0.5 means word absent
        walk(t.children_left[node], conds + [("-", w)])
        walk(t.children_right[node], conds + [("+", w)])
    walk(0, [])
    rules = [r for r in rules if r[0] >= min_samples and r[1] >= MIN_PURITY]   # frequent, reliable leaves only
    rules.sort(key=lambda r: -r[0])
    return rules[:max_rules]

def main():
    df = pd.read_csv(ROOT / "out/clean.csv")
    classes = sorted(df.category.unique())
    y = df.category.astype(str).to_numpy(dtype=object)

    # ---------------- DECISION TREE ----------------
    say("# Decision Tree classification (target = category)\n")
    idx = np.arange(len(df))
    tr_i, te_i = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)       # hold-out test set
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    gi_tr, gi_te = next(gss.split(idx, y, groups=df.cluster_id))                         # unseen-product test set

    # data selection: how many chi2-ranked words to keep is itself chosen by 5-fold CV on the training rows
    say("## Data selection: how many words to keep\n\nWords are ranked by their chi-square score against the category. "
        "Each vocabulary size is scored by 5-fold CV on the training rows with a Gini tree and macro F1.\n")
    say("| words kept | CV macro F1 (mean +- sd) |\n|---|---|")
    kres = {}
    for k in (250, 500, 1000, "all"):
        pipe = make_pipeline(CountVectorizer(binary=True, min_df=5, token_pattern=r"\S+"), SelectKBest(chi2, k=k),
                             DecisionTreeClassifier(random_state=42))
        sc = cross_val_score(pipe, df.tokens.iloc[tr_i], y[tr_i], scoring="f1_macro",
                             cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1)
        kres[k] = sc.mean(); say(f"| {k} | {sc.mean():.4f} +- {sc.std():.4f} |")
    best_f = max(kres.values())
    K = next(k for k in (250, 500, 1000, "all") if kres[k] >= best_f - 0.002)    # smallest k within 0.002 of the best
    say(f"\nRule: keep the smallest vocabulary whose CV score is within 0.002 of the best. "
        + ("No smaller vocabulary comes that close, so all words are kept and chi-square selection is not applied.\n"
           if K == "all" else f"That gives k = {K}.\n"))

    def featurize(train_rows):
        """Vocabulary + chi2 selection are fitted on TRAINING rows only (no test leakage)."""
        cv = CountVectorizer(binary=True, min_df=5, token_pattern=r"\S+").fit(df.tokens.iloc[train_rows])
        sel = SelectKBest(chi2, k=K).fit(cv.transform(df.tokens.iloc[train_rows]), y[train_rows])
        return cv, sel, sel.transform(cv.transform(df.tokens))

    cv, sel, X = featurize(tr_i)
    names = cv.get_feature_names_out()[sel.get_support()]
    say(f"The vocabulary (words in at least 5 training listings, built on training rows only) has {len(cv.vocabulary_):,} words; "
        + ("all are kept, since CV shows that dropping chi-square-ranked words lowers F1. " if K == 'all'
           else f"the top {K} by chi-square are kept. ")
        + f"The words with the highest chi-square scores are {', '.join(cv.get_feature_names_out()[np.argsort(-np.nan_to_num(sel.scores_))[:12]])}.\n")
    Xtr, Xte, ytr, yte = X[tr_i], X[te_i], y[tr_i], y[te_i]

    # hyper-parameters chosen by 5-fold stratified CV on the training split; the test set is used once at the end
    say("## Hyper-parameter experiments (5-fold CV on the training split, scored by macro F1)\n")
    params = {"criterion": ["gini", "entropy"], "max_depth": [5, 10, 20, 40, None], "min_samples_leaf": [1, 5]}
    gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, scoring="f1_macro",
                      cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1, return_train_score=True).fit(Xtr, ytr)
    g = pd.DataFrame(gs.cv_results_)
    g = g[["param_criterion", "param_max_depth", "param_min_samples_leaf", "mean_train_score", "mean_test_score",
           "std_test_score", "mean_fit_time"]].rename(columns=lambda c: c.replace("param_", ""))
    g.to_csv(OUT / "dt_grid_cv.csv", index=False)
    say("| criterion | max_depth | min_samples_leaf | CV train F1 | CV val F1 (mean +- sd) | fit time (s) |\n|---|---|---|---|---|---|")
    for r in g.itertuples():
        say(f"| {r.criterion} | {r.max_depth} | {r.min_samples_leaf} | {r.mean_train_score:.4f} | "
            f"{r.mean_test_score:.4f} +- {r.std_test_score:.4f} | {r.mean_fit_time:.2f} |")
    CRIT = {"gini": "the Gini criterion", "entropy": "entropy"}
    bc = {c: g[g.criterion == c].mean_test_score.max() for c in ("gini", "entropy")}
    say(f"\nThe best CV macro F1 is {bc['gini']:.4f} with {CRIT['gini']} and {bc['entropy']:.4f} with {CRIT['entropy']}.")
    bp = gs.best_params_
    bestrow = g.loc[g.mean_test_score.idxmax()]
    leaf = f"at least {bp['min_samples_leaf']} listing{'s' if bp['min_samples_leaf'] != 1 else ''} per leaf"
    depth = "no depth limit" if bp["max_depth"] is None else f"maximum depth {bp['max_depth']}"
    say(f"\nCross-validation selects {CRIT[bp['criterion']]}, {depth} and {leaf} (CV macro F1 {gs.best_score_:.4f}). "
        f"Validation F1 rises with depth, from "
        f"{g[(g.max_depth == 5) & (g.criterion == bp['criterion'])].mean_test_score.max():.4f} at depth 5 to {gs.best_score_:.4f} for the selected tree; the selected tree's "
        f"mean F1 is {bestrow.mean_train_score:.3f} on the training folds and {bestrow.mean_test_score:.3f} on the validation folds.\n")
    say(f"The data-selection table above scores the same tree at {kres['all']:.4f} rather than {gs.best_score_:.4f}: there the "
        "vocabulary is rebuilt inside every fold, while the grid search reuses one vocabulary built on all training rows.\n")

    clf = gs.best_estimator_
    pred = clf.predict(Xte)
    say("## Performance of the best tree (stratified split)\n```\n" + classification_report(yte, pred, digits=4) + "```")
    # grouped split: score on unseen products (K and the tree settings were tuned on the stratified training rows)
    _, _, Xg = featurize(gi_tr)                                   # re-fit features on the grouped training rows
    clf_g = DecisionTreeClassifier(random_state=42, **bp).fit(Xg[gi_tr], y[gi_tr])
    acc_g, P_g, R_g, F_g = metrics(y[gi_te], clf_g.predict(Xg[gi_te]))
    acc_s, P_s, R_s, F_s = metrics(yte, pred)
    say(f"\n| split | accuracy | macro precision | macro recall | macro F1 |\n|---|---|---|---|---|\n"
        f"| stratified random | {acc_s:.4f} | {P_s:.4f} | {R_s:.4f} | {F_s:.4f} |\n"
        f"| grouped by product (cluster_id) | {acc_g:.4f} | {P_g:.4f} | {R_g:.4f} | {F_g:.4f} |\n")
    cnt = pd.Series(y[gi_te]).value_counts()
    say(f"Grouped split: {len(gi_tr)} train / {len(gi_te)} test listings, {df.cluster_id.iloc[gi_te].nunique()} test "
        f"products, no product on both sides; every category is in the test set ({cnt.min()}-{cnt.max()} listings each).\n")
    # one grouped split is one draw; repeat it with 5 seeds to see how much the unseen-product score moves
    g_acc = []
    for sd in range(5):
        g_tr, g_te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=sd).split(idx, y, groups=df.cluster_id))
        _, _, Xs = featurize(g_tr)
        g_acc.append(metrics(y[g_te], DecisionTreeClassifier(random_state=42, **bp).fit(Xs[g_tr], y[g_tr]).predict(Xs[g_te]))[0])
    say(f"Repeating the grouped split with 5 seeds gives accuracy {np.mean(g_acc):.4f} +- {np.std(g_acc):.4f} "
        f"(range {min(g_acc):.4f}-{max(g_acc):.4f}). The vocabulary size and tree settings were tuned on the stratified "
        f"training rows, which share products with these test sets, so this score is slightly optimistic; the settings "
        f"picked ({CRIT[bp['criterion']]}, {depth}, {leaf}) are "
        f"close to sklearn's defaults, so the effect should be small.\n")
    (a_lo, a_hi), (f_lo, f_hi) = boot_ci(yte, pred)
    say(f"95% bootstrap interval on the stratified test set (1,000 resamples of the {len(yte)} test rows): accuracy "
        f"{a_lo:.4f}-{a_hi:.4f}, macro F1 {f_lo:.4f}-{f_hi:.4f}.\n")
    # baselines on the same features and split (Naive Bayes from the syllabus) to put the tree's score in context
    nb_rows, nb_pred = [], {}
    for name_, m in (("Multinomial Naive Bayes", MultinomialNB()), ("Bernoulli Naive Bayes", BernoulliNB())):
        m.fit(Xtr, ytr); nb_pred[name_] = m.predict(Xte); a_, P_, R_, F_ = metrics(yte, nb_pred[name_]); nb_rows.append((name_, a_, F_))
    # cost-complexity pruning: alpha by the same 5-fold CV, largest alpha within one standard error of the best (1-SE rule)
    path = DecisionTreeClassifier(random_state=42, **bp).cost_complexity_pruning_path(Xtr, ytr).ccp_alphas[:-1]
    alphas = np.r_[0.0, np.unique(np.quantile(path, [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]))]   # 0 = the unpruned tree
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cvs = [cross_val_score(DecisionTreeClassifier(random_state=42, ccp_alpha=a, **bp), Xtr, ytr, cv=skf, scoring="f1_macro")
           for a in alphas]
    mu = np.array([c.mean() for c in cvs]); se = np.array([c.std() / np.sqrt(len(c)) for c in cvs])
    alpha = alphas[np.where(mu >= mu.max() - se[mu.argmax()])[0].max()]
    pr = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha, **bp).fit(Xtr, ytr)
    a_p, _, _, F_p = metrics(yte, pr.predict(Xte))
    say("The selected tree is compared with a pruned copy (cost-complexity pruning, penalty ccp_alpha chosen by the same "
        "5-fold CV with the one-standard-error rule) and with two Naive Bayes models on the same features.\n")
    say("| model (same features, stratified split) | accuracy | macro F1 |\n|---|---|---|\n"
        f"| Decision tree (selected) | {acc_s:.4f} | {F_s:.4f} |\n" +
        f"| Decision tree, pruned (ccp_alpha {alpha:.2e}, {pr.get_n_leaves()} leaves) | {a_p:.4f} | {F_p:.4f} |\n" +
        "\n".join(f"| {n_} | {a_:.4f} | {F_:.4f} |" for n_, a_, F_ in nb_rows) + "\n")
    say(f"Pruning keeps {pr.get_n_leaves()} of {clf.get_n_leaves()} leaves and changes test accuracy by "
        f"{100 * (a_p - acc_s):+.2f} points. "
        + (f"The unpruned tree is kept as the reported model because it scores {'slightly ' if acc_s - a_p < 0.01 else ''}higher; "
           "the rules below come from it.\n" if a_p < acc_s else
           "The unpruned tree is still kept as the reported model, although the pruned tree scores as well or better; "
           "the rules below come from it.\n"))
    say("McNemar's test compares two classifiers on the listings where exactly one of them is right.\n")
    for n_, p_ in nb_pred.items():
        b, c, pv = mcnemar(yte, pred, p_)
        say(f"McNemar, tree vs {n_}: the tree alone is right on {b} test listings, {n_} alone on {c} (p = {pv:.2g}"
            f"{', so the difference is unlikely to be chance' if pv < 0.05 else ', so the difference could be chance'}).\n")
    d_ = 100 * (round(acc_s, 4) - round(acc_g, 4))
    say(f"The grouped split never shows the classifier another listing of a test product. Accuracy is {d_:.2f} points "
        f"lower on it, so seeing other listings of the same product adds {'under 1 point' if d_ < 1 else f'{d_:.1f} points'} to the stratified score.\n")

    cm = confusion_matrix(yte, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title("Decision tree - confusion matrix (test set)"); fig.savefig(OUT / "dt_confusion.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    off = cm.copy(); np.fill_diagonal(off, 0); i, j = ci, cj = np.unravel_index(off.argmax(), off.shape)
    say(f"Most frequent confusion: {off[i, j]} '{classes[i]}' listings predicted as '{classes[j]}'.\n")
    rec = pd.Series(np.diag(cm) / cm.sum(axis=1), index=classes).sort_values()
    pairs = sorted(((off[a, b], classes[a], classes[b]) for a in range(len(classes)) for b in range(len(classes)) if off[a, b]), reverse=True)[:3]
    say("Confusion-matrix inferences:\n"
        f"- Lowest recall: {rec.index[0]} ({rec.iloc[0]:.3f}) and {rec.index[1]} ({rec.iloc[1]:.3f}); highest: {rec.index[-1]} ({rec.iloc[-1]:.3f}).\n"
        "- Largest off-diagonal cells: " + "; ".join(f"{a} -> {b} ({n})" for n, a, b in pairs) +
        ". The errors sit between appliance types that share words (fridge, freezer, built-in), not between unrelated products.\n")

    imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(7, 6)); imp[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("Decision tree - top 20 feature importances"); ax.set_xlabel("Gini/entropy importance")
    fig.savefig(OUT / "dt_importance.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    say("Top 10 important words: " + ", ".join(f"{k} ({v:.3f})" for k, v in imp.head(10).items()) + "\n")
    brands = [w for w in imp.head(10).index if w in BRANDS]
    say(f"Importance inference: the top 10 words carry {imp.head(10).sum():.1%} of the total importance. Most name a product "
        "type (fridge, tv, camera)" + (f"; {' and '.join(brands)} are brands that stand in for their category" if brands else "")
        + ".\n")

    # IF-THEN rules straight from the TUNED tree that was evaluated above: every root-to-leaf path is a rule
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(clf, max_depth=2, feature_names=list(names), class_names=classes, filled=True, impurity=False, label='root', fontsize=12, ax=ax)
    for t in ax.texts:                                     # the 10 class counts per node overlap; the class line says enough
        t.set_text(re.sub(r"\n(value = )?\[[^\]]*\]", "", t.get_text(), flags=re.S))
    ax.set_title(f"Tuned decision tree ({bp['criterion']}, {clf.get_n_leaves()} leaves, depth {clf.get_depth()}); top 2 levels drawn")
    fig.savefig(OUT / "dt_tree.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    rules = tree_rules(clf, names, classes, max_rules=25, min_samples=100)
    covered = sum(n for n, *_ in rules)
    say(f"## IF-THEN rules from the tuned tree\n\nThe tuned tree has {clf.get_n_leaves()} leaves (depth {clf.get_depth()}); "
        f"each root-to-leaf path is one IF-THEN rule. Listed: the {len(rules)} rules whose leaf holds >= 100 training "
        f"listings with purity >= 0.9. Together they cover {covered} of {len(ytr)} training listings "
        f"({100*covered/len(ytr):.1f}%); the remaining listings fall into many small leaves.\n")
    def rule_text(conds):
        """Full antecedent: words that must be present, then words that must be absent (nothing is cut)."""
        pos = [w for s_, w in conds if s_ == "+"]; neg = [w for s_, w in conds if s_ == "-"]
        txt = "title has {" + ", ".join(pos) + "}" if pos else ""
        if neg:
            txt += (" AND " if pos else "") + "title has none of {" + ", ".join(neg) + "}"
        return txt
    short = [r for r in rules if len(r[2]) <= 8]          # rules short enough to read in full
    say(f"{len(short)} of these {len(rules)} rules have at most 8 conditions and are printed below; the other "
        f"{len(rules) - len(short)} need 9-{max(len(r[2]) for r in rules)} conditions (mostly long 'none of' lists that "
        "exclude other categories) and follow in a second table.\n")
    te_leaf = clf.apply(Xte)                              # which leaf (= which rule) each test listing falls into
    def test_cols(node, c):
        hit = te_leaf == node
        return f"{hit.sum()} | {(yte[hit] == c).mean():.3f}" if hit.any() else "0 | -"
    say("| # | IF | THEN category | training listings | purity | test listings | test accuracy |\n|---|---|---|---|---|---|---|")
    for k, (n, pur, conds, c, node) in enumerate(short, 1):
        say(f"| {k} | {rule_text(conds)} | {c} | {n} | {pur:.3f} | {test_cols(node, c)} |")
    long_ = [r for r in rules if len(r[2]) > 8]
    if long_:
        say(f"\n### The other {len(long_)} rules (9 or more conditions)\n")
        say("| # | IF | THEN category | training listings | purity | test listings | test accuracy |\n|---|---|---|---|---|---|---|")
        for k, (n, pur, conds, c, node) in enumerate(long_, len(short) + 1):
            say(f"| {k} | {rule_text(conds)} | {c} | {n} | {pur:.3f} | {test_cols(node, c)} |")
    # one rule per category, so every class gets an explicit IF-THEN rule: the shortest pure leaf (>= 30 listings)
    min_leaf = 30
    allr = tree_rules(clf, names, classes, max_rules=10**6, min_samples=min_leaf)
    say(f"\n### Shortest rule for each category\n\nOnly leaves with at least {min_leaf} training listings and purity of "
        f"at least {MIN_PURITY} are used.\n")
    say("| category | IF | conditions | training listings | purity |\n|---|---|---|---|---|")
    for c in classes:
        rc = [r for r in allr if r[3] == c]
        if not rc:
            say(f"| {c} | no leaf meets the threshold | - | - | - |"); continue
        n, pur, conds, _, _ = min(rc, key=lambda r: (len(r[2]), -r[0]))
        say(f"| {c} | {rule_text(conds)} | {len(conds)} | {n} | {pur:.3f} |")
    easy = sorted({r[3] for r in short})
    ex = next(([w for s_, w in r[2] if s_ == "+"], [w for s_, w in r[2] if s_ == "-"], r[3]) for r in rules
              if any(s_ == "-" for s_, _ in r[2]))              # first listed rule with a 'none of' part
    say(f"\nReading the rules: for {len(easy)} of the {len(classes)} categories ({', '.join(easy)}) one strong word plus a few "
        "absent words is enough; "
        "the 'none of' parts stop a listing that also mentions another product type from being caught by the wrong rule "
        f"(a title with '{ex[0][0]}' that also has '{ex[1][0]}' does not reach the {ex[2]} rule above).\n")

    # preprocessing ablation: same tuned settings on RAW titles (no cleaning) vs cleaned tokens
    raw_cv = CountVectorizer(binary=True, min_df=5).fit(df.title.iloc[tr_i])           # default tokenizer, raw text
    Xr = raw_cv.transform(df.title)
    raw_tree = DecisionTreeClassifier(random_state=42, **bp).fit(Xr[tr_i], y[tr_i])
    acc_r, _, _, F_r = metrics(y[te_i], raw_tree.predict(Xr[te_i]))
    raw_cv_g = CountVectorizer(binary=True, min_df=5).fit(df.title.iloc[gi_tr]); Xrg = raw_cv_g.transform(df.title)
    acc_rg = accuracy_score(y[gi_te], DecisionTreeClassifier(random_state=42, **bp).fit(Xrg[gi_tr], y[gi_tr]).predict(Xrg[gi_te]))
    say(f"\n### Effect of preprocessing (same split and tree settings)\n\n| input | features | test accuracy | macro F1 | leaves | accuracy, unseen products |\n|---|---|---|---|---|---|\n"
        f"| raw titles (no cleaning) | {Xr.shape[1]} words | {acc_r:.4f} | {F_r:.4f} | {raw_tree.get_n_leaves()} | {acc_rg:.4f} |\n"
        f"| cleaned tokens{'' if K == 'all' else f' + chi-square top-{K}'} | {X.shape[1]} words | {acc_s:.4f} | {F_s:.4f} | {clf.get_n_leaves()} | {acc_g:.4f} |\n")
    dl = raw_tree.get_n_leaves() - clf.get_n_leaves()
    say(f"Cleaning changes accuracy by {100*(round(acc_s,4)-round(acc_r,4)):+.2f} points on the stratified split and "
        f"{100*(round(acc_g,4)-round(acc_rg,4)):+.2f} on unseen products, uses {1 - X.shape[1]/Xr.shape[1]:.0%} fewer words, and gives a tree "
        f"with {abs(dl)} {'fewer' if dl > 0 else 'more'} leaves. The gain from cleaning is a smaller model, not higher accuracy.\n")

    # ---------------- K-MEANS ----------------
    say("\n# K-Means clustering (labels not used for fitting)\n")
    tf = TfidfVectorizer(min_df=3, token_pattern=r"\S+", sublinear_tf=True)
    T = tf.fit_transform(df.tokens)
    svd = TruncatedSVD(100, random_state=42); Z = normalize(svd.fit_transform(T))
    say("LSA (latent semantic analysis) reduces the TF-IDF matrix to a few dimensions with a truncated SVD. SSE is the sum "
        "of squared distances from each listing to its cluster centre; silhouette (-1 to 1) measures how much closer a "
        "listing is to its own cluster than to the next one; purity is the share of listings that belong to their "
        "cluster's majority category; ARI and NMI measure how well clusters match categories (1 = identical, about 0 "
        "for random clusters).\n")
    say(f"The TF-IDF matrix has {T.shape[0]:,} rows (listings) and {T.shape[1]:,} columns (words in at least 3 listings). "
        f"LSA reduces it to {svd.n_components} dimensions, which keep {svd.explained_variance_ratio_.sum():.1%} of the "
        "variance, and each row is then scaled to unit length.\n")
    rng = np.random.RandomState(42); samp = rng.choice(len(Z), 6000, replace=False)
    ks, sse, sil = list(range(2, 25)), [], []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=42).fit(Z)
        sse.append(km.inertia_); sil.append(silhouette_score(Z[samp], km.labels_[samp]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, sse, "o-"); ax[0].set(title="Elbow: SSE vs k", xlabel="k", ylabel="SSE (inertia)")
    ax[1].plot(ks, sil, "o-", color="C1"); ax[1].set(title="Silhouette vs k", xlabel="k", ylabel="silhouette")
    fig.savefig(OUT / "km_elbow_silhouette.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(dict(k=ks, sse=sse, silhouette=sil)).to_csv(OUT / "km_k_sweep.csv", index=False)
    kbest = ks[int(np.argmax(sil))]
    say("| k | SSE | silhouette |\n|---|---|---|\n" + "\n".join(f"| {k} | {s:.1f} | {si:.4f} |" for k, s, si in zip(ks, sse, sil)))
    s10 = sil[ks.index(10)]
    say(f"\nThe elbow has no sharp bend, and silhouette stays low for every k (range {min(sil):.3f}-{max(sil):.3f}; best k={kbest}"
        f"{', the edge of the grid' if kbest == ks[-1] else ''}). So the title space has no clear natural number of clusters. "
        f"I use k=10. Because no k stands out, the granularity is set by the task: 10 is the number of product "
        f"categories, which lets the clusters be read against them. In this sweep (seed 42 for every k) its silhouette is {s10:.3f}, {s10/max(sil):.0%} of the best value. "
        f"The labels are used only for this granularity and for scoring, never to fit the clusters.\n")

    kmb = KMeans(kbest, n_init=10, random_state=42).fit(Z); lb = kmb.labels_
    ctb = pd.crosstab(lb, y)
    say(f"For reference, k={kbest} (highest silhouette): SSE {kmb.inertia_:.1f}, silhouette {silhouette_score(Z[samp], lb[samp]):.4f}, "
        f"purity {ctb.max(axis=1).sum()/len(df):.4f}, ARI {adjusted_rand_score(y, lb):.4f}, NMI {normalized_mutual_info_score(y, lb):.4f}\n")
    # k-means depends on the start; fit 5 seeds and keep the run with the lowest SSE (a choice that ignores the labels)
    seeds = [42, 0, 1, 2, 3]
    t0 = time.perf_counter(); runs = [KMeans(10, n_init=10, random_state=sd).fit(Z) for sd in seeds]
    tkm = (time.perf_counter() - t0) / len(seeds)
    best = int(np.argmin([r.inertia_ for r in runs])); km = runs[best]; SEED = seeds[best]
    say(f"k=10 is fitted with seeds {', '.join(map(str, seeds))}. The run used below is the one with the lowest SSE (seed {SEED}); "
        "this choice does not look at the categories.\n")
    lab = km.labels_
    ct = pd.crosstab(lab, df.category)
    purity = ct.max(axis=1).sum() / len(df)
    say(f"With k=10 and seed {SEED}, one fit takes {tkm:.2f} s. This run has SSE {km.inertia_:.1f}, silhouette "
        f"{silhouette_score(Z[samp], lab[samp]):.4f} (the seed-42 sweep run scores {s10:.3f}), purity {purity:.4f}, "
        f"ARI {adjusted_rand_score(y, lab):.4f} and NMI {normalized_mutual_info_score(y, lab):.4f}.\n")
    # sensitivity: k-means is seed-dependent and the LSA size is a free choice -> report spread, choose nothing by labels
    say("## Sensitivity of the k=10 comparison (labels used only to score, never to choose)\n")
    rs = [r.labels_ for r in runs]
    nm = [normalized_mutual_info_score(y, l) for l in rs]; ar = [adjusted_rand_score(y, l) for l in rs]
    pu = [pd.crosstab(l, y).max(axis=1).sum() / len(df) for l in rs]; big = [np.bincount(l).max() for l in rs]
    say(f"Over 5 seeds (LSA-100): NMI {np.mean(nm):.3f} +- {np.std(nm):.3f}, ARI {np.mean(ar):.3f} +- {np.std(ar):.3f}, "
        f"purity {np.mean(pu):.3f} +- {np.std(pu):.3f} (NMI range {min(nm):.3f}-{max(nm):.3f}; the lowest-SSE run above is "
        f"{'the lowest' if nm[best] == min(nm) else 'the highest' if nm[best] == max(nm) else 'inside the range'}). "
        f"The largest cluster holds {min(big)}-{max(big)} listings across these seeds.\n")
    say("| LSA dimensions | explained variance | silhouette (k=10) | NMI (k=10) |\n|---|---|---|---|")
    nmi_dim = []
    for dim in (20, 50, 100, 200):
        sv = TruncatedSVD(dim, random_state=42); Zd = normalize(sv.fit_transform(T))
        ld = KMeans(10, n_init=10, random_state=42).fit(Zd).labels_
        nmi_dim.append(normalized_mutual_info_score(y, ld))
        say(f"| {dim} | {sv.explained_variance_ratio_.sum():.3f} | {silhouette_score(Zd[samp], ld[samp]):.4f} | {nmi_dim[-1]:.4f} |")
    say(f"\nThis table and the k sweep above use seed 42 for every run, so their k=10, LSA-100 values differ slightly from the "
        f"seed-{SEED} run reported above. "
        f"Silhouette values from spaces of different dimension are not directly comparable, so this table is not used "
        f"to pick a dimension. NMI moves between {min(nmi_dim):.3f} and {max(nmi_dim):.3f} with the LSA size and between "
        f"{min(nm):.3f} and {max(nm):.3f} with the seed, so a single k=10 run should be read with that spread.\n")
    terms = np.array(tf.get_feature_names_out())
    say("| cluster | size | majority category (share) | top TF-IDF terms |\n|---|---|---|---|")
    for c in range(10):
        mean_tfidf = np.asarray(T[lab == c].mean(0)).ravel()
        top = ", ".join(terms[np.argsort(-mean_tfidf)[:8]])
        say(f"| {c} | {ct.loc[c].sum()} | {ct.loc[c].idxmax()} ({ct.loc[c].max()/ct.loc[c].sum():.2f}) | {top} |")
    share = ct.max(axis=1) / ct.sum(axis=1)
    pure = [f"{ct.loc[c].idxmax()} (C{c}, {share[c]:.2f})" for c in share.index if share[c] >= 0.95]
    mix = share.idxmin(); m2 = ct.loc[mix].sort_values(ascending=False)
    mt = terms[np.argsort(-np.asarray(T[lab == mix].mean(0)).ravel())[:6]]
    if mt[0] in BRANDS:
        line = f" and its product line ({mt[0]}, {mt[1]})" if PRODUCT_LINE.get(mt[0]) == mt[1] else f" ({mt[0]})"
        why = f"Its top terms start with a brand{line}, so k-means grouped these titles by brand rather than by product type."
    else:
        why = f"Its top terms ({', '.join(mt)}) mix several product types, so k-means puts these titles together without a shared product word."
    say(f"\nCluster inferences (k=10, seed {SEED}):\n"
        f"- {len(pure)} clusters are at least 95% one category: {', '.join(pure)}.\n"
        f"- The most mixed cluster is C{mix} ({ct.loc[mix].sum()} listings, largest share {share[mix]:.2f}): "
        + ", ".join(f"{k} {v}" for k, v in m2.head(4).items()) +
        f". {why}\n")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(np.log1p(ct.values), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ct.columns)), ct.columns, rotation=45, ha="right"); ax.set_yticks(range(10), [f"C{c}" for c in range(10)])
    for (a, b), v in np.ndenumerate(ct.values): ax.text(b, a, v, ha="center", va="center", color="w", fontsize=10)
    ax.set_title("K-Means (k=10) clusters vs true categories"); fig.colorbar(im, label="log(1+count)")
    fig.savefig(OUT / "km_vs_category.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    P2 = TruncatedSVD(2, random_state=0).fit_transform(Z[samp])
    fig, ax = plt.subplots(figsize=(7, 6)); sc = ax.scatter(P2[:, 0], P2[:, 1], c=lab[samp], cmap="tab10", s=4, alpha=.6)
    ax.set(title="K-Means clusters in 2-D (SVD projection, 6000 listings)", xlabel="SVD component 1", ylabel="SVD component 2")
    ax.legend(*sc.legend_elements(), title="cluster", loc="upper right", fontsize=9, ncol=2)
    fig.savefig(OUT / "km_scatter2d.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---------------- DENDROGRAMS ----------------
    cent = np.vstack([Z[y == c].mean(0) for c in classes])
    Lc = linkage(cent, method="ward")
    fig, ax = plt.subplots(figsize=(9, 5)); dendrogram(Lc, labels=classes, leaf_rotation=45, ax=ax)
    ax.set_title("Dendrogram of category centroids (Ward, LSA space)"); ax.set_ylabel("merge distance")
    fig.savefig(OUT / "dendro_categories.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    say("\n## Dendrogram merges (category centroids, Ward)\n")
    names_c = list(classes); members = [[c] for c in classes]
    def part(i, last=False):
        """One side of a merge: a category name, its group, or for the final merge 'the other N categories'."""
        m = members[i]
        if len(m) == 1: return m[0]
        return f"the other {len(m)} categories" if last else "{" + ", ".join(m) + "}"
    lines = []
    for k, (a, b, h, _) in enumerate(Lc):
        a, b = int(a), int(b); na, nb = names_c[a], names_c[b]; names_c.append(f"({na} + {nb})")
        if len(members[a]) > len(members[b]): a, b = b, a       # name the smaller side first
        last = k == len(Lc) - 1
        if k < 3 or last:
            if last: lines.append(f"The last merge, at height {h:.3f}, joins {part(a, True)} with {part(b, True)}.")
            elif k == 0: lines.append(f"{part(a)} and {part(b)} merge first, at height {h:.3f}.")
            elif len(members[a]) == 1: lines.append(f"{part(a)} joins {part(b)} at height {h:.3f}.")
            else: lines.append(f"{part(a)} and {part(b)} merge at height {h:.3f}.")
        members.append(members[a] + members[b])
    say(" ".join(lines[:3]) + " The dendrogram figure shows the merges in between. " + lines[-1])
    first = (names_c[int(Lc[0, 0])], names_c[int(Lc[0, 1])])
    g3 = fcluster(Lc, 3, criterion="maxclust")
    last_single = [names_c[int(a)] for a, b, _, _ in Lc[-1:] for a in (a, b) if int(a) < len(classes)]
    say(f"\nDendrogram inferences:\n- The closest categories are {first[0]} and {first[1]} (first merge, height {Lc[0, 2]:.3f}); "
        f""
        + (f"{classes[ci]} joins them at the next merge ({Lc[1, 2]:.3f}), and {classes[ci]} -> {classes[cj]} is also the tree's most frequent error.\n"
           if set(map(int, Lc[1, :2])) == {len(classes), ci} and cj in map(int, Lc[0, :2]) else "\n")   # compare node ids, not names
        + f"- The last merge is at {Lc[-1, 2]:.3f}, {Lc[-1, 2]/Lc[0, 2]:.1f}x the first; "
        + (f"{last_single[0]} joins last, so its vocabulary is the most distinct. " if last_single else "") +
        "Cutting the tree into 3 groups gives: " + "; ".join(
            "{" + ", ".join(np.array(classes)[g3 == k]) + "}" for k in np.unique(g3)) + ".\n")
    idx = np.concatenate([rng.choice(np.where(y == c)[0], 6, replace=False) for c in classes])
    Ls = linkage(Z[idx], method="average", metric="cosine")
    fig, ax = plt.subplots(figsize=(7, 11))   # horizontal, so each of the 60 leaf labels gets its own line
    SHORT = {"CPUs": "CPU", "Digital Cameras": "CAM", "Dishwashers": "DW", "Freezers": "FZ", "Fridge Freezers": "FF",
             "Fridges": "FR", "Microwaves": "MW", "Mobile Phones": "PH", "TVs": "TV", "Washing Machines": "WM"}
    CODE = {c: SHORT.get(c, c[:3].upper()) for c in classes}
    dendrogram(Ls, labels=[CODE[c] for c in y[idx]], orientation="left", leaf_font_size=11, ax=ax)
    ax.set_xlabel("cosine distance")
    ax.set_title("60 listings (6 per category), average link")
    fig.savefig(OUT / "dendro_sample.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    coph = squareform(cophenet(Ls)); np.fill_diagonal(coph, np.inf)
    # nearest leaf in the dendrogram has the same category? ties are common, so average over all tied nearest leaves
    near = coph == coph.min(1, keepdims=True)
    same = np.mean([np.mean(y[idx][near[i]] == y[idx][i]) for i in range(len(idx))])
    say(f"- 60-listing dendrogram (average link, cosine): for {same:.0%} of the listings the closest leaf in the tree "
        f"has the same category, against {5/59:.0%} expected by chance (6 listings per category). On this small sample, "
        "listings group by category without the labels being used.\n")
    (OUT / "model_summary.md").write_text("\n".join(md))

if __name__ == "__main__":
    main()
