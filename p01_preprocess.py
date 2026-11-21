"""
Step 1 - Data preprocessing for the PriceRunner product dataset (UCI #837).
  (i)  Cleaning       : whitespace/case/punctuation noise, duplicates, model-code noise, length outliers
  (ii) Selection      : drop identifier / redundant columns, keep task-relevant ones
  (iii)Transformation : title -> token set (one transaction per listing), label encoding, min-max scaled lengths
Feature-level steps (chi2 word ranking, binary bag-of-words, TF-IDF) happen later in p04, on the training rows only.
Output: exam/out/clean.csv, exam/out/preprocess_log.txt
"""
import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# dataset: data/ inside the repo (fetch_data.sh puts it there), else the course folder layout ../data/uci837/
RAW = next((p for p in (ROOT / "data/pricerunner_aggregate.csv", ROOT.parent / "data/uci837/pricerunner_aggregate.csv")
            if p.exists()), ROOT / "data/pricerunner_aggregate.csv")
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
log = []
def say(*a):
    """Print a line and keep it for preprocess_log.txt (the report quotes this log)."""
    s = " ".join(map(str, a)); print(s); log.append(s)

# small English + listing stop-word list (noise words carrying no product meaning)
STOP = {"in", "with", "and", "the", "for", "a", "of", "by", "to", "on", "&", "-", "+", "x", "w",
        "new", "uk", "eu", "free", "only", "inch", "cm", "litre", "l", "kg"}
MODEL_CODE = re.compile(r"^(?=.*\d)(?=.*[a-z])[a-z0-9]{6,}$")   # e.g. lwf28441w, mq8n2b -> SKU noise
# spec tokens look like codes but carry meaning (spin speed, memory, screen size, CPU socket): keep them
SPEC = re.compile(r"^(\d+(\.\d+)?(rpm|gb|tb|mb|kg|w|l|mp|in|inch|mah|hz|khz|mhz|ghz|mm|cm|p|k)|lga\d+|am\d|ddr\d)$")

def is_code(w):
    """SKU/model code = letters+digits, 6+ chars, unless it is a spec token like 1400rpm or lga1151."""
    return bool(MODEL_CODE.match(w)) and not SPEC.match(w)

# German listings (from .de merchants) lost their umlauts in the source file: 'geschirrspüler' -> 'geschirrsp ler'.
# Repair the split fragments, then translate the frequent German words to English (language normalisation).
UMLAUT_FIX = [(r"geschirrsp ler", "geschirrspuler"), (r"\bk hl", "kuhl"), (r"\bwei\b", "weiss"), (r"\bgr n\b", "grun")]
DE_EN = {"geschirrspuler": "dishwasher", "kuhlschrank": "fridge", "gefrierschrank": "freezer", "waschmaschine": "washing machine",
         "mikrowelle": "microwave", "fernseher": "tv", "weiss": "white", "schwarz": "black", "silber": "silver",
         "edelstahl": "stainless steel", "grau": "grey", "spacegrau": "space grey", "einbau": "built-in",
         "unterbau": "under-counter", "gefrier": "freezer", "kombi": "combi", "handy": "mobile phone", "kamera": "camera"}

def clean_title(t):
    """Normalise one raw title: lower case, German repair/translation, punctuation out, units glued to numbers."""
    t = t.lower().strip()
    for pat, rep in UMLAUT_FIX:
        t = re.sub(pat, rep, t)
    t = re.sub(r"[\"'(),;:!?*#|\[\]{}]", " ", t)       # punctuation noise
    t = t.replace("/", " ")
    t = re.sub(r"(?<=[a-z])-(?=\s|$)", " ", t)          # 'einbau-' / 'kuhl-' -> plain word (hyphenated German prefixes)
    t = " ".join(DE_EN.get(w, w) for w in t.split())     # translate after punctuation is gone, so 'schwarz/silber' is caught
    t = re.sub(r"(?<=\d)\s*(gb|tb|mb)\b", r"\1", t)      # '64 gb' -> '64gb'
    t = re.sub(r"(?<=\d)\s*(kg|l|w|rpm|mp|in)\b", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()

def tokens(t):
    """Informative words of a cleaned title, each once. These become the ARM transaction and the text features."""
    out = []
    for w in t.split():
        if w in STOP or len(w) < 2 or is_code(w) or re.fullmatch(r"[\d.]+", w):
            continue          # drop stop words, 1-char junk, SKU codes, bare numbers
        out.append(w)
    return list(dict.fromkeys(out))   # dedupe, keep order

def main():
    d = pd.read_csv(RAW)
    say("RAW shape:", d.shape)
    d.columns = d.columns.str.strip()                     # header has leading spaces -> noise
    d = d.rename(columns={"Product ID": "product_id", "Product Title": "title", "Merchant ID": "merchant_id",
                          "Cluster ID": "cluster_id", "Cluster Label": "cluster_label",
                          "Category ID": "category_id", "Category Label": "category"})
    say("Missing values per column:", d.isna().sum().to_dict())
    say("dtypes:", d.dtypes.astype(str).to_dict())

    # ---------- (i) CLEANING ----------
    n0 = len(d)
    d = d.dropna()
    d["title"] = d["title"].astype(str)
    d["title_clean"] = d["title"].map(clean_title)
    d = d[d["title_clean"].str.len() > 0]
    exact = d.drop(columns="product_id").duplicated().sum()   # product_id is unique per row, so ignore it
    dup = d.duplicated(subset=["title_clean", "merchant_id", "cluster_id"]).sum()
    d = d.drop_duplicates(subset=["title_clean", "merchant_id", "cluster_id"])
    say(f"Exact duplicate rows: {exact}; duplicate (clean title, merchant, cluster) listings removed: {dup}")
    # cleaning audit: how many raw words each token rule removed
    audit = {"stop word": 0, "1-char junk": 0, "SKU/model code": 0, "bare number": 0, "repeated word": 0, "spec token kept": 0}
    for t in d["title_clean"]:
        seen = set()
        for w in t.split():
            if w in STOP: audit["stop word"] += 1
            elif len(w) < 2: audit["1-char junk"] += 1
            elif is_code(w): audit["SKU/model code"] += 1
            elif re.fullmatch(r"[\d.]+", w): audit["bare number"] += 1
            elif SPEC.match(w) and MODEL_CODE.match(w): audit["spec token kept"] += 1
            elif w in seen: audit["repeated word"] += 1
            seen.add(w)
    n_words = d["title_clean"].str.split().str.len().sum()
    german = d["title"].str.contains(r"geschirrsp ler|\bk hl|\bwei\b", regex=True).sum()
    say(f"Cleaning audit: {n_words} words in cleaned titles; removed as noise -> " +
        ", ".join(f"{k}: {v}" for k, v in audit.items()) + f"; listings with repaired German umlaut fragments: {german}")
    d["tokens"] = d["title_clean"].map(tokens)
    d["n_tokens"] = d["tokens"].str.len()
    d["n_chars"] = d["title_clean"].str.len()            # length of the cleaned title
    d["n_codes"] = d["title_clean"].map(lambda t: sum(is_code(w) for w in t.split()))
    d["brand"] = d["tokens"].str[0]                       # first informative token (SKU codes already removed)
    empty = (d["n_tokens"] == 0).sum()
    d = d[d["n_tokens"] > 0]
    say(f"Titles left with no informative token (pure model code) removed: {empty}")
    # length outliers via IQR on cleaned-title char length (very long titles = keyword-stuffed listings)
    q1, q3 = d["n_chars"].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_mask = (d["n_chars"] < lo) | (d["n_chars"] > hi)
    say(f"Char-length IQR fence [{lo:.1f}, {hi:.1f}] -> {out_mask.sum()} outlier titles removed")
    d = d[~out_mask]
    say(f"Rows: {n0} -> {len(d)} after cleaning ({n0-len(d)} removed, {100*(n0-len(d))/n0:.2f}%)")

    # ---------- (ii) SELECTION ----------
    one_to_one = (d.groupby("category_id")["category"].nunique().max() == 1 and
                  d.groupby("category")["category_id"].nunique().max() == 1)
    say("category_id <-> category is 1:1 (redundant):", one_to_one)
    say("product_id unique per row (identifier, no predictive value):", d["product_id"].is_unique)
    say(f"Distinct after cleaning: merchants={d.merchant_id.nunique()}, clusters={d.cluster_id.nunique()}, categories={d.category.nunique()}")
    say("Selected for analytics: title_clean/tokens (features), category (class label), "
        "merchant_id & cluster_id (descriptive only), derived n_tokens/n_chars/brand.")
    say("Not used as features: product_id (identifier, kept in clean.csv only as a row key), category_id (duplicate of category), cluster_label (the product name of a cluster; its words are already in the titles, and cluster_id is used only to group listings for the product-held-out split).")

    # ---------- (iii) TRANSFORMATION ----------
    cats = sorted(d["category"].unique())
    d["category_code"] = d["category"].map({c: i for i, c in enumerate(cats)})        # label encoding
    for c in ["n_tokens", "n_chars"]:                                                   # min-max scaled copies (stored for reference; models use the token features)
        d[c + "_mm"] = (d[c] - d[c].min()) / (d[c].max() - d[c].min())
    d["tokens"] = d["tokens"].map(" ".join)                                            # space-separated transaction
    keep = ["product_id", "title", "title_clean", "tokens", "merchant_id", "cluster_id", "category", "category_code",
            "brand", "n_tokens", "n_chars", "n_codes", "n_tokens_mm", "n_chars_mm"]
    d[keep].to_csv(OUT / "clean.csv", index=False)
    say("Label encoding:", {c: i for i, c in enumerate(cats)})
    say("Saved", OUT / "clean.csv", d[keep].shape)
    vocab = pd.Series(" ".join(d["tokens"]).split()).value_counts()
    say(f"Vocabulary after cleaning: {len(vocab)} tokens; top 15:", vocab.head(15).to_dict())
    (OUT / "preprocess_log.txt").write_text("\n".join(log))

if __name__ == "__main__":
    main()
