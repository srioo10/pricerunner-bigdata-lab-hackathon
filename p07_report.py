"""
Step 7 - Build the PDF report (report/report.pdf) from the generated outputs.
The result sections are converted from the .md/.csv files written by p01-p06, and the dataset counts in the
first section are computed from the raw and cleaned CSVs. Two parts are hand-written: the short method text in
this script and exam/conclusions.md. For conclusions.md, check_traceability() lists every number that does not
appear in any output file, so a stale number is visible before submission.
Run p01..p06 first, then: python p07_report.py
"""
import re, subprocess, textwrap
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"; REP = ROOT / "report"; REP.mkdir(exist_ok=True)
# dataset: data/ inside the repo (fetch_data.sh puts it there), else the course folder layout ../data/uci837/
RAW = next((p for p in (ROOT / "data/pricerunner_aggregate.csv", ROOT.parent / "data/uci837/pricerunner_aggregate.csv")
            if p.exists()), ROOT / "data/pricerunner_aggregate.csv")
CONCLUSIONS = ROOT / "conclusions.md"       # hand-written source, not a script output

def esc(s):
    """Escape LaTeX special characters and convert simple markdown (bold, code, arrows, ^superscripts)."""
    s = s.replace("&#124;", "|")
    sups = []                                     # x^2, rank^-1.49, 10^3 -> math superscripts (protected from escaping)
    def _sup(m):
        sups.append(m.group(1)); return f"\x00SUP{len(sups) - 1}\x00"
    s = re.sub(r"(?<![A-Za-z])R\^2(?![\d.])", "\x00RSQ\x00", s)
    s = re.sub(r"\^([+\-]?\d+(?:\.\d+)?)", _sup, s)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_\allowbreak{}"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    s = s.replace("\\_\\allowbreak\\{\\}", "\\_\\allowbreak{}")   # undo the brace escaping applied to our own \allowbreak{}
    s = s.replace("->", r"$\rightarrow$").replace("+-", r"$\pm$").replace(">=", r"$\geq$").replace("<=", r"$\leq$")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s)
    s = s.replace("\x00RSQ\x00", "$R^2$")
    s = re.sub(r"\x00SUP(\d+)\x00", lambda m: "$^{" + sups[int(m.group(1))] + "}$", s)
    return s.replace("\\textbackslash{}rightarrow", "\\rightarrow")

def short_heading(t):
    """Long sentence-like headings -> (short heading, full text or None). Leading 'A. '/'B. ' letters are dropped."""
    t = re.sub(r"^[A-Z]\.\s+", "", t.strip())
    if len(t) <= 60: return t, None
    seps = (" -> ", " (", ":", ";", ", ")
    cut = lambda x: min([k for k in (x.find(sep) for sep in seps) if k > 0] or [len(x)])
    head = t[:cut(t)].rstrip(" ,.:;")
    if len(head) < 12 and ":" in t:                  # 'Extra: K-Means ...' -> use the text after the colon
        rest = t.split(":", 1)[1].strip(); head = rest[:cut(rest)].rstrip(" ,.:;") or head
    return (head, t) if head != t else (t, None)

def short_header(c):
    """Shorten verbose machine column names (speedup_vs_mlxtend_apriori -> x vs apriori)."""
    return re.sub(r"^speedup_vs_(?:mlxtend_|sklearn_)?(\w+?)(?:_dt)?$", lambda m: "x vs " + m.group(1).replace("_", " "), c)

def classification_report_tex(block):
    """Render an sklearn classification_report text block as a booktabs table; None if it does not parse."""
    if not block or not re.search(r"precision\s+recall\s+f1-score\s+support", block[0] + " ".join(block[:2])): return None
    rows, summary = [], []
    for l in block:
        if not l.strip() or "precision" in l: continue
        m = re.match(r"^\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", l)
        a = re.match(r"^\s*(accuracy)\s+([\d.]+)\s+(\d+)\s*$", l)
        if m: (summary if m.group(1) in ("macro avg", "weighted avg", "micro avg", "samples avg") else rows).append(list(m.groups()))
        elif a: summary.append([a.group(1), "", "", a.group(2), a.group(3)])
        else: return None
    if not rows: return None
    fmt = lambda r: " & ".join(esc(c) for c in r) + r" \\"
    return ("{\\scriptsize\\begin{tabular}{lrrrr}\\toprule\nclass & precision & recall & f1-score & support \\\\\\midrule\n"
            + "\n".join(fmt(r) for r in rows) + ("\n\\midrule\n" + "\n".join(fmt(r) for r in summary) if summary else "")
            + "\n\\bottomrule\\end{tabular}}\\par\\medskip")

def md_to_tex(md, base_level=1):
    """Small markdown -> LaTeX converter for the outputs written by the scripts (headings, tables, bullets, code)."""
    lines, out, i = md.splitlines(), [], 0
    secs = ["section", "subsection", "subsubsection", "paragraph"]
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            j = i + 1; block = []
            while j < len(lines) and not lines[j].startswith("```"): block.append(lines[j]); j += 1
            out.append(classification_report_tex(block)
                       or "{\\scriptsize\\begin{verbatim}\n" + "\n".join(block) + "\n\\end{verbatim}}"); i = j + 1; continue
        m = re.match(r"^(#+)\s+(.*)", l)
        if m:
            lvl = min(len(m.group(1)) - 1 + base_level, 3)
            head, full = short_heading(m.group(2))
            out.append(f"\\{secs[lvl]}{{{esc(head)}}}" + (f"\n{esc(full)}\\par" if full else "")); i += 1; continue
        if l.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[\s\-|:]+\|$", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            n = len(rows[0])
            rows = [(r + [""] * n)[:n] for r in rows]
            rows[0] = [short_header(c) for c in rows[0]]
            if n >= 12 and len(rows) <= 10:          # very wide, short table: transpose (columns become rows)
                rows = [list(c) for c in zip(*rows)]; n = len(rows[0])
                body = [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
                out.append("{\\scriptsize\\begin{tabular}{l" + "r" * (n - 1) + "}\\toprule\n" + body[0] + "\\midrule\n"
                           + "\n".join(body[1:]) + "\n\\bottomrule\\end{tabular}}\\par\\medskip"); continue
            wide = max(range(n), key=lambda k: max(len(r[k]) for r in rows))
            spec = "".join("X" if k == wide else "l" for k in range(n))
            body = [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
            size = "\\tiny" if n >= 9 else "\\scriptsize"          # wide tables get a smaller font so they fit
            other = sum(max(len(r[k]) for r in rows) for k in range(n) if k != wide)
            per_line = max(20, 150 - other)                             # rough characters per line left for the wide column
            if sum(1 + len(r[wide]) // per_line for r in rows) > 45:     # too tall for one page: longtable breaks across pages
                spec = "".join(f"p{{\\dimexpr\\linewidth-{0.55 * other:.1f}em-{2 * n}\\tabcolsep\\relax}}" if k == wide else "l"
                               for k in range(n))
                out.append("{" + size + "\\setlength\\LTleft{0pt}\\begin{longtable}{" + spec + "}\\toprule\n" + body[0]
                           + "\\midrule\\endhead\n" + "\n".join(body[1:]) + "\n\\bottomrule\\end{longtable}}\\par\\medskip"); continue
            out.append("{" + size + "\\begin{tabularx}{\\linewidth}{" + spec + "}\\toprule\n" + body[0] + "\\midrule\n"
                       + "\n".join(body[1:]) + "\n\\bottomrule\\end{tabularx}}\\par\\medskip"); continue
        if l.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "): items.append(esc(lines[i][2:])); i += 1
            out.append("\\begin{itemize}\\setlength\\itemsep{0pt}\n" + "\n".join("\\item " + x for x in items) + "\n\\end{itemize}"); continue
        out.append(esc(l) if l.strip() else ""); i += 1
    return "\n".join(out)

def fig(path, caption, width="0.85\\linewidth"):
    return (f"\\begin{{figure}}[htbp]\\centering\\includegraphics[width={width}]{{{path}}}"
            f"\\caption{{{esc(caption)}}}\\end{{figure}}")

def plot_caption(title, body):
    """Caption that says what the plot shows: the first inference, cut at its first clause and shortened."""
    first = next((l[2:] for l in body.splitlines() if l.startswith("- ")), "")
    if not first: return title
    cut = re.split(r";\s| - |, so |, i\.e\. |\.\s", first, maxsplit=1)[0].rstrip(" .")
    if len(cut) > 120: cut = cut[:120].rsplit(" ", 1)[0] + " ..."
    return f"{title}. {cut}."

def plots_section():
    md = (OUT / "plot_inferences.md").read_text()
    blocks = re.split(r"^## ", md, flags=re.M)
    tex = [md_to_tex(blocks[0].split("\n", 1)[1] if "\n" in blocks[0] else "")]
    figs = {p.name[:5]: p for p in sorted((OUT / "figs").glob("fig*.png"))}
    for b in blocks[1:]:
        title, body = b.split("\n", 1) if "\n" in b else (b, "")
        key = title.split()[0][:5]
        name = re.sub(r"^fig\d+\s*-\s*", "", title.strip())
        tex.append(f"\\subsection{{{esc(name)}}}")
        if key in figs: tex.append(fig(figs[key].relative_to(ROOT).as_posix().replace("out/", "../out/"),
                                       plot_caption(name, body), "0.62\\linewidth"))
        tex.append("\\needspace{8\\baselineskip}\\textbf{Key inferences:}\n" + md_to_tex(body.strip()))
    return "\n".join(tex)

REQUIRED = ["clean.csv", "preprocess_log.txt", "stats_transform.md", "plot_inferences.md", "arm/arm_summary.md",
            "model/model_summary.md", "mafia/mafia_summary.md", "mafia/fig_mafia_sweep.png", "spark/results.md"]

def check_outputs():
    """Fail loudly if a step was not run, so the PDF never silently misses a section."""
    missing = [f for f in REQUIRED if not (OUT / f).exists()]
    if missing:
        raise SystemExit(f"missing outputs (run the p0x scripts first): {missing}")
    figs = list((OUT / "figs").glob("fig*.png"))
    assert len(figs) >= 10, f"only {len(figs)} plots found, the exam needs at least 10"
    if not CONCLUSIONS.exists():
        raise SystemExit(f"missing {CONCLUSIONS} (hand-written conclusions)")
    # p02 must have run on the current clean.csv: its header line quotes the row count
    n = sum(1 for _ in open(OUT / "clean.csv")) - 1
    assert f"{n:,}" in (OUT / "plot_inferences.md").read_text(), "plot_inferences.md is stale (row count differs)"

NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?x?(?![\w])")

def _variants(tok):
    """Forms a number may take in the outputs: '1,113' -> '1113'; '94.7%' -> '94.7' or '0.947'; '4.4x' -> '4.4'."""
    v = tok.replace(",", "").rstrip("x")
    if not v.endswith("%"):
        return {v}
    v = v[:-1]
    dec = len(v.split(".")[1]) if "." in v else 0
    return {v, f"{float(v) / 100:.{dec + 2}f}"}

def check_traceability():
    """Warn about numbers in the hand-written conclusions.md that no output file contains. Prints, never fails.
    A number counts as found if some output number equals it or rounds to it (0.9475 in a table backs 94.7%)."""
    found = set()
    for p in OUT.rglob("*"):
        if p.suffix not in (".md", ".csv", ".json", ".txt") or not p.is_file():
            continue
        text = re.sub(r"(?<=\d),(?=\d{3})", "", p.read_text(errors="ignore"))
        for x in set(re.findall(r"\d+(?:\.\d+)?", text)):
            found.add(x)
            if "." in x:
                found.update(f"{float(x):.{d}f}" for d in range(1, len(x.split(".")[1])))
    missing = []
    for tok in dict.fromkeys(NUM.findall(CONCLUSIONS.read_text())):
        if len(tok.replace(",", "").rstrip("%x")) < 3:
            continue                                   # 1-2 character numbers (k=10, 5%) match almost anywhere
        if not _variants(tok) & found:
            missing.append(tok)
    if missing:
        print(f"WARNING: {len(missing)} number(s) in conclusions.md not found in any out/ file: {', '.join(missing)}")
    else:
        print("conclusions.md: every number with 3+ characters appears in an output file")

CODE_LINK = "https://github.com/srioo10/pricerunner-bigdata-lab-hackathon"   # public repo with the code

def implementation_section():
    """Short map of the code (no source listing): which script does what and where its output appears."""
    rows = [("p01\\_preprocess.py", "cleaning, German repair, token rules, IQR outliers, selection, encoding", "2"),
            ("p01b\\_stats\\_transform.py", "descriptive statistics, scaling, binning, outliers, skew transforms", "2.1"),
            ("p02\\_plots.py", "16 plots, each with inferences computed from the data", "3"),
            ("p03\\_apriori.py", "Apriori sweep, chosen rules, class rules, per-category rules, own-Apriori check", "4"),
            ("own\\_apriori.py", "from-scratch Apriori (join + prune) used for that check", "4"),
            ("p04\\_tree\\_kmeans.py", "decision tree (CV), IF-THEN rules, baselines, K-Means, dendrograms", "5, 6"),
            ("p05\\_mafia.py", "MAFIA (FIMI C++ binary) MFIs, maximal rules, fpmax check", "7"),
            ("p06\\_pyspark.py", "PySpark FPGrowth and DecisionTree vs mlxtend/sklearn, 1x-20x", "8"),
            ("p07\\_report.py", "builds this PDF from the outputs of p01-p06", "-")]
    body = "\n".join(f"\\texttt{{{f}}} & {d} & {sec} \\\\" for f, d, sec in rows)
    link = (f"The code is available at \\url{{{CODE_LINK}}}. " if CODE_LINK else "")
    return ("\\section{Implementation}\n" + link +
            "Each script writes its tables and figures to \\texttt{out/}, and this report is built only from those files, "
            "so every number above can be traced to the script that produced it. \\texttt{run\\_all.sh} runs the steps in order.\n\n"
            "\\begin{center}\\small\\begin{tabularx}{\\linewidth}{lXl}\\toprule script & what it does & section \\\\ \\midrule\n"
            + body + "\n\\bottomrule\\end{tabularx}\\end{center}\n")

def main():
    check_outputs()
    check_traceability()
    raw = pd.read_csv(RAW); raw.columns = raw.columns.str.strip()
    clean = pd.read_csv(OUT / "clean.csv")
    n_raw, m_raw, m_clean = len(raw), raw["Merchant ID"].nunique(), clean["merchant_id"].nunique()
    rel = lambda p: "../out/" + p
    parts = []
    parts.append(r"""\section{Dataset and objective}
\textbf{Dataset:} PriceRunner product classification and clustering (UCI ML Repository \#837), file
\texttt{pricerunner\_\allowbreak aggregate.csv}: """ + f"{n_raw:,}".replace(",", "{,}") + f""" product listings collected from {m_raw} online merchants ({m_clean} after cleaning), with
{raw.shape[1]} attributes: Product ID, Product Title, Merchant ID, Cluster ID (the actual product a listing refers to),
Cluster Label, Category ID and Category Label ({raw["Category Label"].str.strip().nunique()} product categories).""" + r"""\\
\textbf{Objective:} descriptive analytics (preprocessing + visualisation) followed by predictive analytics:
Apriori association rules, Decision Tree classification of the category from the title, K-Means clustering
of titles (with dendrograms), MAFIA maximal frequent itemsets, and a PySpark comparison of two techniques.\\
\textbf{Assumed parameters} (the exam allows them to be assumed and stated) are given in each section.
\textbf{Tools:} Python 3.12, pandas, scikit-learn, mlxtend, matplotlib/seaborn, PySpark (local mode), MAFIA 1.4 C++
binary from the FIMI repository. Source files: \texttt{p01\_preprocess.py} \dots\ \texttt{p07\_report.py}.""")
    parts.append("\\section{Data preprocessing}\n" + r"""Steps applied by \texttt{p01\_preprocess.py}:
\begin{itemize}\setlength\itemsep{0pt}
\item \textbf{(i) Cleaning:} header whitespace stripped; titles lower-cased; punctuation noise removed; units joined
('64 gb'$\rightarrow$'64gb'); German listings whose umlauts were lost in the source file repaired and translated
('geschirrsp ler'$\rightarrow$'dishwasher'); stop words, one-character junk, bare numbers and SKU/model codes
(alphanumeric strings of 6+ characters mixing letters and digits, e.g. \texttt{lwf28441w}) removed from the token set;
duplicate listings checked; titles with no informative token dropped; title-length outliers removed with the
1.5$\times$IQR rule.
\item \textbf{(ii) Selection:} Not used as features: Product ID (pure identifier, kept only as a row key), Category ID (1:1 with Category Label),
Cluster Label (the product name of a cluster; its words already appear in the titles); Cluster ID is kept only to build a product-held-out test split.
\textbf{Brand} is not a column of the dataset: it is approximated by the first informative word of the title (a proxy). Title tokens are the features, Category is the class label.
For the decision tree, the number of $\chi^2$-ranked words to keep (250, 500, 1000 or all) is chosen by 5-fold CV on the training rows (Section 5.1).
\item \textbf{(iii) Transformation:} title $\rightarrow$ token set (transactions for Apriori/MAFIA, with an extra item
\texttt{cat=\allowbreak<category>}); binary bag-of-words (decision tree); TF-IDF + truncated SVD (K-Means); label encoding of
the category; min-max scaling of the length attributes; FIMI integer format for MAFIA.
\end{itemize}
Log produced by the script:""")
    log = (OUT / "preprocess_log.txt").read_text()
    # wrap long log lines so the verbatim block stays inside the margins; hide local paths
    log = log.replace(str(ROOT) + "/", "")
    parts.append("{\\scriptsize\\begin{verbatim}\n" + "\n".join(w for l in log.splitlines() for w in (textwrap.wrap(l, 105) or [""])) + "\n\\end{verbatim}}")
    st = OUT / "stats_transform.md"
    if st.exists():
        parts.append(md_to_tex(st.read_text(), 1).replace("\\subsection{Descriptive statistics and data transformation}",
                     "\\subsection{Descriptive statistics and further transformations (p01b\\_stats\\_transform.py)}"))
        parts.append(fig(rel("fig_skew_transform.png"), "Skew reduction of listings per product (log, sqrt, Box-Cox)", "\\linewidth"))
    parts.append("\\section{Descriptive analytics: visualisations}\n" + plots_section())

    arm = (OUT / "arm/arm_summary.md").read_text()
    parts.append(md_to_tex(arm.replace("# Apriori association rules", "# Association rules with Apriori"), 0))

    model = (OUT / "model/model_summary.md").read_text()
    dt, km = model.split("# K-Means clustering", 1)
    parts.append(md_to_tex(dt, 0))
    parts += [fig(rel("model/dt_confusion.png"), "Decision tree: confusion matrix on the test set", "0.62\\linewidth"),
              fig(rel("model/dt_importance.png"), "Decision tree: top-20 feature importances", "0.55\\linewidth"),
              fig(rel("model/dt_tree.png"), "Top two levels of the tuned decision tree (the model evaluated and used for the IF-THEN rules)", "\\linewidth")]
    km_md, dendro_md = ("# K-Means clustering" + km).split("## Dendrogram merges", 1)
    parts.append(md_to_tex(km_md, 0))
    parts += [fig(rel("model/km_elbow_silhouette.png"), "Elbow (SSE) and silhouette for k = 2..24", "0.85\\linewidth"),
              fig(rel("model/km_vs_category.png"), "K-Means (k=10) clusters vs true categories", "0.6\\linewidth"),
              fig(rel("model/km_scatter2d.png"), "K-Means clusters projected to 2-D", "0.5\\linewidth")]
    parts.append(md_to_tex("## Dendrograms (hierarchical clustering)" + dendro_md, 0))
    parts += [fig(rel("model/dendro_categories.png"), "Dendrogram of the 10 category centroids (Ward)", "0.75\\linewidth"),
              fig(rel("model/dendro_sample.png"), "Dendrogram of 60 listings, 6 per category (average link, cosine)", "\\linewidth")]

    parts.append(md_to_tex((OUT / "mafia/mafia_summary.md").read_text().replace("# MAFIA maximal frequent itemsets", "# Maximal frequent itemsets with MAFIA"), 0))
    parts.append(fig(rel("mafia/fig_mafia_sweep.png"), "Run time of Apriori, FP-growth and MAFIA, and FI/CFI/MFI counts, against minimum support", "0.95\\linewidth"))

    sp = OUT / "spark/results.md"
    if sp.exists():
        parts.append(md_to_tex(sp.read_text(), 0))
        for f in sorted((OUT / "spark").glob("fig*.png")):
            cap = {"fig_speedup_arm": "PySpark FPGrowth vs mlxtend Apriori/FP-growth: run time vs data size (1x-20x)",
                   "fig_speedup_dt": "PySpark vs sklearn decision tree: run time vs data size (1x-20x)"}.get(f.stem, f.stem)
            parts.append(fig(rel("spark/" + f.name), cap, "0.7\\linewidth"))
    parts.append(md_to_tex(CONCLUSIONS.read_text(), 0))
    parts.append(implementation_section())

    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}\usepackage{graphicx,booktabs,tabularx,float,xcolor,amsmath,needspace,placeins,longtable}
\usepackage[hidelinks]{hyperref}\usepackage[T1]{fontenc}\usepackage{lmodern}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}\sloppy\setlength{\emergencystretch}{3em}
\renewcommand{\topfraction}{0.85}\renewcommand{\textfraction}{0.1}\renewcommand{\floatpagefraction}{0.75}
\title{Analytics \& Systems of Big Data Practice -- End-semester\\\large Descriptive and predictive analytics on the PriceRunner product dataset}
\author{Sooraj S \quad (CS23I1006)}
\date{25CS5004 / CS5010}
\begin{document}\maketitle
""" + (f"\\begin{{center}}\\small Code: \\url{{{CODE_LINK}}}\\end{{center}}\n" if CODE_LINK else "") + r"""\tableofcontents\newpage
""" + re.sub(r"\\(sub)?section\{", lambda m: "\\FloatBarrier\n" + m.group(0), "\n\n".join(parts)) + "\n\\FloatBarrier\n\\end{document}\n"
    (REP / "report.tex").write_text(tex)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=REP,
                           capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:]); raise SystemExit("pdflatex failed")
    print("built", REP / "report.pdf")

if __name__ == "__main__":
    main()
