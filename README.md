# PriceRunner product data: descriptive and predictive analytics

End-semester lab, Analytics & Systems of Big Data Practice (25CS5004 / CS5010).
Submitted by Sooraj S (CS23I1006).
The report is [`report/cs23i1006_report.pdf`](report/cs23i1006_report.pdf).

## Data
UCI ML Repository #837, "Product Classification and Clustering" (CC BY 4.0):
https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering.
`./fetch_data.sh` downloads `pricerunner_aggregate.csv` into `data/`. The scripts also accept the course-folder
layout `../data/uci837/`.

## Requirements
- Python 3.12 (the pinned versions in `requirements.txt` were tested on 3.12), then `pip install -r requirements.txt`
- Java 17+ for PySpark (local mode)
- `pdflatex` (TeX Live) for the report, with these packages: geometry, graphicx, booktabs, tabularx, float, xcolor,
  amsmath, needspace, placeins, longtable, hyperref, fontenc, lmodern
- `bin/mafia`: MAFIA 1.4 (Burdick, Calimlim, Gehrke), C++ source from the FIMI repository
  (https://fimi.uantwerpen.be/src/), built with `g++ -O3 -o mafia src/*.cpp` and placed in `bin/`

## Run
`./run_all.sh` runs every step in order (set `PY=/path/to/python` if the venv is elsewhere).
`./run_all.sh --skip-spark` skips `p06` and keeps the existing `out/spark/` results.

| script | step | main outputs |
|---|---|---|
| `p01_preprocess.py` | cleaning, selection, transformation | `out/clean.csv`, `out/preprocess_log.txt` |
| `p01b_stats_transform.py` | descriptive statistics, normalisation, binning, outliers, skew, encoding | `out/stats_transform.md` |
| `p02_plots.py` | 16 plots with computed inferences | `out/figs/`, `out/plot_inferences.md` |
| `p03_apriori.py` | Apriori association rules (support/confidence sweep, class rules, per category) | `out/arm/` |
| `p04_tree_kmeans.py` | decision tree (CV-tuned, IF-THEN rules), K-Means, dendrograms | `out/model/` |
| `p05_mafia.py` | MAFIA maximal frequent itemsets and maximal rules | `out/mafia/` |
| `p06_pyspark.py` | PySpark FPGrowth and DecisionTree vs single-machine, 1x-20x data | `out/spark/` |
| `p07_report.py` | builds the PDF from the outputs and `conclusions.md` | `report/cs23i1006_report.pdf` |

`own_apriori.py` is a small from-scratch Apriori, used in `p03` to cross-check the library results.

`conclusions.md` is written by hand, not generated. `p07` includes it in the report and prints a warning listing
any number in it (3 or more characters) that does not appear in an `out/` file, so a stale number shows up after a rerun.

## Reproducibility
All random seeds are fixed in the code and `run_all.sh` sets `PYTHONHASHSEED=0`, so a rerun gives the same counts,
rules, scores and plots. Run times (`p03` to `p06`) vary from run to run and from machine to machine, so the timings
and speed ratios in the report will not match exactly.
