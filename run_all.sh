#!/usr/bin/env bash
# Runs the whole analysis end to end and builds report/report.pdf.
# Usage: ./run_all.sh [--skip-spark]
#   --skip-spark  skip p06 (PySpark benchmark, the slow step) and keep the existing out/spark results
# Expects a Python with requirements.txt installed; set PY to override the default venv.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$( [ -x ../.venv/bin/python ] && echo ../.venv/bin/python || echo python3 )}
export PYTHONHASHSEED=0          # fixes set/dict iteration order of strings, so itemset order is the same every run
SKIP_SPARK=0
for a in "$@"; do
  case "$a" in
    --skip-spark) SKIP_SPARK=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done
for s in p01_preprocess p01b_stats_transform p02_plots p03_apriori p04_tree_kmeans p05_mafia p06_pyspark p07_report; do
  if [[ $s == p06_pyspark && $SKIP_SPARK == 1 ]]; then
    [[ -f out/spark/results.md ]] || { echo "--skip-spark needs an existing out/spark/results.md" >&2; exit 1; }
    echo "== $s (skipped, keeping out/spark)"; continue
  fi
  echo "== $s"; "$PY" "$s.py"
done
