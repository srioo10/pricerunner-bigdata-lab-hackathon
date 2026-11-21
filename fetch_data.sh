#!/usr/bin/env bash
# Downloads the PriceRunner dataset (UCI #837, CC BY 4.0) into data/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
curl -fL -o data/uci837.zip "https://archive.ics.uci.edu/static/public/837/product+classification+and+clustering.zip"
unzip -o -j data/uci837.zip "*pricerunner_aggregate.csv" -d data
rm data/uci837.zip
echo "saved data/pricerunner_aggregate.csv"
