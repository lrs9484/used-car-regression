# Used Car Auction Regression

This project predicts used-car retail sale prices and compares two auction buying rules. It uses a fixed 70/30 split of 240 historical sales and evaluates both rules on a separate set of 40 auction lots.

## How to run the analysis

Download or clone this repository, then open a terminal in the project folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

The script prints the model coefficients, p-values, adjusted R², test-set results, and simulated auction profits. It also creates `test_predictions.png` and `auction_results.csv`.
