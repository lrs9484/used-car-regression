import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

PREP_COST = 1500

sales = pd.read_csv("used_car_sales.csv")
lots = pd.read_csv("auction_lots.csv")
train, test = train_test_split(sales, test_size=0.30, random_state=42)

formula = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident, Treatment(reference='no'))"
full = smf.ols(formula + " + Num_Owners", data=train).fit()
v1 = smf.ols(formula, data=train).fit()
baseline_price = train["Sale_Price"].mean()

print(f"Historical sales: {len(sales)} | Train: {len(train)} | Test: {len(test)} | Auction lots: {len(lots)}")
for name, model in [("All features", full), ("v1", v1)]:
    print(f"\n{name} training adjusted R²: {model.rsquared_adj:.6f}")
    print(pd.DataFrame({"Coefficient": model.params, "p-value": model.pvalues}).to_string())


def choose_buy_or_pass(car_features, auction_price, *, use_baseline=False):
    predicted_price = baseline_price
    if not use_baseline:
        predicted_price = v1.predict(pd.DataFrame([car_features])).iloc[0]
    return "Buy" if auction_price + PREP_COST < predicted_price else "Pass"


test_predictions = {
    "Baseline": np.full(len(test), baseline_price),
    "v1": v1.predict(test),
}

print(f"\nBaseline prediction: ${baseline_price:,.2f}")
for label, predicted in test_predictions.items():
    actual = test["Sale_Price"]
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    print(f"{label}: test R² = {r2_score(actual, predicted):.4f}, RMSE = ${rmse:,.2f}")

lots["Profit_if_bought"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP_COST
feature_names = ["Mileage_K", "Age_Years", "Horsepower", "Accident"]
profits = {}

for label in test_predictions:
    lots[label] = lots.apply(
        lambda lot: choose_buy_or_pass(
            lot[feature_names].to_dict(), lot["Auction_Price"], use_baseline=(label == "Baseline")
        ),
        axis=1,
    )
    bought = lots.loc[lots[label].eq("Buy"), "Profit_if_bought"]
    profits[label] = bought.sum()
    print(f"{label}: bought {len(bought)}, losing buys {(bought < 0).sum()}, profit ${bought.sum():,.0f}")

print(f"Profit improvement: ${profits['v1'] - profits['Baseline']:,.0f}")
lots.to_csv("auction_results.csv", index=False)

actual = test["Sale_Price"]
low = min(actual.min(), *(min(pred) for pred in test_predictions.values()))
high = max(actual.max(), *(max(pred) for pred in test_predictions.values()))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
for ax, (label, predicted) in zip(axes, test_predictions.items()):
    ax.plot([low, high], [low, high], "--", color="gray", linewidth=1)
    ax.scatter(actual, predicted, alpha=0.65, color="#214D73")
    ax.set(title=label, xlabel="Actual retail sale price", aspect="equal")
    ax.xaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))

axes[0].set_ylabel("Predicted retail sale price")
fig.tight_layout()
fig.savefig("test_predictions.png", dpi=220, bbox_inches="tight")
