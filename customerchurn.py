"""Exploratory analysis, model training and SHAP explanations.

Run:  python customerchurn.py
Plots are shown on screen and also saved to the 'images' folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from churn_model import build_models, load_data, split_data

IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- Load & clean
df = load_data()
print("Rows and columns after cleaning:", df.shape)
print(df.head())
print("\nChurn counts:")
print(df["Churn"].value_counts())

print(
    df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]]
    .mean()
    .round(2)
)

# --------------------------------------------------------------------- EDA
churn_by_contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
print("\nChurn % by contract:")
print(churn_by_contract.round(1))

churn_by_contract.plot(kind="bar", stacked=True, color=["steelblue", "tomato"])
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Customers (%)")
plt.legend(title="Churn")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "churn_by_contract.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df.boxplot(column="tenure", by="Churn", ax=axes[0])
axes[0].set_title("Tenure by Churn")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Months")
df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[1])
axes[1].set_title("Monthly Charges by Churn")
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Amount")
fig.suptitle("")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "tenure_charges_by_churn.png", dpi=150)
plt.show()

# ------------------------------------------------------------------ Modelling
X_train, X_test, y_train, y_test = split_data(df)
models = build_models(X_train)

probabilities = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities[name] = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"\n{name} classification report:")
    print(classification_report(y_test, predictions, target_names=["No churn", "Churn"]))

print("\nProbability-based model comparison:")
for name, probs in probabilities.items():
    print(f"{name} ROC-AUC:", round(roc_auc_score(y_test, probs), 3))
    print(f"{name} average precision:", round(average_precision_score(y_test, probs), 3))

# ----------------------------------------------------------------------- SHAP
xgb_model = models["XGBoost"]
xgb_preprocessor = xgb_model.named_steps["preprocessor"]
X_test_prepared = xgb_preprocessor.transform(X_test)
if hasattr(X_test_prepared, "toarray"):
    X_test_prepared = X_test_prepared.toarray()

X_test_prepared = pd.DataFrame(
    X_test_prepared,
    columns=xgb_preprocessor.get_feature_names_out(),
    index=X_test.index,
)

explainer = shap.TreeExplainer(xgb_model.named_steps["classifier"])
shap_explanation = explainer(X_test_prepared)

shap.plots.bar(shap_explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig(IMAGES_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()

shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig(IMAGES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()
