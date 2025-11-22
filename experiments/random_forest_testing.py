from src.modeling.random_forest import RandomForest
from src.config import RESULTS_DIR
from src.data_loader import load_and_split_data
import matplotlib.pyplot as plt
import pandas as pd

X_train, X_test, y_train, y_test = load_and_split_data(random_state=0, file_name="abalone.data", test_size=0.2)
n_list = [10, 50, 100, 200, 300]

metrics_over_n = {
    "n_estimators": [],
    "train_f1": [],
    "test_f1": [],
    "train_auc": [],
    "test_auc": []
}

rf = RandomForest(random_state=0, n_estimator=100)
rf.train_optuna(X_train, y_train)
best_params = rf.get_best_params()

for n in n_list:
    rf = RandomForest(random_state=0, n_estimator=n)
    rf.train_with_fixed_estimator(X_train, y_train, best_params, n_estimators=n)
    results = rf.evaluate(X_train, X_test, y_train, y_test)

    metrics_over_n["n_estimators"].append(n)
    for key in ["train_f1", "test_f1", "train_auc", "test_auc"]:
        metrics_over_n[key].append(results[key])

df_metrics = pd.DataFrame(metrics_over_n)
df_metrics.to_excel(f"{RESULTS_DIR}/rf_metrics_over_n.xlsx", index=False)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df_metrics["n_estimators"], df_metrics["train_f1"], marker='o')
plt.title("Train F1 vs n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("F1 Score")

plt.subplot(2, 2, 2)
plt.plot(df_metrics["n_estimators"], df_metrics["test_f1"], marker='o')
plt.title("Test F1 vs n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("F1 Score")

plt.subplot(2, 2, 3)
plt.plot(df_metrics["n_estimators"], df_metrics["train_auc"], marker='o')
plt.title("Train AUC vs n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("AUC")

plt.subplot(2, 2, 4)
plt.plot(df_metrics["n_estimators"], df_metrics["test_auc"], marker='o')
plt.title("Test AUC vs n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("AUC")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/rf_metrics_plots.png")
plt.show()
    