import numpy as np
import pandas as pd
from src.config import RESULTS_DIR
from src.data_loader import load_and_split_data, apply_pca
from src.modeling.xgboost import XGBoost

model_names = ["95_var", "98_var"]
metrics = {name: {"train_f1": [], "test_f1": [], "train_auc": [], "test_auc": []} for name in model_names}

X_train, X_test, y_train, y_test = load_and_split_data(random_state=0, file_name="abalone.data", test_size=0.2)
y_train -= 1
y_test  -= 1
X_train_95, X_test_95 = apply_pca(X_train, X_test, variance_ratio=0.95, random_state=0)
X_train_98, X_test_98 = apply_pca(X_train, X_test, variance_ratio=0.98, random_state=0)

print("==== Optimizing Hyperparameters ====")
model_95 = XGBoost(random_state=0, n_trials=100)  
model_95.train_optuna(X_train_95, y_train)
params_95 = model_95.get_best_params()
print(f"Best params for 95% variance: {params_95}")

model_98 = XGBoost(random_state=0, n_trials=100)
model_98.train_optuna(X_train_98, y_train)
params_98 = model_98.get_best_params()
print(f"Best params for 98% variance: {params_98}")

print("\n==== Evaluating Across Splits ====")
for i in range(1, 51):
    X_train, X_test, y_train, y_test = load_and_split_data(random_state=i, file_name="abalone.data", test_size=0.2)
    y_train -= 1
    y_test -= 1
    X_train_95, X_test_95 = apply_pca(X_train, X_test, variance_ratio=0.95, random_state=i)
    X_train_98, X_test_98 = apply_pca(X_train, X_test, variance_ratio=0.98, random_state=i)

    model = XGBoost(random_state=i)
    model.train_with_params(X_train_95, y_train, params_95)
    results_95 = model.evaluate(X_train_95, X_test_95, y_train, y_test)

    model = XGBoost(random_state=i)
    model.train_with_params(X_train_98, y_train, params_98)
    results_98 = model.evaluate(X_train_98, X_test_98, y_train, y_test)

    for metric in ["train_f1", "test_f1", "train_auc", "test_auc"]:
        metrics["95_var"][metric].append(results_95[metric])
        metrics["98_var"][metric].append(results_98[metric])

summary = {}

for model_name, model_metrics in metrics.items():
    summary[model_name] = {}
    for metric_name, values in model_metrics.items():
        values = np.array(values)
        summary[model_name][metric_name + "_mean"] = values.mean()
        summary[model_name][metric_name + "_std"] = values.std()

df = pd.DataFrame(summary).T
df.to_excel(f"{RESULTS_DIR}/best_model_pca.xlsx", index=True)
print(f"\nResults saved to Excel file in {RESULTS_DIR}!")