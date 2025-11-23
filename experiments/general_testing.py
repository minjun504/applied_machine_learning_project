import numpy as np
import pandas as pd
import pickle
from src.modeling.decision_tree import DecisionTree
from src.modeling.random_forest import RandomForest
from src.modeling.gradient_boost import GradientBoost
from src.modeling.xgboost import XGBoost
from src.data_loader import load_and_split_data
from src.config import RESULTS_DIR

model_names = ["dt", "rf", "gb", "xgb"]
n_splits = 50

metrics = {name: {"train_f1": [], "test_f1": [], "train_auc": [], "test_auc": []} for name in model_names}

best_per_type = {name: {"model": None, "score": -np.inf, "split": None, "params": None} for name in model_names}

best_model = None
best_model_name = None
best_score = -np.inf
best_params = None
best_split = None

X_train_opt, X_test_opt, y_train_opt, y_test_opt = load_and_split_data(50, file_name="abalone.data", test_size=0.2)
y_train_opt = y_train_opt - 1
y_test_opt = y_test_opt - 1

optimized_params = {}
models_for_tuning = {
    "dt": DecisionTree(random_state=50, n_trials=50),
    "rf": RandomForest(random_state=50, n_trials=60),
    "gb": GradientBoost(random_state=50, n_trials=60),
    "xgb": XGBoost(random_state=50, n_trials=100)
}

print("==== Optimizing Hyperparameters ====")
for model_name, model in models_for_tuning.items():
    print(f"Tuning {model_name}...")
    model.train_optuna(X_train_opt, y_train_opt)
    optimized_params[model_name] = model.get_best_params()
    print(f"Best params for {model_name}: {optimized_params[model_name]}")


def compute_model_score(results, alpha=0.5):
    test_performance = (results["test_f1"] + results["test_auc"]) / 2
    f1_gap = results["train_f1"] - results["test_f1"]
    auc_gap = results["train_auc"] - results["test_auc"]
    avg_gap = (f1_gap + auc_gap) / 2
    
    return alpha * test_performance - (1 - alpha) * avg_gap


print("\n==== Evaluating Across Splits ====")
for i in range(n_splits):
    X_train, X_test, y_train, y_test = load_and_split_data(i, file_name="abalone.data", test_size=0.2)
    y_train = y_train - 1
    y_test = y_test - 1

    models = {
        "dt": DecisionTree(random_state=i),
        "rf": RandomForest(random_state=i),
        "gb": GradientBoost(random_state=i),
        "xgb": XGBoost(random_state=i)
    }

    for model_name, model in models.items():
        model.train_with_params(X_train, y_train, optimized_params[model_name])

        results = model.evaluate(X_train, X_test, y_train, y_test)
        for metric_name in results:
            metrics[model_name][metric_name].append(results[metric_name])
        
        current_score = compute_model_score(results, alpha=0.75)

        if current_score > best_per_type[model_name]["score"]:
            best_per_type[model_name]["score"] = current_score
            best_per_type[model_name]["model"] = model
            best_per_type[model_name]["split"] = i
            best_per_type[model_name]["params"] = optimized_params[model_name]

        if current_score > best_score:
            best_score = current_score
            best_model = model                  
            best_params = optimized_params[model_name]
            best_model_name = model_name
            best_split = i

summary = {}

for model_name, model_metrics in metrics.items():
    summary[model_name] = {}
    for metric_name, values in model_metrics.items():
        values = np.array(values)
        summary[model_name][metric_name + "_mean"] = values.mean()
        summary[model_name][metric_name + "_std"] = values.std()

df = pd.DataFrame(summary).T
df.to_excel(f"{RESULTS_DIR}/model_comparison_table_1.xlsx", index=True)
print(f"\nResults saved to Excel file in {RESULTS_DIR}!")

with open(f"{RESULTS_DIR}/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(f"{RESULTS_DIR}/best_decision_tree.pkl", "wb") as f:
    pickle.dump(best_per_type["dt"]["model"], f)

print("\n==== Best Model Per Type ====")
for model_name, info in best_per_type.items():
    print(f"{model_name.upper()}: split={info['split']}, score={info['score']:.4f}")

print("\n==== Overall Best Model ====")
print(f"Model: {best_model_name}")
print(f"Best split: {best_split}")
print(f"Best composite score: {best_score:.4f}")
print(f"Best hyperparameters: {best_params}")