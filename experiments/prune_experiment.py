import numpy as np
import pandas as pd
from src.modeling.decision_tree import DecisionTree
from src.data_loader import load_and_split_data
from src.config import RESULTS_DIR

model_names = ["pre_prune", "post_prune"]
n_splits = 50

metrics = {name: {"train_f1": [], "test_f1": [], "train_auc": [], "test_auc": []} for name in model_names}

X_train_opt, X_test_opt, y_train_opt, y_test_opt = load_and_split_data(
    n_splits, file_name="abalone.data", test_size=0.2
)
y_train_opt = y_train_opt - 1
y_test_opt = y_test_opt - 1

optimized_params = {}
models_for_tuning = {
    "pre_prune": DecisionTree(prune="pre", random_state=n_splits, n_trials=50),
    "post_prune": DecisionTree(prune="post", random_state=n_splits, n_trials=50)
}

print("==== Optimizing Hyperparameters ====")
for model_name, model in models_for_tuning.items():
    print(f"Tuning {model_name}...")
    model.train_optuna(X_train_opt, y_train_opt)
    optimized_params[model_name] = model.get_best_params()
    print(f"Best params for {model_name}: {optimized_params[model_name]}")

print("\n==== Evaluating Across Splits ====")
for i in range(n_splits):
    X_train, X_test, y_train, y_test = load_and_split_data(i, file_name="abalone.data", test_size=0.2)
    y_train = y_train - 1
    y_test = y_test - 1

    models = {
        "pre_prune": DecisionTree(prune="pre", random_state=i),
        "post_prune": DecisionTree(prune="post", random_state=i)
    }

    for model_name, model in models.items():
        model.train_with_params(X_train, y_train, optimized_params[model_name])

        results = model.evaluate(X_train, X_test, y_train, y_test)
        for metric_name in results:
            metrics[model_name][metric_name].append(results[metric_name])

summary = {}

for model_name, model_metrics in metrics.items():
    summary[model_name] = {}
    for metric_name, values in model_metrics.items():
        values = np.array(values)
        summary[model_name][metric_name + "_mean"] = values.mean()
        summary[model_name][metric_name + "_std"] = values.std()

df = pd.DataFrame(summary).T
df.to_excel(f"{RESULTS_DIR}/prune_experiment.xlsx", index=True)
print(f"\nResults saved to Excel file in {RESULTS_DIR}!")