import numpy as np
import pandas as pd
from ml_assignment_2.modeling.decision_tree import DecisionTree
from ml_assignment_2.modeling.random_forest import RandomForest
from ml_assignment_2.modeling.gradient_boost import GradientBoost
from ml_assignment_2.modeling.xgboost import XGBoost
from ml_assignment_2.dataset import load_and_split_data
from ml_assignment_2.config import RESULTS_DIR

model_names = ["pre_tree", "post_tree", "rf", "gb", "xgb"]
hp_method = "optuna"
n_splits = 5


metrics = {name: {"train_f1": [], "test_f1": []} for name in model_names}
best_params_storage = {name: [] for name in model_names}
trained_models = {}  

for i in range(n_splits):
    X_train, X_test, y_train, y_test = load_and_split_data(i, file_name="abalone.data", test_size=0.2)
    y_train -= 1
    y_test  -= 1

    models = {
        "pre_tree": DecisionTree(random_state=i, prune="pre", hp_method=hp_method),
        "post_tree": DecisionTree(random_state=i, prune="post", hp_method=hp_method),
        "rf": RandomForest(random_state=i, hp_method=hp_method),
        "gb": GradientBoost(random_state=i, hp_method=hp_method),
        "xgb": XGBoost(random_state=i, hp_method=hp_method)
    }

    for model_name, model in models.items():
        if hp_method == "optuna":
            model.train_optuna(X_train, y_train)
        else:
            model.train(X_train, y_train)

        train_f1, test_f1 = model.evaluate(X_train, X_test, y_train, y_test)
        metrics[model_name]["train_f1"].append(train_f1)
        metrics[model_name]["test_f1"].append(test_f1)

        best_params_storage[model_name].append(model.get_best_params())

        trained_models[model_name] = model

summary_metrics = {}
for model_name, model_metrics in metrics.items():
    summary_metrics[model_name] = {}
    for metric_name, values in model_metrics.items():
        values = np.array(values)
        summary_metrics[model_name][metric_name + "_mean"] = values.mean()
        summary_metrics[model_name][metric_name + "_std"] = values.std()

df_metrics = pd.DataFrame(summary_metrics).T
df_metrics.to_excel(f"{RESULTS_DIR}/model_comparison_table_1.xlsx", index=True)
print(f"Metrics saved to Excel file in {RESULTS_DIR}!")


best_model_name = max(summary_metrics, key=lambda x: summary_metrics[x]["test_f1_mean"])
best_model = trained_models[best_model_name]
best_hyperparams = best_model.get_best_params()

print(f"\nBest performing model: {best_model_name}")
print(f"Mean test F1: {summary_metrics[best_model_name]['test_f1_mean']:.4f}")
print("Best hyperparameters found:")
for k, v in best_hyperparams.items():
    print(f"  {k}: {v}")

df_hyperparams = pd.DataFrame(best_params_storage[best_model_name])
df_hyperparams.to_excel(f"{RESULTS_DIR}/{best_model_name}_best_hyperparams.xlsx", index=False)
print(f"Best hyperparameters saved to Excel for {best_model_name}!")
