import numpy as np
import pandas as pd
from src.modeling.decision_tree import DecisionTree
from src.modeling.random_forest import RandomForest
from src.modeling.gradient_boost import GradientBoost
from src.modeling.xgboost import XGBoost
from src.data_loader import load_and_split_data
from src.config import RESULTS_DIR

model_names = ["dt", "rf", "gb", "xgb"]
n_splits = 5

metrics = {name: {"train_f1": [], "test_f1": [], "train_auc": [], "test_auc": []} for name in model_names}

best_model = None
best_model_name = None
best_score = -np.inf
best_params = None
best_split = None

for i in range(n_splits):
    X_train, X_test, y_train, y_test = load_and_split_data(i, file_name="abalone.data", test_size=0.2)
    y_train -= 1
    y_test  -= 1

    models = {
        "dt": DecisionTree(random_state=i),
        "rf": RandomForest(random_state=i),
        "gb": GradientBoost(random_state=i),
        "xgb": XGBoost(random_state=i)
    }

    for model_name, model in models.items():
        model.train_optuna(X_train, y_train)

        results = model.evaluate(X_train, X_test, y_train, y_test)
        for metric_name in results:
            metrics[model_name][metric_name].append(results[metric_name])
        
        current_score = results["test_f1"]

        if current_score > best_score:
            best_score = current_score
            best_model = model                  
            best_params = model.get_best_params()
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
print(f"Results saved to Excel file in {RESULTS_DIR}!")
