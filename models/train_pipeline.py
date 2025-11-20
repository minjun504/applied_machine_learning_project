import numpy as np
import pandas as pd
from ml_assignment_2.modeling.decision_tree import DecisionTree
from ml_assignment_2.modeling.random_forest import RandomForest
from ml_assignment_2.modeling.gradient_boost import GradientBoost
from ml_assignment_2.modeling.xgboost import XGBoost
from ml_assignment_2.dataset import load_and_split_data
from ml_assignment_2.config import RESULTS_DIR


model_names = ["pre_tree", "post_tree", "rf", "gb", "xgb"]
metrics = {name: {"train_f1": [], "test_f1": [], "train_auc": [], "test_auc": []}
           for name in model_names
}

for i in range(5):
    models = {
    "pre_tree": DecisionTree(random_state=i, prune="pre"),
    "post_tree": DecisionTree(prune="post", random_state=i),
    "rf": RandomForest(random_state=i),
    "gb": GradientBoost(random_state=i),
    "xgb": XGBoost(random_state=i)
    }

    X_train, X_test, y_train, y_test = load_and_split_data(i, file_name="abalone.data", test_size=0.2)
    y_train -= 1
    y_test  -= 1

    for model_name, model in models.items():
        model.train_optuna(X_train, y_train)
        results = model.evaluate(
            X_train, X_test, y_train, y_test
        )
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
df.to_excel(f"{RESULTS_DIR}/model_comparison_table_1.xlsx", index=True)
print(f"Results saved to Excel file in {RESULTS_DIR}!")




