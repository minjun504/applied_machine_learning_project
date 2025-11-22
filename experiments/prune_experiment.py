import numpy as np
import pandas as pd
from src.modeling.decision_tree import DecisionTree
from src.data_loader import load_and_split_data
from src.config import RESULTS_DIR

model_names = {"pre_prune", "post_prune"}
metrics = {name: {"train_f1": [], "test_f1": [], "train_auc": [], "test_auc": []} for name in model_names}


for i in range(10):
    models = {
        "pre_prune": DecisionTree(prune="pre", random_state=i),
        "post_prune": DecisionTree(prune="post", random_state=i)
    }
    X_train, X_test, y_train, y_test = load_and_split_data(i, file_name="abalone.data", test_size=0.2)

    for model_name, model in models.items():
        model.train_optuna(X_train, y_train)

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
print(f"Results saved to Excel file in {RESULTS_DIR}!")


