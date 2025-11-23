import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from src.modeling.neural_network import NeuralNetwork
from src.data_loader import load_and_split_data, apply_pca
from src.config import RESULTS_DIR

n_splits = 50
alpha_values = [0.0001, 0.001, 0.01]  
pca_ratios = [0.95, 0.98]
solvers = ["adam", "sgd"]
selected_features = ["length", "diameter"]

def get_dataset_variants(X_train, X_test, random_state):
    variants = {}
    
    variants["unnorm_all"] = (X_train.copy(), X_test.copy())
    
    sel_cols = [c for c in selected_features if c in X_train.columns]
    variants["unnorm_selected"] = (X_train[sel_cols].copy(), X_test[sel_cols].copy())
    
    scaler = MinMaxScaler()
    X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    variants["norm_all"] = (X_train_norm, X_test_norm)
    
    variants["norm_selected"] = (X_train_norm[sel_cols].copy(), X_test_norm[sel_cols].copy())
    
    for ratio in pca_ratios:
        X_tr_pca, X_te_pca = apply_pca(X_train, X_test, ratio, random_state)
        variants[f"pca_{ratio}"] = (X_tr_pca, X_te_pca)
    
    return variants

def run_experiments():
    dataset_names = ["unnorm_all", "unnorm_selected", "norm_all", "norm_selected", 
                     "pca_0.95", "pca_0.98"]
    
    model_configs = {
        "adam_default": {"solver": "adam"},
        "sgd_default": {"solver": "sgd"},
        "adam_l2_0.0001": {"solver": "adam", "alpha": 0.0001},
        "adam_l2_0.001": {"solver": "adam", "alpha": 0.001},
        "adam_l2_0.01": {"solver": "adam", "alpha": 0.01},
    }
    
    metrics = {
        ds: {cfg: {"train_f1": [], "test_f1": [], "train_auc": [], "test_auc": []} 
             for cfg in model_configs.keys()}
        for ds in dataset_names
    }
    
    best_model, best_score = None, -np.inf
    best_info = {"dataset": None, "config": None, "split": None, "params": None}
    
    print("==== Evaluating Across Splits ====")
    for i in range(n_splits):
        if i % 10 == 0:
            print(f"Processing split {i}/{n_splits}...")
        
        X_train, X_test, y_train, y_test = load_and_split_data(
            i, file_name="abalone.data", test_size=0.2
        )
        y_train, y_test = y_train - 1, y_test - 1
        
        variants = get_dataset_variants(X_train, X_test, random_state=i)
        
        for ds_name, (X_tr, X_te) in variants.items():
            for cfg_name, cfg_params in model_configs.items():
                solver = cfg_params.get("solver", "adam")
                train_params = {k: v for k, v in cfg_params.items() if k != "solver"}
                
                nn = NeuralNetwork(random_state=i, solver=solver)
                nn.train_with_params(X_tr, y_train, train_params)
                results = nn.evaluate(X_tr, X_te, y_train, y_test)
                
                for m in results:
                    metrics[ds_name][cfg_name][m].append(results[m])
                
                if results["test_f1"] > best_score:
                    best_score = results["test_f1"]
                    best_model = nn
                    best_info = {"dataset": ds_name, "config": cfg_name, 
                                "split": i, "params": cfg_params}
    
    print("\n==== Compiling Results ====")
    
    metric_names = ["train_f1_mean", "train_f1_std", "test_f1_mean", "test_f1_std",
                    "train_auc_mean", "train_auc_std", "test_auc_mean", "test_auc_std"]
    
    col_tuples = [(cfg, metric) for cfg in model_configs.keys() for metric in metric_names]
    col_index = pd.MultiIndex.from_tuples(col_tuples, names=["model_config", "metric"])
    
    data = []
    for ds_name in dataset_names:
        row = []
        for cfg_name in model_configs.keys():
            for base_metric in ["train_f1", "test_f1", "train_auc", "test_auc"]:
                values = np.array(metrics[ds_name][cfg_name][base_metric])
                row.append(values.mean())
                row.append(values.std())
        data.append(row)
    
    df = pd.DataFrame(data, index=dataset_names, columns=col_index)
    df.index.name = "dataset"
    
    df.to_excel(f"{RESULTS_DIR}/nn_experiment_results.xlsx")
    print(f"Results saved to {RESULTS_DIR}/nn_experiment_results.xlsx")
    
    with open(f"{RESULTS_DIR}/best_nn_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    print("\n==== Overall Best Model ====")
    print(f"Dataset: {best_info['dataset']}")
    print(f"Config: {best_info['config']}")
    print(f"Split: {best_info['split']}")
    print(f"Best test F1: {best_score:.4f}")
    print(f"Parameters: {best_info['params']}")
    
    return df, best_model, best_info

if __name__ == "__main__":
    df, best_model, best_info = run_experiments()
    print("\n==== Summary Table ====")
    print(df.to_string())