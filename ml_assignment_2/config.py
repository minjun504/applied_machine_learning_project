from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "reports"

optuna_pre_prune_tree_params = {
    "max_depth": (1, 20),              
    "min_samples_leaf": (1, 20),        
    "min_samples_split": (2, 50),             
    "max_leaf_nodes": (10, 300),            
    "min_impurity_decrease": (0.0, 0.2),      
    "criterion": ["gini", "entropy"]
}

optuna_post_prune_tree_params = {
    "ccp_alpha": (0.0, 0.05),     
    "criterion": ["gini", "entropy"]
}

optuna_random_forest_params = {
    "n_estimators": (50, 500),    
    "max_depth": (3, 20),         
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 10),  
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False]
}

optuna_gradient_boost_params = {
    "n_estimators": (50, 400),       
    "learning_rate": (0.01, 0.2),       
    "max_depth": (2, 6),              
    "min_samples_split": (2, 20),      
    "min_samples_leaf": (1, 10)         
}

optuna_xgboost = {
    "n_estimators": (50, 400),            
    "learning_rate": (0.01, 0.3),         
    "max_depth": (3, 10),                 
    "min_child_weight": (1, 10),          
    "subsample": (0.5, 1.0),              
    "colsample_bytree": (0.5, 1.0),       
    "gamma": (0, 0.3),                    
    "reg_alpha": (0, 1.0),                
    "reg_lambda": (0, 1.0) 
}


pre_prune_tree_params = {
                "max_depth": np.arange(1, 11),
                "min_samples_leaf": np.arange(1, 21),
                "min_samples_split": np.arange(2, 51),
                "max_leaf_nodes": np.arange(10, 1001),
                "min_impurity_decrease": np.linspace(0.0, 0.5, 51),
                "criterion": ["gini", "entropy"]
            }

post_prune_tree_params = {
                "ccp_alpha": np.linspace(0.0, 1.0, 100),
                "criterion": ["gini", "entropy"]
            }

random_forest_params = {
    "n_estimators": np.arange(1, 400),
    "criterion": ["gini", "entropy"]
}

gradient_boost_params = {
    "n_estimators": np.arange(1, 400),
    "learning_rate": np.linspace(0.0001, 0.2, 1000)
}

xgboost_params = {
    "n_estimators": np.arange(50, 501, 50),      
    "learning_rate": np.linspace(0.01, 0.3, 30),  
    "max_depth": np.arange(3, 11),             
    "min_child_weight": np.arange(1, 11),      
    "subsample": np.linspace(0.5, 1.0, 6),           
    "colsample_bytree": np.linspace(0.5, 1.0, 6),    
    "gamma": np.linspace(0, 0.5, 6),                  
    "reg_alpha": np.linspace(0, 1, 5),                
    "reg_lambda": np.linspace(0, 1, 5)    
}            

