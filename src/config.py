from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "reports"

decision_tree_params = {
    "max_depth": (1, 20),              
    "min_samples_leaf": (1, 20),        
    "min_samples_split": (2, 50),             
    "max_leaf_nodes": (10, 300),            
    "ccp_alpha": (0.0, 0.05),      
    "criterion": ["gini", "entropy"]
}

pre_prune_tree_params = {
    "max_depth": (1, 20),              
    "min_samples_leaf": (1, 20),        
    "min_samples_split": (2, 50),             
    "max_leaf_nodes": (10, 300),            
    "min_impurity_decrease": (0.0, 0.2),      
    "criterion": ["gini", "entropy"]
}

post_prune_tree_params = {
    "ccp_alpha": (0.0, 0.05),     
    "criterion": ["gini", "entropy"]
}

random_forest_params = {
    "n_estimators": (50, 500),    
    "max_depth": (3, 20),         
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 10),  
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False]
}

gradient_boost_params = {
    "n_estimators": (50, 400),       
    "learning_rate": (0.01, 0.2),       
    "max_depth": (2, 6),              
    "min_samples_split": (2, 20),      
    "min_samples_leaf": (1, 10),
    "subsample": (0.7, 1.0)     
}

xgboost_params = {
    "n_estimators": (50, 400),            
    "learning_rate": (0.01, 0.3),         
    "max_depth": (2, 6),                 
    "min_child_weight": (1, 10),          
    "subsample": (0.6, 1.0),              
    "colsample_bytree": (0.6, 1.0),       
    "gamma": (0, 0.3),                    
    "reg_lambda": (0, 1.0) 
}

neural_network_params ={}