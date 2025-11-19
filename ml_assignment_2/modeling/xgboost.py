from xgboost import XGBClassifier
from .models import Models
from ml_assignment_2.config import xgboost_params, optuna_xgboost

class XGBoost(Models):
    def __init__(self, prune=None, random_state=None, param_dist=None, n_trials=50, hp_method="optuna"):
        super().__init__(prune=prune, 
                         random_state=random_state, 
                         param_dist=param_dist,
                         n_trials=n_trials,
                         hp_method=hp_method)

    def classifier(self):
        return XGBClassifier(random_state=self.random_state)
    
    def get_randomsearch_params(self):
        return xgboost_params
    
    def get_optuna_params(self):
        return optuna_xgboost