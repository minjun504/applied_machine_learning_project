from xgboost import XGBClassifier
from .models import Models
from ml_assignment_2.config import xgboost_params

class XGBoost(Models):
    def __init__(self, prune=None, random_state=None, param_dist=None, n_trials=50):
        super().__init__(prune=prune, random_state=random_state, param_dist=param_dist, n_trials=n_trials)

    def classifier(self):
        return XGBClassifier(random_state=self.random_state)
    
    def get_default_params(self):
        return xgboost_params