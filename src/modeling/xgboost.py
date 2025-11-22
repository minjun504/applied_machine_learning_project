from xgboost import XGBClassifier
from .models import Models
from src.config import xgboost_params

class XGBoost(Models):
    def __init__(self, random_state=None, n_trials=50):
        super().__init__(random_state=random_state, 
                         n_trials=n_trials)

    def classifier(self):
        return XGBClassifier(random_state=self.random_state)
    
    def get_params(self):
        return xgboost_params