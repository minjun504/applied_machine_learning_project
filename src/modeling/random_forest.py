from .models import Models
from sklearn.ensemble import RandomForestClassifier
from src.config import random_forest_params

class RandomForest(Models):
    def __init__(self, prune=None, random_state=None, n_trials=50):
        super().__init__(prune=prune, 
                         random_state=random_state, 
                         n_trials=n_trials)               

    def classifier(self):
        return RandomForestClassifier(random_state=self.random_state)
    
    def get_params(self):
        return random_forest_params
    
    def train_with_fixed_n_estimator(self, X_train, y_train, base_params, n_estimators):
        clf = self.classifier()
        params = base_params.copy()
        params["n_estimators"] = n_estimators
        clf.set_params(**params)
        self.model = clf.fit(X_train, y_train)
        return self.model