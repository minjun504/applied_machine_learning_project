from .models import Models
from sklearn.ensemble import RandomForestClassifier
from src.config import random_forest_params

class RandomForest(Models):
    def __init__(self, random_state=None, n_trials=50, n_estimator=None):
        super().__init__(random_state=random_state, 
                         n_trials=n_trials)       
        self.n_estimator = n_estimator        

    def classifier(self):
        if self.n_estimator is None:
            return RandomForestClassifier(random_state=self.random_state)
        else:
            return RandomForestClassifier(random_state=self.random_state, n_estimators=self.n_estimator)
    
    def get_params(self):
        params = random_forest_params.copy()
        if self.n_estimator is not None:
            params.pop("n_estimators", None)
        return params
    
    def train_with_fixed_estimator(self, X_train, y_train, base_params, n_estimators):
        clf = self.classifier()
        params = base_params.copy()
        params["n_estimators"] = n_estimators
        clf.set_params(**params)
        self.model = clf.fit(X_train, y_train)
        return self.model