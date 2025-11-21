from sklearn.ensemble import GradientBoostingClassifier
from ml_assignment_2.config import gradient_boost_params
from .models import Models

class GradientBoost(Models):
    def __init__(self, prune=None, random_state=None, n_trials=50):
        super().__init__(prune=prune, 
                         random_state=random_state,
                         n_trials=n_trials
                        )

    def classifier(self):
        return GradientBoostingClassifier(random_state=self.random_state)
    
    def get_params(self):
        return gradient_boost_params
    
