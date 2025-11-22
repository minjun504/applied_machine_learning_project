from sklearn.neural_network import MLPClassifier
from .models import Models
from src.config import neural_network_params

class NeuralNetwork(Models):
    def __init__(self, random_state=None, n_trials=50, solver=None):
        super().__init__(random_state=random_state,
                         n_trials=n_trials)
        self.solver = solver
        
    def classifier(self):
        if self.solver is None:
            return MLPClassifier(random_state=self.random_state)
        else:
            return MLPClassifier(random_state=self.random_state, solver=self.solver)
            
    def get_params(self):
        params = neural_network_params.copy()
        if self.solver is not None:
            params.pop("solver", None)
        return params
    
    def train_with_fixed_alpha(self, X_train, y_train, base_params, alpha):
        clf = self.classifier()
        params = base_params.copy()
        params["alpha"] = alpha
        clf.set_params(**params)
        self.model = clf.fit(X_train, y_train)
        return self.model