from sklearn.tree import DecisionTreeClassifier
from .models import Models
from src.config import pre_prune_tree_params, post_prune_tree_params, decision_tree_params

class DecisionTree(Models):
    def __init__(self, prune=None, random_state=None, n_trials=50):
        super().__init__(random_state=random_state, 
                         n_trials=n_trials)
        self.prune = prune

    def classifier(self):
        return DecisionTreeClassifier(random_state=self.random_state)
    
    def get_params(self):
        if self.prune is None:
            return decision_tree_params
        if self.prune == "pre":
            return pre_prune_tree_params
        if self.prune == "post":
            return post_prune_tree_params