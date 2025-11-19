from sklearn.tree import DecisionTreeClassifier
from .models import Models
from ml_assignment_2.config import pre_prune_tree_params, post_prune_tree_params, optuna_pre_prune_tree_params, optuna_post_prune_tree_params

class DecisionTree(Models):
    def __init__(self, prune=None, random_state=None, param_dist=None, n_trials=50, hp_method="optuna"):
        super().__init__(prune=prune, 
                         random_state=random_state, 
                         param_dist=param_dist, 
                         n_trials=n_trials, 
                         hp_method=hp_method)

    def classifier(self):
        return DecisionTreeClassifier(random_state=self.random_state)
    
    def get_randomsearch_params(self):
        if self.prune == "pre":
            return pre_prune_tree_params
        elif self.prune == "post":
            return post_prune_tree_params
        return {}

    def get_optuna_params(self):
        if self.prune == "pre":
            return optuna_pre_prune_tree_params
        elif self.prune == "post":
            return optuna_post_prune_tree_params
        return {}

    





