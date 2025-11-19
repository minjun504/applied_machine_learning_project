from abc import ABC, abstractmethod
import optuna
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score

class Models(ABC):
    def __init__(self, prune=None, random_state=None, param_dist=None, n_trials=50, hp_method="optuna"):
        self.prune  = prune
        self.random_state = random_state
        self.param_dist = param_dist
        self.n_trials = n_trials
        self.hp_method=hp_method
        self.model = None
        self.clf = None

    @abstractmethod
    def classifier(self):
        raise NotImplementedError("Subclasses must define a classifier.")

    @abstractmethod
    def get_optuna_params(self):
        pass

    @abstractmethod
    def get_randomsearch_params(self):
        pass

    def get_param_dist(self):
        if self.param_dist is not None:
            return self.param_dist
        if self.hp_method == "optuna":
            return self.get_optuna_params()
        elif self.hp_method =="random":
            return self.get_randomsearch_params()
        else:
            raise ValueError(f"Unknown search method {self.hp_method}")

    def objective(self, trial, X_train, y_train):
        clf = self.classifier()
        param_dist = self.get_param_dist()
        params = {}
        for key, value in param_dist.items():
            if isinstance(value, list):
                params[key]= trial.suggest_categorical(key, value)
            elif isinstance(value, tuple) and len(value) == 2:
                if all(isinstance(v, int) for v in value):
                    params[key] = trial.suggest_int(key, value[0], value[1])
                else:
                    params[key] = trial.suggest_float(key, value[0], value[1])
            else:
                raise ValueError(f"Unsupported param type for {key}: {value}")
        clf.set_params(**params)
        score = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1-micro").mean()
        return score
    
    def train_optuna(self, X_train, y_train):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=self.n_trials)
        best_params = study.best_params
        self.clf = self.classifier()
        self.clf.set_params(**best_params)
        self.model = self.clf.fit(X_train, y_train)
        self.best_params = best_params
        return self.model

    def train(self, X_train, y_train):
        if self.clf is None:
            self.clf = self.classifier()
        search = RandomizedSearchCV(
            estimator = self.clf, 
            param_distributions=self.get_param_dist(),
            n_iter=50,
            cv=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model = search.fit(X_train, y_train)
        return self.model.best_estimator_

    def train_pred(self, X_train):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train(X_train, y_train) or .train_optuna(X_train, y_train) first.")
        return self.model.predict(X_train)

    def test_pred(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train(X_train, y_train) or .train_optuna(X_train, y_train) first.")
        return self.model.predict(X_test)

    def evaluate(self, X_train, X_test, y_train, y_test):
        y_train_pred = self.train_pred(X_train)
        y_test_pred = self.test_pred(X_test)
        avg = "micro"
        return f1_score(y_train, y_train_pred, average=avg), f1_score(y_test, y_test_pred, average=avg)