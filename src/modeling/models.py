from abc import ABC, abstractmethod
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score

class Models(ABC):
    def __init__(self, random_state=None, n_trials=50):
        self.random_state = random_state
        self.n_trials = n_trials
        self.best_params = None
        self.model = None
        self.clf = None

    @abstractmethod
    def classifier(self):
        raise NotImplementedError("Subclasses must define a classifier.")

    @abstractmethod
    def get_params(self):
        pass

    def objective(self, trial, X_train, y_train):
        clf = self.classifier()
        param_dist = self.get_params()
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
        score = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_micro").mean()
        return score
    
    def train_optuna(self, X_train, y_train):
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(n_startup_trials=15, multivariate=True),
                                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), 
                       n_trials=self.n_trials,
                       n_jobs=-1)
        best_params = study.best_params
        self.clf = self.classifier()
        self.clf.set_params(**best_params)
        self.model = self.clf.fit(X_train, y_train)
        self.best_params = best_params
        return self.model
    
    def train_with_params(self, X_train, y_train, params):
        self.clf = self.classifier()
        self.clf.set_params(**params)
        self.model = self.clf.fit(X_train, y_train)
        self.best_params = params
        return self.model

    def get_best_params(self):
        if self.best_params is None:
            raise ValueError("No hyperparameters were found.")
        return self.best_params

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

        y_train_proba = self.model.predict_proba(X_train)
        y_test_proba = self.model.predict_proba(X_test)

        avg = "micro"
        train_f1 = f1_score(y_train, y_train_pred, average=avg)
        test_f1 = f1_score(y_test, y_test_pred, average=avg)

        if y_train_proba.shape[1] == 2:
            train_auc = roc_auc_score(y_train, y_train_proba[:, 1])
            test_auc = roc_auc_score(y_test, y_test_proba[:, 1])
        else:
            train_auc = roc_auc_score(y_train, y_train_proba, multi_class="ovr")
            test_auc = roc_auc_score(y_test, y_test_proba, multi_class="ovr")
        
        return {
            "train_f1": train_f1,
            "test_f1": test_f1,
            "train_auc": train_auc,
            "test_auc": test_auc
        }