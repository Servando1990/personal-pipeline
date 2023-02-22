from random import random
from src.modeling.model import Model
from optuna import Trial
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

from optuna import Trial


class DecisionTreeClassifier(Model):
    def __init__(self, trial: Trial):
        super().__init__()
        self.model = DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 3, 4, 5, 6)
        )

    def train(self, X, y, cv: int = 5):
        super().train(X, y)

        cv_iterator = StratifiedKFold(n_splits=cv, shuffle=True, random_state=123)

        cv_scores = np.zeros(cv)
        for idx, (train_sub_idx, valid_idx) in enumerate(cv_iterator.split(X, y)):

            X_train_sub, X_valid = X[train_sub_idx], X[valid_idx]
            y_train_sub, y_valid = y[train_sub_idx], y[valid_idx]

            model = self.model.fit(X_train_sub, y_train_sub)

            preds = model.predict_proba(X_valid)[:, 1]
            cv_scores[idx] = roc_auc_score(y_valid, preds)

            print(f"CV scores:{cv_scores}")

        return np.mean(cv_scores)

    def predict(self, X, y):
        if not self._trained:
            raise ValueError("Model must be traiend before predictions")
        return self.predict(X)


class RandomForestClassifierWrapper:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, trial, X, y, cv: int = 5):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 40),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "min_impurity_decrease": trial.suggest_uniform(
                "min_impurity_decrease", 0.01, 0.1
            ),
            "max_samples": trial.suggest_uniform("max_samples", 0.5, 1.0),
            "max_features": trial.suggest_uniform("max_features", 0.5, 0.8),
            "ccp_alpha": trial.suggest_loguniform("ccp_alpha", 1e-10, 1e-2),
        }

        cv_iterator = StratifiedKFold(n_splits=cv, shuffle=True, random_state=123)

        cv_scores = np.zeros(cv)
        for idx, (train_sub_idx, valid_idx) in enumerate(cv_iterator.split(X, y)):

            X_train_sub, X_valid = X[train_sub_idx], X[valid_idx]
            y_train_sub, y_valid = y[train_sub_idx], y[valid_idx]

            model = RandomForestClassifier(**params, random_state=123)
            model.fit(X_train_sub, y_train_sub)

            preds = model.predict_proba(X_valid)[:, 1]
            cv_scores[idx] = roc_auc_score(y_valid, preds)

            print(f"CV scores:{cv_scores}")

        return np.mean(cv_scores)
