import lightgbm as lightgbm
import numpy as np
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
from optuna.integration import LightGBMPruningCallback
import optuna
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class Train:


    def objective(trial, X_train: np.ndarray, y_train: np.ndarray, cv=7):
        # TODO SET as parameters of function: objective, eval_metric, metric

        param_grid = {
            "n_estimators": trial.suggest_categorical("n_estimators", [10, 100]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01]),
        }

        cv_iterator = StratifiedKFold(n_splits=cv, shuffle=True, random_state=123)

        cv_scores = np.zeros(cv)
        for idx, (train_sub_idx, valid_idx) in enumerate(
            cv_iterator.split(X_train, y_train)
        ):

            X_train_sub, X_valid = X_train[train_sub_idx], X_train[valid_idx]
            y_train_sub, y_valid = y_train[train_sub_idx], y_train[valid_idx]

            model = lightgbm.LGBMClassifier(
                class_weight="balanced", colsample_bytree=0.65, objective="multiclass",
                **param_grid)
            model.fit(
                X_train_sub,
                y_train_sub,
                eval_set=[(X_valid, y_valid)],
                eval_metric="multiclass",
                verbose=-2,
                early_stopping_rounds=50,
                callbacks=[LightGBMPruningCallback(trial=trial, metric="multiclass")],
            )
            preds = model.score(X_valid, y_valid)

            cv_scores[idx] = preds

        return 1 - np.mean(cv_scores)
