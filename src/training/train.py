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
    def clustering(
        self,
        df: pd.DataFrame,
        classification_data: pd.DataFrame,
        post_analysis: pd.DataFrame,
    ):

        """Functions that makes preprocessing steps for clustering and predicts clusters given any data

        Args:
            df (pd.DataFrame):
            classification_data (pd.DataFrame):
            post_analysis (pd.DataFrame):

        Returns:
            classification_data:
            post_analysis
        """

        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        df[:] = imputer.fit_transform(df)

        # Encoding
        df["region"] = df["region"].astype("category")
        df["utm_src"] = df["utm_src"].astype("category")
        df["maildomain"] = df["maildomain"].astype("category")
        df["channel"] = df["channel"].astype("category")
        df["product"] = df["product"].astype("category")
        df["product2"] = df["product2"].astype("category")
        df["is_active"] = df["is_active"].astype("category")

        encoder = CountFrequencyEncoder(encoding_method="frequency")
        encoder.fit(df)
        df = encoder.transform(df)

        # Scaling
        scaler = StandardScaler()
        scaled_clustering = scaler.fit_transform(df)

        # Kmeans
        km_4 = KMeans(n_clusters=4, random_state=123)

        classification_data["cluster"] = km_4.fit_predict(scaled_clustering)
        # classification_data_1it = classification_data.drop(columns=['orig_1', 'maildomain'], axis=1)
        # Encoding clasifcation data
        # classification_data_2it = pd.get_dummies(classification_data_1it, drop_first=True)

        post_analysis["cluster"] = km_4.fit_predict(scaled_clustering)

        return classification_data, post_analysis

    def objective(trial, X_train: np.ndarray, y_train: np.ndarray, cv=7):

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
