from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ModelPlots:
    def plot_feature_importances(self, df: pd.DataFrame, model):
        """Plot feature importance given a tree based model and compute Gini coefficient

        Args:
            df pd.DataFrame: dataframe to detect column names
            model: fitted tree based models
        """
        # get the feature importances
        importances = model.feature_importances_

        # create a dataframe of feature importances
        feature_importances = pd.DataFrame(
            {"feature": df.drop("default", axis=1).columns, "importance": importances}
        )

        # sort the dataframe by feature importance
        feature_importances.sort_values(by="importance", ascending=True, inplace=True)

        # Create a bar chart of the feature importances
        plt.figure(figsize=(8, 10))
        plt.barh(
            feature_importances["feature"],
            feature_importances["importance"],
        )
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importances")
        plt.show()

    def plot_learning_curve(
        self,
        model,
        X,
        y,
        train_sizes=[0.1, 0.3, 0.5],
        title=None,
        ylim=None,
        cv=None,
        scoring=None,
    ):
        """Plot a learning curve given a model

        Args:
            model (_type_): _description_
            X (_type_): input features
            y (_type_): target variable
            train_sizes (list, optional): sizes of the training set. Defaults to [0.1, 0.3, 0.5].
            title (_type_, optional): title of th plot. Defaults to None.
            ylim (_type_, optional): the limits of the y-axis.. Defaults to None.
            cv (_type_, optional): the number of cross-validation folds.. Defaults to None.
            scoring (_type_, optional): the evaluation metric used for scoring the model.. Defaults to None.
        """

        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label="Training score")
        plt.plot(train_sizes, test_mean, label="Cross-validation score")

        if ylim is not None:
            plt.ylim(*ylim)
        plt.fill_between(
            train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1
        )
        plt.fill_between(
            train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1
        )

        plt.title(title)
        plt.xlabel("Training Set Size")
        plt.ylabel(scoring)
        plt.legend(loc="best")
        plt.show()
