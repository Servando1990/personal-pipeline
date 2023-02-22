from typing import Any
import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


class Eda:
    def missing_values_table(self, df: pd.DataFrame):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * mis_val / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table.columns = ["Missing Values", "% of Total Values"]
        mis_val_table = (
            mis_val_table[mis_val_table["% of Total Values"] != 0]
            .sort_values("% of Total Values", ascending=False)
            .round(1)
        )
        print(
            f"The selected dataframe has {df.shape[1]} columns and {mis_val_table.shape[0]} columns with missing values."
        )
        return mis_val_table

    def detect_outliers_boxplot(df: pd.DataFrame, features: List[str], fac=1.5):
        if isinstance(features, str):
            features = [features]

        q1 = df.loc[:, features].quantile(0.25)
        q3 = df.loc[:, features].quantile(0.75)
        iqr = q3 - q1
        lower_threshold = q1 - fac * iqr
        upper_threshold = q3 + fac * iqr

        outliers = (
            (df[features] < lower_threshold) | (df[features] > upper_threshold)
        ).any(axis=1)
        return outliers

    def plot_cat_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        target_feature: str,
        label_rotation=False,
        horizontal_layout=True,
    ):

        """
        Barplot plotter for categorical features

        Parameters
        ---------
        df: Dataframe
        feature: String, feature to be plotted

        Returns
        -------
        Barplot1: Count of values across the dataset given a feature
        Barplot2: Percentage of feature value with customer == 1

        """

        temp = df[feature].value_counts()
        df1 = pd.DataFrame({feature: temp.index, "Count of values": temp.values})

        # Calculate the percentage of buying customer per category value
        cat_perc = (
            df[[feature, target_feature]].groupby([feature], as_index=False).mean()
        )
        cat_perc.sort_values(by=target_feature, ascending=False, inplace=True)

        if horizontal_layout:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 14))

        s = sns.barplot(ax=ax1, x=feature, y="Count of values", data=df1)
        if label_rotation:
            s.set_xticklabels(s.get_xticklabels(), rotation=90)

        s = sns.barplot(
            ax=ax2, x=feature, y=target_feature, order=cat_perc[feature], data=cat_perc
        )
        if label_rotation:
            s.set_xticklabels(s.get_xticklabels(), rotation=90)
        plt.ylabel("Percent of target with value 1 [%]", fontsize=10)
        plt.tick_params(axis="both", which="major", labelsize=10)

        fig.show()

    def plot_distribution_comp(self, df, target, var, nrow=2):
        i = 0
        t1 = df.loc[df[target] != 0]
        t0 = df.loc[df[target] == 0]

        sns.set_style("whitegrid")
        plt.figure()
        fig, ax = plt.subplots(nrow, 2, figsize=(12, 6 * nrow))

        for feature in var:
            i += 1
            plt.subplot(nrow, 2, i)
            sns.histplot(
                data=t1,
                x=feature,
                stat="density",
                color="red",
                bins=30,
                alpha=0.2,
                label="TARGET == 1",
            )
            sns.histplot(
                data=t0,
                x=feature,
                stat="density",
                color="blue",
                bins=30,
                alpha=0.2,
                label="TARGET == 0",
            )
            plt.ylabel("Hist plot", fontsize=12)
            plt.xlabel(feature, fontsize=12)
            plt.legend()  # added this line to show the legend
            locs, labels = plt.xticks()
            plt.tick_params(axis="both", which="major", labelsize=12)
            plt.title(f"Distribution of {target} ")
        plt.show()
