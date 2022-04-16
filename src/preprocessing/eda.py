from typing import Any
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  
import seaborn as sns

class Eda:

    def missing_values_table(self, df: pd.DataFrame):

        """
        Generates a missing values table of a given dataframe

        Returns
        -------
            mis_val_table_ren_columns: missing values table
        """
        mis_val = df.isnull().sum()   
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: "Missing Values", 1: "% of Total Values"}
        )

        mis_val_table_ren_columns = (
            mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0]
            .sort_values("% of Total Values", ascending=False)
            .round(1)
        )

        print(
            "Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) 
            + " columns that have missing values."
        )

        return mis_val_table_ren_columns

    def detect_outliers_boxplot(self, df: pd.DataFrame, features, fac=1.5): 
        # TODO Find out what type hint is features: it could be a str: 'col' or list: [''col1', 'col2']
        # median = data.loc[:, features].median()
        q25 = df.loc[:, features].quantile(0.25)
        q75 = df.loc[:, features].quantile(0.75)

        iqr = q75 - q25

        lower_threshold = q25 - (fac * iqr)
        # print("lower threshold")
        print(lower_threshold)
        upper_threshold = q75 + (fac * iqr)
        # print("upper threshold")
        print(upper_threshold)
        isoutlier = df.loc[:, features].apply(
            lambda x: np.any((x < lower_threshold) | (x > upper_threshold))
        )

        return isoutlier

    def plot_cat_feature(
        self, 
        df: pd.DataFrame, 
        feature: str, 
        target_feature: str,
        label_rotation=False, 
        horizontal_layout=True
        
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
        cat_perc = df[[feature, target_feature]].groupby([feature], as_index=False).mean()
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