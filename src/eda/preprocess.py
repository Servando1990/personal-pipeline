# %%
import numpy as np
import pandas as pd
# %%


class Preprocess:
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

    def change_dtypes(self, purchases: pd.DataFrame, users: pd.DataFrame):
        """Basic functions to convert dtypes and remove duplicates

        Args:
            purchases (pd.DataFrame)
            users (pd.DataFrame):

        Returns:
            purchases (pd.DataFrame)
            users (pd.DataFrame):

        """
        # Save a raw copy of data
        # purchases_raw = purchases.copy()
        # users_raw = users.copy()

        # Change datatype
        # Justification: 
        # According with the data dictionary some datetypes are wrong like product
        purchases["purchased_at"] = pd.to_datetime(purchases["purchased_at"])
        purchases["user_id"] = purchases["user_id"].astype(object)
        purchases["product"] = purchases["product"].astype(object)
        purchases["product2"] = purchases["product2"].astype(object)
        purchases["value"] = purchases["value"].astype(float)

        # Drop duplicates
        # Justification: We are droping duplicates since this is a transactional dataset
        purchases = purchases.drop_duplicates()

        users["user_id"] = users["user_id"].astype(object)
        users["created_at"] = pd.to_datetime(users["created_at"])
        # users['birthyear'] = users['birthyear'].astype(int)
        users["gender"] = users["gender"].astype(object)
        users["maildomain"] = users["maildomain"].astype(object)
        users["region"] = users["region"].astype(object)
        users["orig_1"] = users["orig_1"].astype(object)
        users["orig_2"] = users["orig_2"].astype(object)
        users["utm_src"] = users["utm_src"].astype(object)
        users["utm_med"] = users["utm_med"].astype(object)
        users["utm_cpg"] = users["utm_cpg"].astype(object)
        users["channel"] = users["channel"].astype(object)
        users["is_active"] = users["is_active"].astype(object)

        return purchases, users

    def feature_eng(self, features: pd.DataFrame): 
        """Helper function that creates aggregated features of a given dataset

        Args:
            features (pd.DataFrame):

        Returns:
            post_analisys: pandas Dataframe
            clustering_data: pandas Dataframe
            classification_data
        """
        
        # Time feature just quarter
        features["purchased_at_quarter"] = features["purchased_at"].dt.quarter
        features["purchased_at_year"] = features["purchased_at"].dt.year
        # Purchase age is a educated guess of the users age
        features["purchase_age"] = features.apply(
            lambda x: x["purchased_at_year"] - x["birthyear"], axis=1
        )
        # Total value per user: strategy sum all values grouped per user
        features["user_total_value"] = features.groupby("user_id")["value"].transform(
            "sum"
        )
        # how long does an user take on average to buy
        features["time_to_purchase"] = features["purchased_at"] - features["created_at"]

        features["mean_time_to_purchase_per_user"] = features.groupby("user_id")[
            "time_to_purchase"
        ].transform("mean")
        # features['mean_time_to_purchase_per_user_days'] = features['mean_time_to_purchase_per_user_days'].dt.days

        # Min purchase time per user
        features["min_time_to_purchase_per_user"] = features.groupby("user_id")[
            "time_to_purchase"
        ].transform("min")
        # features['min_time_to_purchase_per_user_minute'] = features['min_time_to_purchase_per_user_minute'].dt.minute

        # Max purchases time per user
        features["max_time_to_purchase_per_user"] = features.groupby("user_id")[
            "time_to_purchase"
        ].transform("max")
        # features['max_time_to_purchase_per_user_days'] = features['max_time_to_purchase_per_user_days'].dt.days

        # How many purchases per user
        features["user_purchases"] = features.groupby("user_id")["value"].transform(
            "count"
        )

        # total_value_per_channel
        features["total_value_per_channel"] = features.groupby("channel")[
            "value"
        ].transform("sum")
        # total_value_per_region
        features["total_value_per_region"] = features.groupby("region")[
            "value"
        ].transform("sum")
        # total_value_per_utm_src
        features["total_value_per_utm_src"] = features.groupby("utm_src")[
            "value"
        ].transform("sum")
        # total_value_per_utm_src
        features["total_value_per_orig_1"] = features.groupby("orig_1")[
            "value"
        ].transform("sum")

        # Post Analysis Dataset
        post_analysis = features.copy()
        # Clustering dataset
        clustering_data = features.drop(
            columns=[
                "user_id",
                "created_at",
                "birthyear",
                "purchased_at_year",
                "purchased_at",
            ]
        )
        # Changing timedelta variables
        clustering_data["time_to_purchase"] = clustering_data[
            "time_to_purchase"
        ].dt.days  # .astype('int')
        clustering_data["mean_time_to_purchase_per_user"] = clustering_data[
            "mean_time_to_purchase_per_user"
        ].dt.days  # .astype('int')
        clustering_data["min_time_to_purchase_per_user"] = clustering_data[
            "min_time_to_purchase_per_user"
        ].dt.days  # .astype('int')
        clustering_data["max_time_to_purchase_per_user"] = clustering_data[
            "max_time_to_purchase_per_user"
        ].dt.days  # .astype('int')

        # Clasification dataset
        classification_data = clustering_data.copy()

        return post_analysis, clustering_data, classification_data

    # %%
    def grouped_feature_eng(df: pd.DataFrame, grouped_feature: str, features: list) -> pd.DataFrame:

        
        if grouped_feature != "" and features != []:
            new_features = []
            for col in df[[c for c in df.columns if c in features]].columns:
                df[col + '_sum'] = df.groupby(grouped_feature).transform('sum')

                new_features.append(col)
        return df

            

    # %%
    def detect_outliers_boxplot(self, data, features, fac=1.5):

        # median = data.loc[:, features].median()
        q25 = data.loc[:, features].quantile(0.25)
        q75 = data.loc[:, features].quantile(0.75)

        iqr = q75 - q25

        lower_threshold = q25 - (fac * iqr)
        print("lower threshold")
        print(lower_threshold)
        upper_threshold = q75 + (fac * iqr)
        print("upper threshold")
        print(upper_threshold)
        isoutlier = data.loc[:, features].apply(
            lambda x: np.any((x < lower_threshold) | (x > upper_threshold)), axis=1
        )

        return isoutlier
