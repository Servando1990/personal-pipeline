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



    def change_data_types(self, df :pd.DataFrame, start_column: str, end_column: str, output_dtype):
        """ Change pandas dtypes based on index position

        Args:
            df (pd.DataFrame): 
            start_column (str): 
            end_column (str):
            output_dtype (_type_): 

        Returns:
            df (pd.Dataframe): with desired dtypes
        """
        columns = list(df.columns)
        start = columns.index(start_column)
        end = columns.index(end_column)

        for index, col in enumerate(columns):
            if (start <= index) & (index <= end):
                df[col] = df[col].astype(output_dtype)
            
        return df 
        


# %%
    def grouped_feature_eng(df: pd.DataFrame, grouped_feature: str, features: list, target_feature: str):
        """Feature engineering based on grouped transformations 
        # TODO Come up with a better summary

        Args:
            df (pd.DataFrame): Dataframe to be transformed
            grouped_feature (str): Feature selected to grouped the transformation. Ej 'user_id
            features (list): features to be transform
            target_feature (str): feature to be used the agregated transformation (Numerical)

        Returns:
            df: transfomed pd.DataFrame
        """

        for  index, col in enumerate(features):
            df[col + '_sum'] = df.groupby(grouped_feature)[target_feature].transform('sum')
            df[col + '_mean'] = df.groupby(grouped_feature)[target_feature].transform('mean')
            df[col + '_min'] = df.groupby(grouped_feature)[target_feature].transform('min')
            df[col + '_max'] = df.groupby(grouped_feature)[target_feature].transform('max')
            df[col + '_len'] = df.groupby(grouped_feature)[target_feature].transform('len')
            df[col + '_count'] = df.groupby(grouped_feature)[target_feature].transform('count')
        return df
# %%
    def datetime_transform(df:pd.DataFrame, date_feature: str):
        """Aggregate datetime features

        Args:
            df (pd.DataFrame): 
            date_feature (str): 

        Returns:
            df: 
        """

        df[date_feature] = pd.to_datetime(df[date_feature])
        df[date_feature + '_month'] = df[date_feature].dt.month
        df[date_feature + '_day'] = df[date_feature].dt.day
        df[date_feature + '_year'] = df[date_feature].dt.year
        df[date_feature + '_hour'] = df[date_feature].dt.hour
        df[date_feature + '_minute'] = df[date_feature].dt.minute
        df[date_feature + '_second'] = df[date_feature].dt.second


        return df

 # %%           


    def detect_outliers_boxplot(self, data, features, fac=1.5):

        # median = data.loc[:, features].median()
        q25 = data.loc[:, features].quantile(0.25)
        q75 = data.loc[:, features].quantile(0.75)

        iqr = q75 - q25

        lower_threshold = q25 - (fac * iqr)
        # print("lower threshold")
        print(lower_threshold)
        upper_threshold = q75 + (fac * iqr)
        # print("upper threshold")
        print(upper_threshold)
        isoutlier = data.loc[:, features].apply(
            lambda x: np.any((x < lower_threshold) | (x > upper_threshold)), axis=1
        )

        return isoutlier

