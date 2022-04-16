import pandas as pd


class Preprocess:
    def change_data_types(
        self, df: pd.DataFrame, start_column: str, end_column: str, output_dtype
    ):
        """Change pandas dtypes based on index position

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

    def grouped_feature_eng(
        self,
        df: pd.DataFrame,
        grouped_feature: str,
        features: list,
        target_feature: str,
    ):
        """Transform aggreagation based on grouping an specific feature


        Args:
            df (pd.DataFrame): Dataframe to be transformed
            grouped_feature (str): Feature selected to grouped the transformation. Ej 'user_id
            features (list): features to be transform
            target_feature (str): feature to be used the agregated transformation (Numerical)

        Returns:
            df: transfomed pd.DataFrame
        """

        for index, col in enumerate(features):
            df[col + "_sum"] = df.groupby(grouped_feature)[target_feature].transform(
                "sum"
            )
            df[col + "_mean"] = df.groupby(grouped_feature)[target_feature].transform(
                "mean"
            )
            df[col + "_min"] = df.groupby(grouped_feature)[target_feature].transform(
                "min"
            )
            df[col + "_max"] = df.groupby(grouped_feature)[target_feature].transform(
                "max"
            )
            # df[col + '_len'] = df.groupby(grouped_feature)[target_feature].transform('len')
            df[col + "_count"] = df.groupby(grouped_feature)[target_feature].transform(
                "count"
            )
        return df

    def datetime_transform(self, df: pd.DataFrame, date_feature: str):
        """Aggregate datetime features

        Args:
            df (pd.DataFrame):
            date_feature (str): df column to be transformed

        Returns:
            df: Dataframe with date transformations, (month, day, year, hour, minute, secod)
        """

        df[date_feature] = pd.to_datetime(df[date_feature])
        df[date_feature + "_month"] = df[date_feature].dt.month
        df[date_feature + "_day"] = df[date_feature].dt.day
        df[date_feature + "_year"] = df[date_feature].dt.year
        df[date_feature + "_quarter"] = df[date_feature].dt.quarter
        df[date_feature + "_hour"] = df[date_feature].dt.hour
        df[date_feature + "_minute"] = df[date_feature].dt.minute
        df[date_feature + "_second"] = df[date_feature].dt.second

        return df
