import pandas as pd
import List


class FeatureEngineeringProcess:
    def grouped_feature_eng(
        self,
        df: pd.DataFrame,
        group_features: list,
        features: list,
        target_feature: str,
    ):
        """Transform aggregation based on grouping a set of features

        Args:
            df (pd.DataFrame): Dataframe to be transformed
            group_features (list): List of features selected to group the transformation.
            features (list): List of features to be transformed.
            target_feature (str): Feature to be used in the aggregated transformation (Numerical)

        Returns:
            df: Transformed pd.DataFrame
        """

        for feature in features:
            df[feature + "_sum"] = df.groupby(group_features)[target_feature].transform(
                "sum"
            )
            df[feature + "_mean"] = df.groupby(group_features)[
                target_feature
            ].transform("mean")
            df[feature + "_min"] = df.groupby(group_features)[target_feature].transform(
                "min"
            )
            df[feature + "_max"] = df.groupby(group_features)[target_feature].transform(
                "max"
            )
            df[feature + "_count"] = df.groupby(group_features)[
                target_feature
            ].transform("count")

        return df

    def datetime_transform(
        self,
        df: pd.DataFrame,
        date_feature: str,
        features: List[str] = ["month", "day"],
    ):
        """Aggregate datetime features

        Args:
            df (pd.DataFrame):
            date_feature (str): df column to be transformed
            features (List[str]): List of date features to extract from the date_feature column

        Returns:
            df: Dataframe with date transformations, (month, day, year, hour, minute, secod)
        """

        df[date_feature] = pd.to_datetime(df[date_feature])

        if "month" in features:
            df[date_feature + "_month"] = df[date_feature].dt.month
        if "day" in features:
            df[date_feature + "_day"] = df[date_feature].dt.day
        if "day_name" in features:
            df[date_feature + "_day_name"] = df[date_feature].dt.day_name()
        if "week" in features:
            df[date_feature + "_week"] = df[date_feature].dt.isocalendar().week
        if "year" in features:
            df[date_feature + "_year"] = df[date_feature].dt.year
        if "quarter" in features:
            df[date_feature + "_quarter"] = df[date_feature].dt.quarter
        if "hour" in features:
            df[date_feature + "_hour"] = df[date_feature].dt.hour
        if "minute" in features:
            df[date_feature + "_minute"] = df[date_feature].dt.minute
        if "second" in features:
            df[date_feature + "_second"] = df[date_feature].dt.second

        return df
