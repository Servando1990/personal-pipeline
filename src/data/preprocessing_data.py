import pandas as pd
from typing import Tuple, List
from sklearn.impute import SimpleImputer


class Dataset:
    @staticmethod
    def impute_categorical(df: pd.DataFrame) -> pd.DataFrame:
        categorical_vars = df.select_dtypes(include="object")
        imputer = SimpleImputer(strategy="most_frequent")
        imputer.fit(categorical_vars)
        df[categorical_vars.columns] = imputer.transform(categorical_vars)
        return df

    @staticmethod
    def impute_numeric(df: pd.DataFrame, strategy="mean") -> pd.DataFrame:
        numeric_vars = df.select_dtypes(include=["int", "float"])
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit(numeric_vars)
        df[numeric_vars.columns] = imputer.transform(numeric_vars)
        return df

    @staticmethod
    def categorical_encoding_onehot(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                continue
            elif len(df[col].unique()) == 2:
                df[col] = pd.factorize(df[col])[0]
            else:
                df = pd.get_dummies(df, columns=[col])
        return df

    @staticmethod
    def basic_checks(df: pd.DataFrame):
        """Function intended to perform general fixes within a dataframe

        Args:
            df (pd.DataFrame): _description_
        """
        pass

    @staticmethod
    def change_data_types(
        df: pd.DataFrame, start_column: str, end_column: str, output_dtype
    ):
        """Change pandas dtypes based on index position

        Args:
            df (pd.DataFrame): DataFrame to be modified
            start_column (str): First column name to start changing dtypes
            end_column (str): Last column name to stop changing dtypes
            output_dtype (type): Desired data type to convert to

        Returns:
            df (pd.Dataframe): DataFrame with desired dtypes
        """
        columns = list(df.columns)
        start = columns.index(start_column)
        end = columns.index(end_column)

        for index, col in enumerate(columns):
            if (start <= index) & (index <= end):
                df[col] = df[col].astype(output_dtype)

        return df

    def preprocess_data(
        self, df: pd.DataFrame, drop_features=List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess dataframe by calling external functions.
        Performs:
        Drops features
        Missing value imputation:
        Cateegorical encodig:
        Feature engineering:

        Args:
            df (pd.DataFrame): _description_
            drop_features (bool, optional): _description_. Defaults to False.
        """

        # Saves original

        df_backup = df.copy()

        if drop_features is not None:
            df = df.drop(columns=drop_features, axis=1)

        # Call external functions
        # df = Dataset.change_data_types(df)
        df = Dataset.impute_categorical(df)
        df = Dataset.impute_numeric(df)
        # df = Dataset.basic_checks(df)

        # df FeatureEngineering.feature_engienering(df)

        df = Dataset.categorical_encoding_onehot(df)

        return df, df_backup
