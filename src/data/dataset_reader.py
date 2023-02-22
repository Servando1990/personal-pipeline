import pandas as pd
from pathlib import Path


class DatasetReader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def get_data(self, filename: str) -> pd.DataFrame:
        data_path = self.data_dir / filename

        df = pd.read_csv(data_path)

        return df
