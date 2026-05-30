import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessing:
    def __init__(self):
        pass

    def drop_missing_geo(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["Latitude", "Longitude"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"{col} column missing from dataset")
            
        before = len(df)
        df = df.dropna(subset=["Latitude", "Longitude"])
        after = len(df)

        logger.info(f"Removed {before-after} rows with missing geo coordinates")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ID" not in df.columns:
            logger.warning("ID column not found. Duplicate removal skipped.")
            return df

        before = len(df)
        df = df.drop_duplicates(subset=["ID"])
        after = len(df)

        logger.info(f"Removed {before-after} duplicate records")

        return df

    def enforce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [
            "Latitude",
            "Longitude",
            "Beat",
            "District",
            "Ward",
            "Community Area",
            "Year",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("Numeric type enforcement completed")
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Data Preprocessing Started")
        df = self.drop_missing_geo(df)
        df = self.remove_duplicates(df)
        df = self.enforce_types(df)

        logger.info(f"Preprocessing completed. Final shape: {df.shape}")
        return df
