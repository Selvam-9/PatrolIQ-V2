import pandas as pd

from src.logger import get_logger
logger = get_logger(__name__)


class FeatureEngineering:
    def __init__(self):
        pass
    # --------------------
    # Temporal Features
    # --------------------
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating temporal features")

        if "Date" not in df.columns:
            raise ValueError("'Date' column not found in dataset")

        df["Hour"] = df["Date"].dt.hour
        df["DayOfWeek_Num"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        df["Is_Weekend"] = df["DayOfWeek_Num"].isin([5, 6]).astype(int)

        season_map = {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Fall",
            10: "Fall",
            11: "Fall",
        }
        df["Season"] = df["Month"].map(season_map)
        logger.info("Temporal features created successfully")
        return df

    # --------------------
    # Crime Severity Score
    # --------------------

    def add_crime_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating crime severity score")
        if "Primary Type" not in df.columns:
            raise ValueError("'Primary Type' column not found")

        severity_map = {
            # Level 4
            "HOMICIDE": 4,
            "CRIM SEXUAL ASSAULT": 4,
            "KIDNAPPING": 4,
            "OFFENSE INVOLVING CHILDREN": 4,
            # Level 3
            "ROBBERY": 3,
            "AGGRAVATED ASSAULT": 3,
            "WEAPONS VIOLATION": 3,
            "SEX OFFENSE": 3,
            # Level 2
            "THEFT": 2,
            "BURGLARY": 2,
            "MOTOR VEHICLE THEFT": 2,
            "NARCOTICS": 2,
            "DECEPTIVE PRACTICE": 2,
            # Level 1
            "CRIMINAL TRESPASS": 1,
            "PUBLIC PEACE VIOLATION": 1,
            "LIQUOR LAW VIOLATION": 1,
            "GAMBLING": 1,
            "OBSCENITY": 1,
        }

        # Unknown crimes assigned medium severity
        df["Crime_Severity_Score"] = df["Primary Type"].map(severity_map).fillna(2)

        logger.info("Crime severity score created successfully")
        return df

    # --------------------
    # Run All
    # --------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Feature Engineering Started")
        df = self.extract_temporal_features(df)
        df = self.add_crime_severity(df)

        logger.info(f"Feature Engineering Completed. Shape: {df.shape}")
        return df