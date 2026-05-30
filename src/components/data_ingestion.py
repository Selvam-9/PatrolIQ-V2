import os
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

class DataIngestion:

    def __init__(self, raw_path: str, processed_path: str):
        self.raw_path = raw_path
        self.processed_path = processed_path

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading dataset from: {self.raw_path}")
        
        try:
            df = pd.read_csv(self.raw_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:        
            logger.error(f"Failed to load dataset: {e}")
            raise

    def sample_recent_data(self,df: pd.DataFrame,n_samples: int = 500000) -> pd.DataFrame:

        logger.info("Starting recent data sampling")

        if 'Date' not in df.columns:
            logger.error("'Date' column not found in dataset")
            raise ValueError("'Date' column not found in dataset")

        df['Date'] = pd.to_datetime(df['Date'],errors='coerce')
        df = df.dropna(subset=['Date'])
        logger.info(f"Records after date cleaning: {len(df)}")
        df = df.sort_values(by='Date',ascending=False)
        n_samples = min(n_samples,len(df))
        sampled_df = df.head(n_samples)

        logger.info(f"Sampled dataset shape: {sampled_df.shape}")
        return sampled_df

    def save_processed_data(self, df: pd.DataFrame):
        logger.info(f"Saving processed dataset to: {self.processed_path}")

        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        
        logger.info("Processed dataset saved successfully")

    def run(self):
        logger.info("Data Ingestion Started")

        df = self.load_data()
        sampled_df = self.sample_recent_data(df)
        self.save_processed_data(sampled_df)
        
        logger.info("Data Ingestion Completed")
        return sampled_df
