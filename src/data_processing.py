# src/data_processing.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CreditRiskDataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from CSV with error handling.
        """
        try:
            logger.info(f"Attempting to load data from {self.file_path}...")
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found at {self.file_path}. Please check the path.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading data: {e}")
            raise

    def perform_feature_engineering(self) -> pd.DataFrame:
        """
        Performs basic feature engineering: Aggregates and datetime extraction.
        """
        if self.df is None:
            logger.warning("Dataframe is empty. Loading data first.")
            self.load_data()

        try:
            logger.info("Starting feature engineering...")
            
            # Example: Extract Date Features
            # Ensure TransactionStartTime exists
            if 'TransactionStartTime' in self.df.columns:
                self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
                self.df['Transaction_Hour'] = self.df['TransactionStartTime'].dt.hour
                self.df['Transaction_Day'] = self.df['TransactionStartTime'].dt.day
            
            # Example: Handle Missing Values (Simple Imputation)
            # You will add your advanced WoE logic here later
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

            logger.info("Feature engineering completed.")
            return self.df
            
        except KeyError as e:
            logger.error(f"Missing column required for feature engineering: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            raise

if __name__ == "__main__":
    # Test the class locally
    processor = CreditRiskDataProcessor(file_path="../data/raw/data.csv")
    df = processor.load_data()
    df_processed = processor.perform_feature_engineering()
    print(df_processed.head())