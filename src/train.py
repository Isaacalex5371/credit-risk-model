# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, model_save_path: str):
    """
    Loads processed data, trains a model, and saves the artifact.
    """
    try:
        # 1. Load Data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed data file not found at {data_path}")
        
        logger.info("Loading processed data for training...")
        df = pd.read_csv(data_path)
        
        # 2. Check for Target Variable (Make sure you created this in Task 4)
        target_col = 'FraudResult' # Change this to 'is_high_risk' later when you have the proxy
        if target_col not in df.columns:
             raise ValueError(f"Target column '{target_col}' not found in dataset.")

        X = df.drop(columns=[target_col, 'TransactionStartTime'], errors='ignore')
        # Drop non-numeric for this quick interim example
        X = X.select_dtypes(include=['number']) 
        y = df[target_col]

        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Train Model
        logger.info("Initializing Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"Model Training Completed. Accuracy: {acc:.4f}")

        # 6. Save Model
        logger.info(f"Saving model to {model_save_path}...")
        joblib.dump(model, model_save_path)
        logger.info("Model saved successfully.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    # Point this to where your data actually is
    train_model(data_path="../data/raw/data.csv", model_save_path="models/rf_model.pkl")