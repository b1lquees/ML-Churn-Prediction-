# test_pipeline_phase1.py
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.utils import configure_console

DATA_PATH = os.getenv(
    "DATA_PATH", os.path.join(PROJECT_ROOT, "data", "raw", "Telco-Customer-Churn.csv")
)
TARGET_COL = "Churn"

def main():
    print("=== Testing Phase 1: Load -> Preprocess -> Build Features ===")

    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(3))

    # 2. Preprocess
    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(3))

    # 3. Build Features
    print("\n[3] Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    print("\nPhase 1 pipeline completed successfully!")

if __name__ == "__main__":
    configure_console()
    main()
