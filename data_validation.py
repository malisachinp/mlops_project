"""Validate the tourism dataset before the ML pipeline runs."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/tourism.csv")
EXPECTED_COLUMNS = ['CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome']

def validate_dataset(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    print("DATA REGISTRATION / VALIDATION")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn validation: PASSED")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts().sort_index())
    print("\nSummary statistics:")
    print(df.describe(include="all").T)
    return df

if __name__ == "__main__":
    validate_dataset()
