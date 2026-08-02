"""Clean tourism data and create train/test workflow artifacts."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/tourism.csv")
ARTIFACT_DIR = Path("artifacts")
TARGET = "ProdTaken"

def clean_data(df):
    df = df.copy()
    # Remove CSV index and identifier; CustomerID is not a predictive feature.
    for col in ["Unnamed: 0", "CustomerID"]:
        if col in df.columns:
            df = df.drop(columns=col)
    # Standardize known inconsistent category labels.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    # Numeric missing values -> median; categorical missing values -> mode.
    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    train, test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df[TARGET]
    )
    train.to_csv(ARTIFACT_DIR / "train.csv", index=False)
    test.to_csv(ARTIFACT_DIR / "test.csv", index=False)
    print(f"Cleaned shape: {df.shape}")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Saved workflow artifacts in {ARTIFACT_DIR}/")

if __name__ == "__main__":
    main()
