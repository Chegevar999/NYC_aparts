from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "test.csv"
OUT_PATH = ROOT / "data" / "processed" / "clean_real_estate_with_clusters.csv"

NUM_FEATURES = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "MasVnrArea", "BsmtFinSF1", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
    "GrLivArea", "FullBath", "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea",
    "WoodDeckSF", "OpenPorchSF"
]

def clean_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(exclude=np.number).columns:
        mode = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")

    for col in [c for c in NUM_FEATURES if c in df.columns]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            df[col] = df[col].clip(low, high)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    return df

if __name__ == "__main__":
    data = clean_data()
    print(f"Saved cleaned dataset: {OUT_PATH}")
    print(data.shape)
