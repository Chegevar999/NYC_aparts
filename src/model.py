from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean_real_estate_with_clusters.csv"

ML_FEATURES = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea",
    "FullBath", "GarageCars", "GarageArea", "TotalBsmtSF", "TotRmsAbvGrd"
]

def train_kmeans() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    cols = [c for c in ML_FEATURES if c in df.columns]
    X = df[cols]
    X_scaled = StandardScaler().fit_transform(X)

    scores = []
    for k in range(2, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        scores.append((k, model.inertia_, silhouette_score(X_scaled, labels)))

    best_k = max(scores, key=lambda row: row[2])[0]
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    pca_values = pca.fit_transform(X_scaled)
    df["PCA1"] = pca_values[:, 0]
    df["PCA2"] = pca_values[:, 1]

    df.to_csv(DATA_PATH, index=False)
    pd.DataFrame(scores, columns=["k", "inertia", "silhouette"]).to_csv(ROOT / "data" / "processed" / "kmeans_metrics.csv", index=False)
    return df

if __name__ == "__main__":
    result = train_kmeans()
    print(result[["Id", "Cluster", "PCA1", "PCA2"]].head())
