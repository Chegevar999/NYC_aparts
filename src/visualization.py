from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean_real_estate_with_clusters.csv"
IMG_DIR = ROOT / "img"

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    IMG_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    df["GrLivArea"].hist(bins=30)
    plt.title("Distribution of living area")
    plt.xlabel("GrLivArea")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "hist_grlivarea_from_script.png", dpi=160)
    plt.close()

    print("Visualization saved to img/")
