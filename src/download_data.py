"""Download IQ-OTH/NCCD from Kaggle into data/raw.

Needs a Kaggle API token: put kaggle.json in ~/.kaggle/ (Kaggle > Settings >
Create New Token), or set KAGGLE_USERNAME and KAGGLE_KEY.
"""
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

from config import RAW_DIR

DATASET = "hamdallak/the-iqothnccd-lung-cancer-dataset"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(RAW_DIR), quiet=False)
    for z in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(RAW_DIR)
        z.unlink()
    print(f"Extracted to {RAW_DIR}")


if __name__ == "__main__":
    main()
