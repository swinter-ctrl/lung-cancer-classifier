"""Shared settings for data prep, training and the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"          # extracted Kaggle dataset goes here
SPLITS_DIR = ROOT / "data" / "splits"    # train/val/test CSVs
MODELS_DIR = ROOT / "models"             # trained weights + metrics.json

# IQ-OTH/NCCD folder names (note the dataset's own spelling "Bengin").
# Assumption: the assignment asks for benign vs malignant, so "Normal cases"
# is excluded by default. Set INCLUDE_NORMAL = True for a 3-class model.
INCLUDE_NORMAL = False
SOURCE_FOLDERS = {
    "Bengin cases": "benign",
    "Malignant cases": "malignant",
    "Normal cases": "normal",
}
CLASSES = ["benign", "malignant"] + (["normal"] if INCLUDE_NORMAL else [])

IMG_SIZE = 224
SEED = 42
SPLIT = (0.70, 0.15, 0.15)  # train / val / test, stratified by class
