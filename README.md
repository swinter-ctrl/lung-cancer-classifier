# Lung Cancer CT Classifier (Benign vs Malignant)

Cloud Computing Lab project: a GUI app (Streamlit) that classifies lung CT
images as **benign** or **malignant**, deployed on **Hugging Face Spaces**.

## Stack
- Dataset: IQ-OTH/NCCD lung cancer dataset (Kaggle: `hamdallak/the-iqothnccd-lung-cancer-dataset`)
- Model: PyTorch EfficientNet-B0, transfer learning from ImageNet
- GUI + dashboard: Streamlit + Altair, Grad-CAM explanations
- Cloud: Hugging Face Spaces

## Steps
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements-train.txt

python src/download_data.py       # needs ~/.kaggle/kaggle.json
python src/prepare_data.py        # builds data/splits/*.csv, prints class counts
python src/train.py --epochs 15   # writes models/model.pt + models/metrics.json
python src/cross_validate.py      # 5-fold block CV -> models/cv_metrics.json
streamlit run streamlit_app.py    # GUI: classify, performance dashboard, system monitor
```

The split is patient-aware (contiguous blocks of slices); `prepare_data.py --mode random`
reproduces the leaky per-image split for comparison.

Settings (classes, image size, split ratio, seed) are in `src/config.py`.
