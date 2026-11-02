import pandas as pd
import streamlit as st

st.markdown(
    "A GUI application that classifies lung CT slices as **benign** or **malignant** with a fine-tuned "
    "EfficientNet-B0 convolutional network, built for the Cloud Computing Lab project.")

c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown("**Dataset**")
    st.table(pd.DataFrame({
        "Item": ["Name", "Source", "Images used", "Excluded", "Target", "Image size", "Preprocessing"],
        "Value": [
            "IQ-OTH/NCCD lung cancer dataset",
            "Kaggle: hamdallak/the-iqothnccd-lung-cancer-dataset",
            "681 (120 benign, 561 malignant)",
            "416 normal-case slices (task is benign vs malignant)",
            "benign / malignant",
            "Mostly 512 × 512 grayscale CT slices",
            "Grayscale → 3 channels, resize 224 × 224, ImageNet normalisation",
        ],
    }), alt="Dataset summary")
    st.caption("Original collection: Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases, 2019.")
with c2, st.container(border=True):
    st.markdown("**Model and training**")
    st.table(pd.DataFrame({
        "Item": ["Architecture", "Transfer learning", "Loss", "Optimiser", "Augmentation", "Evaluation",
                 "Explainability"],
        "Value": [
            "EfficientNet-B0 (5.3 M parameters)",
            "ImageNet weights, all layers fine-tuned",
            "Class-weighted cross-entropy (offsets 1 : 4.7 imbalance)",
            "AdamW, lr 3e-4, cosine schedule",
            "Flip, ±10° rotation, brightness/contrast jitter",
            "Patient-aware block split + 5-fold block cross-validation",
            "Grad-CAM on the last convolutional block",
        ],
    }), alt="Model and training summary")

st.warning("This is an educational project and is not validated for clinical use. "
           "Do not use it for diagnosis.", title="Not a medical device", icon=":material/medical_information:")
