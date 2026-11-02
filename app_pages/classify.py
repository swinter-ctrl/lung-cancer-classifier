import hashlib
import io

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from app_common import CLASSES, ROOT, get_model, inference, prediction_log

SAMPLES = sorted((ROOT / "samples").glob("*.jpg"))

st.caption("Upload a lung CT slice and the model classifies it as benign or malignant. "
           "Educational demo only, not a medical device.")

source = st.segmented_control("Image source", ["Upload", "Sample image"], default="Upload")
if source == "Sample image":
    name = st.selectbox("Sample (file name shows the true label)", [p.name for p in SAMPLES])
    data = (ROOT / "samples" / name).read_bytes()
else:
    up = st.file_uploader("CT image", type=["jpg", "jpeg", "png", "bmp"])
    data = up.getvalue() if up else None

if data is None:
    st.info("Upload a CT image, or switch to **Sample image** to try one from the test set.",
            icon=":material/upload:")
    st.stop()

img = Image.open(io.BytesIO(data))
key = hashlib.sha1(data).hexdigest()

# Predict once per image, so reruns from other widgets don't re-log the same prediction.
if st.session_state.get("result_key") != key:
    with st.spinner("Classifying..."):
        probs, cam, secs = inference.predict(get_model(), img)
    st.session_state.update(result_key=key, probs=probs, cam=cam, secs=secs)
    prediction_log().append({"time": pd.Timestamp.now(), "label": CLASSES[probs.argmax()],
                             "confidence": float(probs.max()), "latency_ms": secs * 1000})
probs, cam, secs = st.session_state.probs, st.session_state.cam, st.session_state.secs
label, conf = CLASSES[probs.argmax()], float(probs.max())

with st.container(horizontal=True):
    st.metric("Prediction", label.capitalize(), border=True)
    st.metric("Confidence", f"{conf:.1%}", border=True)
    st.metric("Inference time", f"{secs * 1000:.0f} ms", border=True)

if conf < 0.75:
    st.warning("The model is unsure about this image. Treat the result with caution.", icon=":material/warning:")

left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown("**Input CT slice**")
    st.image(img, alt="Uploaded CT slice")
with right, st.container(border=True):
    st.markdown("**Grad-CAM: regions that drove the prediction**")
    st.image(inference.overlay(img, cam), alt="Grad-CAM heatmap over the CT slice")
    st.caption("Brighter yellow means more influence on the predicted class.")

with st.container(border=True):
    st.markdown("**Class probabilities**")
    df = pd.DataFrame({"class": CLASSES, "probability": probs})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Probability"),
        y=alt.Y("class:N", title=None),
        color=alt.Color("class:N", legend=None,
                        scale=alt.Scale(domain=["benign", "malignant"], range=["#2a9d8f", "#e76f51"])),
        tooltip=["class", alt.Tooltip("probability:Q", format=".1%")],
    ).properties(height=120)
    st.altair_chart(chart, alt="Bar chart of class probabilities")
