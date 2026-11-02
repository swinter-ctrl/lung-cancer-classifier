import os
import platform

import pandas as pd
import psutil
import streamlit as st
import torch

from app_common import prediction_log, resource_log

st.caption("Live resource usage of the server running this app, and statistics for every prediction it has served. "
           "Resource samples are taken every 2 seconds while this page is open.")

proc = psutil.Process(os.getpid())
with st.container(border=True):
    st.markdown("**Server environment**")
    vm = psutil.virtual_memory()
    st.table(pd.DataFrame({
        "Property": ["Host", "OS", "Python", "PyTorch", "Inference device", "CPU cores", "Total RAM"],
        "Value": [platform.node(), platform.platform(terse=True), platform.python_version(), torch.__version__,
                  "CPU", str(psutil.cpu_count()), f"{vm.total / 2**30:.1f} GB"],
    }), alt="Server environment details")


@st.fragment(run_every="2s")
def live_resources():
    log = resource_log()
    log.append({"time": pd.Timestamp.now(), "CPU %": psutil.cpu_percent(),
                "System memory %": psutil.virtual_memory().percent,
                "App memory (MB)": proc.memory_info().rss / 2**20})
    df = pd.DataFrame(log)
    last = log[-1]
    with st.container(horizontal=True):
        st.metric("CPU utilization", f"{last['CPU %']:.0f}%", border=True)
        st.metric("System memory", f"{last['System memory %']:.0f}%", border=True)
        st.metric("App memory", f"{last['App memory (MB)']:.0f} MB", border=True)
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**CPU utilization (%)**")
        st.line_chart(df, x="time", y="CPU %", x_label="Time", y_label="CPU %", alt="CPU utilization over time")
    with c2, st.container(border=True):
        st.markdown("**Memory utilization**")
        st.line_chart(df, x="time", y="App memory (MB)", x_label="Time", y_label="App memory (MB)",
                      alt="App memory use over time")


live_resources()

st.subheader("Prediction statistics")
preds = pd.DataFrame(prediction_log())
if preds.empty:
    st.info("No predictions yet. Classify an image on the **Classify** page.", icon=":material/info:")
    st.stop()

lat = preds["latency_ms"]
span_min = max((preds["time"].max() - preds["time"].min()).total_seconds() / 60, 1)
with st.container(horizontal=True):
    st.metric("Predictions served", len(preds), border=True)
    st.metric("Mean response time", f"{lat.mean():.0f} ms", border=True)
    st.metric("95th percentile", f"{lat.quantile(0.95):.0f} ms", border=True)
    st.metric("Throughput", f"{len(preds) / span_min:.1f} / min", border=True)

c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown("**Response time per prediction (ms)**")
    st.line_chart(preds.reset_index(names="prediction"), x="prediction", y="latency_ms",
                  x_label="Prediction #", y_label="Response time (ms)", alt="Response time of each prediction")
with c2, st.container(border=True):
    st.markdown("**Predicted class counts**")
    counts = preds["label"].value_counts().rename_axis("class").reset_index(name="count")
    st.bar_chart(counts, x="class", y="count", x_label="Predicted class", y_label="Predictions",
                 alt="Number of predictions per class")
