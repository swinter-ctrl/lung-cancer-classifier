import altair as alt
import pandas as pd
import streamlit as st

from app_common import load_metrics

holdout, cv = load_metrics("metrics.json"), load_metrics("cv_metrics.json")
if holdout is None or cv is None:
    st.error("Metrics files are missing. Run src/train.py and src/cross_validate.py first.")
    st.stop()
classes = holdout["classes"]


def confusion_chart(cm):
    df = pd.DataFrame([(classes[i], classes[j], cm[i][j]) for i in range(len(cm)) for j in range(len(cm))],
                      columns=["actual", "predicted", "count"])
    base = alt.Chart(df).encode(x=alt.X("predicted:N", title="Predicted"), y=alt.Y("actual:N", title="Actual"))
    cells = base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                    tooltip=["actual", "predicted", "count"])
    text = base.mark_text(fontSize=20, fontWeight="bold").encode(
        text="count:Q",
        color=alt.condition(alt.datum.count > df["count"].max() / 2, alt.value("white"), alt.value("black")))
    return (cells + text).properties(height=260)


s = cv["summary"]
st.caption(f"EfficientNet-B0 fine-tuned on IQ-OTH/NCCD (benign vs malignant). "
           f"Headline numbers are {cv['params']['folds']}-fold block cross-validation (mean ± std).")
with st.container(horizontal=True):
    st.metric("CV accuracy", f"{s['accuracy']['mean']:.1%}", border=True,
              help=f"± {s['accuracy']['std']:.1%} across folds")
    st.metric("CV macro F1", f"{s['macro_f1']['mean']:.3f}", border=True,
              help=f"± {s['macro_f1']['std']:.3f} across folds")
    st.metric("CV ROC AUC", f"{s['auc']['mean']:.3f}", border=True, help=f"± {s['auc']['std']:.3f} across folds")
    st.metric("Hold-out test accuracy", f"{holdout['test']['accuracy']:.1%}", border=True,
              help=f"{holdout['split_sizes']['test']} test images; deployed model")

tab_cv, tab_holdout, tab_leak = st.tabs(["Cross-validation", "Deployed model (hold-out split)", "Data leakage study"])

with tab_cv:
    folds = pd.DataFrame(cv["folds"])
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**Metrics per fold**")
        long = folds.melt(id_vars="fold", value_vars=["accuracy", "macro_f1", "auc"], var_name="metric")
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("fold:O", title="Fold"), xOffset="metric:N",
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0.8, 1]), title="Score"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["fold", "metric", alt.Tooltip("value:Q", format=".3f")],
        ).properties(height=300)
        st.altair_chart(chart, alt="Grouped bar chart of accuracy, F1 and AUC for each fold")
    with c2, st.container(border=True):
        st.markdown("**Pooled out-of-fold confusion matrix**")
        st.altair_chart(confusion_chart(cv["pooled"]["confusion_matrix"]),
                        alt="Confusion matrix pooled over all folds")
    with st.container(border=True):
        st.markdown("**Pooled ROC curve (positive class: malignant)**")
        roc = pd.DataFrame({"FPR": cv["pooled"]["roc"]["fpr"], "TPR": cv["pooled"]["roc"]["tpr"]})
        diag = alt.Chart(pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray")
        curve = alt.Chart(roc).mark_line(interpolate="step-after").encode(
            x=alt.X("FPR:Q", title="False positive rate"), y=alt.Y("TPR:Q", title="True positive rate"))
        st.altair_chart((diag + curve).properties(height=320, title=f"AUC = {cv['pooled']['roc']['auc']:.3f}"),
                        alt="ROC curve from pooled cross-validation predictions")
    st.dataframe(folds.drop(columns="confusion_matrix"), hide_index=True, alt="Per-fold metrics table",
                 column_config={"n_test": "Test images", "accuracy": st.column_config.NumberColumn(format="%.3f"),
                                "macro_f1": st.column_config.NumberColumn(format="%.3f"),
                                "auc": st.column_config.NumberColumn(format="%.3f")})

with tab_holdout:
    hist = pd.DataFrame(holdout["history"])
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**Loss per epoch**")
        st.line_chart(hist, x="epoch", y=["train_loss", "val_loss"], x_label="Epoch", y_label="Cross-entropy loss",
                      alt="Training and validation loss per epoch")
    with c2, st.container(border=True):
        st.markdown("**Accuracy per epoch**")
        st.line_chart(hist, x="epoch", y=["train_acc", "val_acc", "val_f1"], x_label="Epoch", y_label="Score",
                      alt="Training and validation accuracy per epoch")
    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        st.markdown("**Test confusion matrix**")
        st.altair_chart(confusion_chart(holdout["test"]["confusion_matrix"]), alt="Hold-out test confusion matrix")
    with c4, st.container(border=True):
        st.markdown("**Per-class report (test set)**")
        rep = pd.DataFrame(holdout["test"]["report"]).T.loc[classes + ["macro avg"]]
        st.dataframe(rep, alt="Precision, recall and F1 per class",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in rep.columns})
        t = holdout["timing"]
        st.caption(f"Training time {t['train_seconds'] / 60:.1f} min on {holdout['params']['device'].upper()}, "
                   f"{holdout['params']['epochs']} epochs. Split sizes: {holdout['split_sizes']}.")
    st.caption("The validation curve plateaus while training accuracy reaches 100%: the model overfits, and one "
               "malignant patient in the validation block (files 415–426) is consistently classified as benign.")

with tab_leak:
    st.markdown(
        "IQ-OTH/NCCD stores 80–200 consecutive CT slices per patient, and neighbouring slices are nearly identical. "
        "A random per-image split leaks the same patient into train and test. Splitting contiguous blocks of slices "
        "(cuts snapped to similarity drops) removes most of it. A *twin* is a slice with correlation ≥ 0.97 to a "
        "training slice.")
    leak = pd.DataFrame([
        {"Split": "Random per image", "Test slices with a train twin": "83 / 103", "Val accuracy": "100%",
         "Test accuracy": "100%"},
        {"Split": "Block (patient-aware)", "Test slices with a train twin": "12 / 99", "Val accuracy": "88.9%",
         "Test accuracy": "100%"},
        {"Split": "Block, 5-fold CV", "Test slices with a train twin": "-",
         "Val accuracy": "-", "Test accuracy": f"{s['accuracy']['mean']:.1%} ± {s['accuracy']['std']:.1%}"},
    ])
    st.table(leak, alt="Comparison of random and block splits")
