"""Lung CT classifier: Streamlit entry point.  Run:  streamlit run streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="Lung CT classifier", page_icon=":material/pulmonology:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/classify.py", title="Classify", icon=":material/image_search:", default=True),
        st.Page("app_pages/performance.py", title="Model performance", icon=":material/monitoring:"),
        st.Page("app_pages/monitor.py", title="System monitor", icon=":material/memory:"),
        st.Page("app_pages/about.py", title="About", icon=":material/info:"),
    ],
    position="top",
)
st.title(page.title)
page.run()
