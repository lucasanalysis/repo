import streamlit as st
from constants import *
from data_check import data_check
from shap_explain import shap_explain
from lime_image import lime_image
from mycam import mycam
from mytsne import mytsne
st.set_page_config(
        page_title="data analysis",
        page_icon="👋",
    )
def Home():

    st.write("# Welcome to Lucas analysis! 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        This is an app built specifically for
        Machine Learning and Data Science projects with streamlit.
        **👈 Select a demo from the sidebar** to see  what we can do!
        ### Want to learn more about streamlit?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
            forums](https://discuss.streamlit.io)
        ### See how to use this app
        - Instructions [click here](https://alidocs.dingtalk.com/i/team/e3RmQygAV34QPmaP/docs/e3RmQxdEyqEPPXaP# 「app使用说明」)
    """
    )
page_names_to_funcs = {
    "Home": Home,
    "Data Check": data_check,
    "T-SNE Plot":mytsne,
    "Shap Explain(tabular data)":shap_explain,
    "LIME Explain(image classification)":lime_image,
    "Grad-CAM Explain(object detection)":mycam
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
