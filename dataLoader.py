import pandas as pd
import streamlit as st
from deepchecks.tabular import Dataset
@st.cache
def read_load_dataset(source_data):
    df = pd.read_csv(source_data)
    y=df.columns[0]
    ds=Dataset(df,label=y,label_type="multiclass")
    return df,ds