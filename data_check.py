import streamlit as st
import dataLoader
from streamlit_pandas_profiling import st_profile_report
import pandasProfiling
from deepChecksReport_html import get_deepchecks_report_html
import streamlit.components.v1 as components
from constants import *
def data_check():
    st.header("Deepcheck data validation with streamlit")
    TEMPLATE_WRAPPER = """
    <div style="height:{height}px;overflow-y:auto;position:relative;">
        {body}
    </div>
    """
    with st.sidebar.header("Source Data Selection:"):
        st.sidebar.write("select dataset")
        source_data1 = st.sidebar.file_uploader("Upload/select source (.csv) train data", type=['csv'])
        source_data2 = st.sidebar.file_uploader("Upload/select source (.csv) test data", type=['csv'])
    df = None
    df2=None
    if source_data1 is not None:
        df,ds = dataLoader.read_load_dataset(source_data1)
    if source_data2 is not None:
        df2,ds2=dataLoader.read_load_dataset(source_data2)
    if df is not None or df2 is not None:
        user_choices = ['Dataset Sample','data integrity train','data integrity test',
        'train test validation','model validation','Pandas Profiling train','Pandas Profiling test']
        selected_choice = st.sidebar.selectbox("Please select your choice:", user_choices)
        if selected_choice == 'Dataset Sample' and df is not None:
            st.info("Select dataset has " + str(df.shape[0]) + " rows and " + str(df.shape[1]) + " columns.")
            st.write(df.sample(10))
        elif selected_choice == "Pandas Profiling train" and df is not None:
            df_report = pandasProfiling.pandas_profiling_report(df)
            st.write("Pandas Profiling Report")
            st_profile_report(df_report)
        elif selected_choice == "Pandas Profiling test" and df2 is not None:
            df_report = pandasProfiling.pandas_profiling_report(df2)
            st.write("Pandas Profiling Report")
            st_profile_report(df_report)
        elif selected_choice == 'data integrity train':
            st.write("data integrity train")
            dc_selection = None
            dc_selection = st.sidebar.selectbox("Select Deepchecks Report Type:", data_integrity_dict)
            if dc_selection is not None:
                result_html=get_deepchecks_report_html(df=ds,type=selected_choice,sub_type=dc_selection,df2=None)
                if result_html:
                    height_px = 1000
                html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
                components.html(html, height=height_px)
        elif selected_choice == 'data integrity test':
            st.write("data integrity test")
            dc_selection = None
            dc_selection = st.sidebar.selectbox("Select Deepchecks Report Type:", data_integrity_choice)
            if dc_selection is not None:
                result_html=get_deepchecks_report_html(df=ds2,type=selected_choice,sub_type=dc_selection,df2=None)
                if result_html:
                    height_px = 1000
                html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
                components.html(html, height=height_px)
        elif selected_choice == 'train test validation':
            st.write("train test validation")
            dc_selection = None
            dc_selection = st.sidebar.selectbox("Select Deepchecks Report Type:", train_test_validation_choice)
            if dc_selection is not None:
                result_html=get_deepchecks_report_html(df=ds,type=selected_choice,sub_type=dc_selection,df2=ds2)
                if result_html:
                    height_px = 1000
                html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
                components.html(html, height=height_px)  
    else:
        st.error("Please select your source data to get started")