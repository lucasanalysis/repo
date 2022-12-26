import streamlit as st
import shap
from shap_ml import shap_ml_explain
from dataLoader_csv import read_load_dataset_csv
from streamlit_shap import st_shap
import numpy as np
def shap_explain():
    st.header("model explain with shap")
    TEMPLATE_WRAPPER = """
    <div style="height:{height}px;overflow-y:auto;position:relative;">
        {body}
    </div>
    """
    with st.sidebar.header("Source Data Selection:"):
        st.sidebar.write("select dataset")
        source_data = st.sidebar.file_uploader("Upload/select source (.csv) train data", type=['csv'])
    if source_data is not None :
        model_choices=['LGBMClassifier','XGBRegressor','KernelExplainer']
        selected_model=st.sidebar.selectbox('Please select your model:',model_choices)
        if selected_model is not None and selected_model=='LGBMClassifier':
            shap.initjs()
            data=read_load_dataset_csv(source_data)
            X,y,model=shap_ml_explain(data,selected_model)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            plot_type=st.sidebar.selectbox('Please choose the plot to show',['summary plot','force plot','dependence plot','beeswarm plot'])
            if plot_type=='summary plot':
                st.write('''### summary_plot''')
                st.write('''- summary plot展示了各个特征的shap value累计值，一个特征的整体条形长度越长，
                表示该特征对于区分类别贡献越大。一个特征的分段条形中，某一个分段很长，
                表示该特征对于区分这一特定类别贡献很大''')
                st_shap(shap.summary_plot(shap_values,X,class_names=np.sort(y.unique())),height=800,width=750)
            elif plot_type=='force plot':
                st.write('''### force plot''')
                st.write('''- 展示整体数据的贡献度，每条竖线代表一条数据,将所有竖线在x轴排列起来，就得到了整体数据的贡献度。
                其中每条竖线的分段表示不同特征的贡献，红色表示正的shap value,蓝色表示负的shap value,并且以长度表示
                绝对值的大小。''')
                shap.initjs()
                no_show=st.sidebar.slider('choose num of sample to display',min_value=1,max_value=1000,value=50,step=1)
                class_y=np.sort(y.unique())
                class_show=st.sidebar.selectbox('choose shap value class',class_y)
                class_index=np.where(class_y==class_show)[0][0]
                st_shap(shap.force_plot(explainer.expected_value[class_index], shap_values[class_index][:no_show,:], X.iloc[:no_show,:]), height=600, width=700)
            elif plot_type=='dependence plot':
                st.write('### dependence plot')
                st.write('''- 每一个点代表一条数据，横坐标为一个特征取值，纵坐标为该特征的shap值，并且另外引入一个特征，通过点的颜色表示
                ，由此可以看出特征间的交互效应。''')
                class_y=np.sort(y.unique())
                class_show=st.sidebar.selectbox('choose shap value class',class_y)
                class_index=np.where(class_y==class_show)[0][0]
                name=st.sidebar.selectbox('choose feature',list(X.columns)+['All'])
                if name is not None and class_show is not None and name!='All':
                    st_shap(shap.dependence_plot(name, shap_values[class_index], X, display_features=X))
                elif name=='All' and class_show is not None:
                    for feature in list(X.columns):
                        st_shap(shap.dependence_plot(feature, shap_values[class_index], X, display_features=X))
            elif plot_type=='beeswarm plot':
                explainer = shap.Explainer(model)
                shap_values = explainer(X)
                st.write('### beeswarm plot')
                st.write('''- 展示每一个feature value和shap value的对应关系，从而看出哪些feature影响更大，具体的影响是什么。''')
                class_y=np.sort(y.unique())
                name=st.sidebar.selectbox('Please choose shap value class',class_y)
                class_index=np.where(class_y==name)[0][0]
                st_shap(shap.plots.beeswarm(shap_values[:,:,class_index]))
        if selected_model is not None and selected_model=='XGBRegressor':
            data=read_load_dataset_csv(source_data)
            X,y,model=shap_ml_explain(data,selected_model)
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            plot_type=st.sidebar.selectbox('Please choose the plot to show',['force plot','beeswarm plot','bar plot','dependence plot'])
            if plot_type=='force plot':
                st.write('''### force plot''')
                st.write('''- 展示整体数据的贡献度，每条竖线代表一条数据,将所有竖线在x轴排列起来，就得到了整体数据的贡献度。
                其中每条竖线的分段表示不同特征的贡献，红色表示正的shap value,蓝色表示负的shap value,并且以长度表示
                绝对值的大小。''')
                st_shap(shap.plots.force(shap_values),width=600,height=400)
            elif plot_type=='beeswarm plot':    
                st.write('### beeswarm plot')
                st.write('''- 展示每一个feature value和shap value的对应关系，从而看出哪些feature影响更大，具体的影响是什么。''')
                st_shap(shap.plots.beeswarm(shap_values))
            elif plot_type=='bar plot':    
                st.write('''### bar plot''')
                st.write('''- 所有数据点每一个特征的shap value绝对值求和，以此特征重要性。''')
                st_shap(shap.plots.bar(shap_values))
            elif plot_type=='dependence plot':    
                st.write('### dependence plot')
                st.write('''- 每一个点代表一条数据，横坐标为一个特征取值，纵坐标为该特征的shap值，并且另外引入一个特征，通过点的颜色表示
                ，由此可以看出特征间的交互效应。''')
                name=st.sidebar.selectbox('choose feature',list(X.columns)+['All'])
                if name!='All':
                    st_shap(shap.plots.scatter(shap_values[:,name], color=shap_values))
                elif name=='All':
                    for column in X.columns:
                        st_shap(shap.plots.scatter(shap_values[:,column], color=shap_values))
        if selected_model is not None and selected_model=='KernelExplainer':
            pass