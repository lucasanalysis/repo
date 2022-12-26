import streamlit as st
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_notebook,output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10,Inferno10,Paired3,Category10_10
import h5py
from io import BytesIO
import io
from PIL import Image
import base64
import cv2
from skimage import io
import os
from openTSNE import TSNE
import torch
from tsne_function import save_image_to_array
from bokeh.layouts import gridplot
from bokeh.layouts import column
from bokeh.models import (ColumnDataSource, DataTable, HoverTool, IntEditor,
                          NumberEditor, NumberFormatter, SelectEditor,
                          StringEditor, StringFormatter, TableColumn)
import torch
import clip
from PIL import Image
from tqdm import tqdm
def mytsne():
    feature=None
    no_plot=st.sidebar.selectbox('Please select the number of datasets',[1,2])
    st.header("TSNE Visualizer")
    if no_plot==1:
        uploaded_file = st.sidebar.file_uploader("Choose a file(pth)")
        if uploaded_file is not None:
            if 'pth' in uploaded_file.name:
                df=torch.load(uploaded_file)
                labels=df['label']
                path=df['path']
                if 'embedding' in df.keys():
                    feature=df['embedding'].numpy()
                else:
                    def clip_cache(path):
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model, preprocess = clip.load("ViT-B/32", device=device)
                        embedding=[]
                        for i in tqdm(range(len(path))):
                            image = preprocess(Image.open(path[i])).unsqueeze(0).to(device)
                            with torch.no_grad():
                                image_features = model.encode_image(image)
                                embedding.append(image_features[0].numpy().reshape(1,512))
                        feature=np.array([embedding[i].reshape(1,-1)[0] for i in range(len(path))])
                        return feature
                    clip_cache_st=st.cache(clip_cache)
                    feature=clip_cache_st(path)

            # elif 'csv' in uploaded_file.name:
            #     df=pd.read_csv(uploaded_file)
            #     labels=np.array(df.iloc[:,1])
            #     feature=np.array(df.iloc[:,2:])
            #     path=df.iloc[:,0]
            parameter_choice=st.sidebar.selectbox('Please choose the parameter',['default','self'])
            if parameter_choice=='default':
                num_perplexity=20
                num_iter=100
                metric='cosine'
                num_jobs=4
                num_components=2
            elif parameter_choice=='self':
                num_perplexity = st.sidebar.slider("Choose No of perplexity: ", min_value=0,   
                        max_value=50, value=20, step=1)
                num_iter = st.sidebar.slider("Choose No of iter: ", min_value=0,   
                        max_value=200, value=100, step=1)
                metric=st.sidebar.selectbox('Please select metric',["euclidean","cosine"])
                num_jobs=st.sidebar.slider("Choose No of jobs: ", min_value=0,   
                        max_value=20, value=4, step=1)
                num_components=st.sidebar.selectbox("Choose No of components: ",[2])
            tsne = TSNE(
                perplexity=num_perplexity,
                n_iter=num_iter,
                metric=metric,
                n_jobs=num_jobs,
                n_components=num_components,
                random_state=42,
                verbose=True,
            )
            def embeddable_image(data):
                img_data = data.astype(np.uint8)
                image = Image.fromarray(img_data)#.resize((64, 64), Image.BICUBIC)
                buffer = BytesIO()
                image.save(buffer, format='png')
                for_encoding = buffer.getvalue()
                return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()
            def plot_tsne_single_2d(images,tsne,labels,n_components = 2):
                if n_components ==2:
                    digits_df = pd.DataFrame(tsne, columns=('pc1', 'pc2'))               
                digits_df['label'] = [str(x) for x in labels]
                digits_df['path']=[str(x) for x in path]
                print(f"Image Data Shape:{images.shape}")
                print("Plotting T-SNE projection of features ======>")
                digits_df['image'] = list(map(embeddable_image, image_np))
                datasource = ColumnDataSource(digits_df)
                color_mapping = CategoricalColorMapper(factors=sorted(digits_df['label'].unique()),
                                                palette=Category10_10)#Spectral10)
                labels_unique=sorted(digits_df['label'].unique())
                path_unique=sorted(digits_df['path'].unique())
                columns=[
                TableColumn(field="label", title="label",
                editor=SelectEditor(options=labels_unique),
                formatter=StringFormatter(font_style="bold")),
                TableColumn(field="path", title="path",
                editor=StringEditor(completions=path_unique)),
                TableColumn(field="pc1", title="pc1", editor=IntEditor()),
                TableColumn(field="pc2", title="pc2", editor=IntEditor())
                ]
                # data_table = DataTable(source=datasource, columns=columns, editable=True, width=800,
                #        index_position=-1, index_header="row index", index_width=60)
                plot_figure = figure(
                title='TSNE projection',
                plot_width=900,
                plot_height=600,
                tools=('pan, box_select, box_zoom,poly_select, wheel_zoom, reset, tap','lasso_select','save'),
                active_drag="lasso_select"
            )

                plot_figure.add_tools(HoverTool(tooltips="""
                <div>
                    <div>
                        <img src='@image' style='float: left; margin: 2px 2px 2px 2px' width="100" 
                height="100" />
                    </div>
                    <div>
                        <span style='font-size: 12px; color: #224499'>Label:</span>
                        <span style='font-size: 12px'>@label</span>
                    </div>
                </div>
                """))
                if n_components ==2:
                  #class_s=st.selectbox('Please select the data to show in table',color_mapping.factors)
                  table_list=[]
                  for i,j in enumerate(labels_unique):
                    datasource_temp=ColumnDataSource(digits_df[digits_df['label']==j])
                    plot_figure.circle(
                        'pc1',
                        'pc2',
                        source=datasource_temp,
                        color=color_mapping.palette[i],#dict(field='label', transform=color_mapping),
                        line_alpha=0.6,
                        fill_alpha=0.6,
                        size=4,
                        legend=color_mapping.factors[i]#'label'
                    )
                    plot_figure.legend.location = "top_left"
                    plot_figure.legend.click_policy="hide"
                    locals()['datatable'+str(i)]= DataTable(source=datasource_temp, columns=columns, editable=True, width=800,height=100,
                        index_position=-1, index_header="row index", index_width=60)
                    table_list.append(locals()['datatable'+str(i)])
                    output_file("interactive_legend.html", title="interactive_legend.py example")
                scatter_plot=[plot_figure]
                scatter_plot.extend(table_list)    
                st.bokeh_chart(column(scatter_plot))
            

            # def plot_tsne_single_3d(images,tsne,labels,n_components = 3):
            #     if n_components ==3:
            #         digits_df = pd.DataFrame(tsne, columns=('pc1', 'pc2','pc3'))               
            #     digits_df['label'] = [str(x) for x in labels]
            #     digits_df['path']=[str(x) for x in path]
            #     print(f"Image Data Shape:{images.shape}")
            #     print("Plotting T-SNE projection of features ======>")
            #     digits_df['image'] = list(map(embeddable_image, image_np))
            #     datasource = ColumnDataSource(digits_df)
            #     color_mapping = CategoricalColorMapper(factors=list(set(labels)),
            #                                     palette=Category10_10)#Spectral10)
            #     labels_unique=sorted(digits_df['label'].unique())
            #     path_unique=sorted(digits_df['path'].unique())
            #     columns=[
            #     TableColumn(field="label", title="label",
            #     editor=SelectEditor(options=labels_unique),
            #     formatter=StringFormatter(font_style="bold")),
            #     TableColumn(field="path", title="path",
            #     editor=StringEditor(completions=path_unique)),
            #     TableColumn(field="pc1", title="pc1", editor=IntEditor()),
            #     TableColumn(field="pc2", title="pc2", editor=IntEditor()),
            #     TableColumn(field="pc3", title="pc3", editor=IntEditor())
            #     ]
            #     data_table = DataTable(source=datasource, columns=columns, editable=True, width=800,
            #            index_position=-1, index_header="row index", index_width=60)
            #     plot_figure_left = figure(
            #     title='TSNE projection 1',
            #     plot_width=650,
            #     plot_height=500,
            #     tools=('pan, box_select,box_zoom, poly_select, wheel_zoom, reset, tap','lasso_select','save'),
            #     active_drag="lasso_select"
            # )
            #     plot_figure_right = figure(
            #     title='TSNE projection 2',
            #     plot_width=650,
            #     plot_height=500,
            #     tools=('pan, box_select, box_zoom,poly_select, wheel_zoom, reset, tap','lasso_select','save'),
            #     active_drag="lasso_select"
            # )

            #     plot_figure_left.add_tools(HoverTool(tooltips="""
            #     <div>
            #         <div>
            #             <img src='@image' style='float: left; margin: 2px 2px 2px 2px' width="100" 
            #     height="100" />
            #         </div>
            #         <div>
            #             <span style='font-size: 12px; color: #224499'>Label:</span>
            #             <span style='font-size: 12px'>@label</span>
            #         </div>
            #     </div>
            #     """))
            #     plot_figure_right.add_tools(HoverTool(tooltips="""
            #     <div>
            #         <div>
            #             <img src='@image' style='float: left; margin: 2px 2px 2px 2px' width="100" 
            #     height="100" />
            #         </div>
            #         <div>
            #             <span style='font-size: 12px; color: #224499'>Label:</span>
            #             <span style='font-size: 12px'>@label</span>
            #         </div>
            #     </div>
            #     """))
            #     if n_components ==3:
            #         plot_figure_left.circle(
            #             'pc2',
            #             'pc1',
            #             source=datasource,
            #             color=dict(field='label', transform=color_mapping),
            #             line_alpha=0.6,
            #             fill_alpha=0.6,
            #             size=4,
            #             legend_field='label'
            #         )
            #         plot_figure_left.xaxis.axis_label = "pc2"
            #         plot_figure_left.yaxis.axis_label = "pc1"
            #         plot_figure_left.legend.location = "top_left"
            #         plot_figure_left.legend.click_policy="hide"
            #         plot_figure_right.circle(
            #             'pc3',
            #             'pc1',
            #             source=datasource,
            #             color=dict(field='label', transform=color_mapping),
            #             line_alpha=0.6,
            #             fill_alpha=0.6,
            #             size=4,
            #             legend_field='label'
            #         )
            #         plot_figure_right.xaxis.axis_label = "pc3"
            #         plot_figure_right.yaxis.axis_label = "pc1"
            #         plot_figure_right.legend.location = "top_left"
            #         plot_figure_right.legend.click_policy="hide"
            #         output_file("interactive_legend.html", title="interactive_legend.py example")
            #     st.bokeh_chart(gridplot([[plot_figure_left,plot_figure_right,data_table]]))
            if feature is not None:
                if num_components==2 and st.button('Display T-SNE'):
                    if uploaded_file is not None:
                        TSNE_embedded = tsne.fit(feature)
                        tsne=np.array(TSNE_embedded)
                    if uploaded_file is not None and path is not None:
                        image_np=save_image_to_array(path)
                    plot_tsne_single_2d(image_np,tsne,labels,n_components = num_components)
            # if num_components==3 and st.button('Display T-SNE'):
            #     if uploaded_file is not None:
            #         if feature is not None:
            #             TSNE_embedded = tsne.fit(feature)
            #             tsne=np.array(TSNE_embedded)
            #     if uploaded_file is not None and path is not None:
            #         image_np=save_image_to_array(path)
            #     plot_tsne_single_3d(image_np,tsne,labels,n_components = num_components)      
    elif no_plot==2:
        uploaded_file1 = st.sidebar.file_uploader("Choose the first file(pth)")
        uploaded_file2 = st.sidebar.file_uploader("Choose the second file(pth)")
        feature1,feature2=None,None
        if uploaded_file1 is not None:
            if 'pth' in uploaded_file1.name:
                df1=torch.load(uploaded_file1)
                labels1=df1['label']
                path1=df1['path']
                if 'embedding' in df1.keys():
                    feature1=df1['embedding'].numpy()
                else:
                    def clip_cache1(path):
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model, preprocess = clip.load("ViT-B/32", device=device)
                        embedding=[]
                        for i in tqdm(range(len(path))):
                            image = preprocess(Image.open(path[i])).unsqueeze(0).to(device)
                            with torch.no_grad():
                                image_features = model.encode_image(image)
                                embedding.append(image_features[0].numpy().reshape(1,512))
                        feature=np.array([embedding[i].reshape(1,-1)[0] for i in range(len(path))])
                        return feature
                    clip_cache_st1=st.cache(clip_cache1)
                    feature1=clip_cache_st1(path1)

            # elif 'csv' in uploaded_file1.name:
            #     df1=pd.read_csv(uploaded_file1)
            #     labels1=np.array(df1.iloc[:,1])
            #     feature1=np.array(df1.iloc[:,2:])
            #     path1=df1.iloc[:,0]
        if uploaded_file2 is not None:
            if 'pth' in uploaded_file2.name:
                df2=torch.load(uploaded_file2)
                labels2=df2['label']
                path2=df2['path']
                if 'embedding' in df2.keys():
                    feature2=df2['embedding'].numpy()
                else:
                    def clip_cache2(path):
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model, preprocess = clip.load("ViT-B/32", device=device)
                        embedding=[]
                        for i in tqdm(range(len(path))):
                            image = preprocess(Image.open(path[i])).unsqueeze(0).to(device)
                            with torch.no_grad():
                                image_features = model.encode_image(image)
                                embedding.append(image_features[0].numpy().reshape(1,512))
                        feature=np.array([embedding[i].reshape(1,-1)[0] for i in range(len(path))])
                        return feature
                    clip_cache_st2=st.cache(clip_cache2)
                    feature2=clip_cache_st2(path2)
            # elif 'csv' in uploaded_file2.name:
            #     df2=pd.read_csv(uploaded_file2)
            #     labels2=np.array(df2.iloc[:,1])
            #     feature2=np.array(df2.iloc[:,2:])
            #     path2=df2.iloc[:,0]
            parameter_choice=st.sidebar.selectbox('Please choose the parameter',['default','self'])
            if parameter_choice=='default':
                num_perplexity=20
                num_iter=100
                metric='cosine'
                num_jobs=4
                num_components=2
            elif parameter_choice=='self':
                num_perplexity = st.sidebar.slider("Choose No of perplexity: ", min_value=0,   
                        max_value=50, value=20, step=1)
                num_iter = st.sidebar.slider("Choose No of iter: ", min_value=0,   
                        max_value=200, value=100, step=1)
                metric=st.sidebar.selectbox('Please select metric',["euclidean","cosine"])
                num_jobs=st.sidebar.slider("Choose No of jobs: ", min_value=0,   
                        max_value=20, value=4, step=1)
                num_components=st.sidebar.selectbox("Choose No of components: ",[2])
            tsne = TSNE(
                perplexity=num_perplexity,
                n_iter=num_iter,
                metric=metric,
                n_jobs=num_jobs,
                n_components=num_components,
                random_state=42,
                verbose=True,
            )
            def embeddable_image(data):
                img_data = data.astype(np.uint8)
                image = Image.fromarray(img_data)#.resize((64, 64), Image.BICUBIC)
                buffer = BytesIO()
                image.save(buffer, format='png')
                for_encoding = buffer.getvalue()
                return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()
            def plot_tsne_double_2d(images1,images2,tsne1,tsne2,labels1,labels2,n_components = 2):
                if n_components ==2:
                    digits_df1 = pd.DataFrame(tsne1, columns=('pc1', 'pc2'))               
                    digits_df1['label'] = [str(x) for x in labels1]
                    digits_df1['path']=[str(x) for x in path1]
                    print(f"Image Data Shape:{images1.shape}")
                    print("Plotting T-SNE projection of features ======>")
                    digits_df1['image'] = list(map(embeddable_image, image_np1))
                    datasource1 = ColumnDataSource(digits_df1)
                    digits_df2 = pd.DataFrame(tsne2, columns=('pc1', 'pc2'))               
                    digits_df2['label'] = [str(x) for x in labels2]
                    digits_df2['path']=[str(x) for x in path2]
                    print(f"Image Data Shape:{images2.shape}")
                    print("Plotting T-SNE projection of features ======>")
                    digits_df2['image'] = list(map(embeddable_image, image_np2))
                    datasource2 = ColumnDataSource(digits_df2)
                labels_unique1=sorted(digits_df1['label'].unique())
                labels_unique2=sorted(digits_df2['label'].unique())
                color_mapping1 = CategoricalColorMapper(factors=labels_unique1,
                                                palette=Category10_10)#Spectral10)
                color_mapping2 = CategoricalColorMapper(factors=labels_unique2,
                                                palette=Category10_10)#Spectral10)
                plot_figure_left = figure(
                title='TSNE projection 1',
                plot_width=500,
                plot_height=500,
                tools=('pan, box_select, box_zoom,poly_select, wheel_zoom, reset, tap','lasso_select','save'),
                active_drag="lasso_select"
            )

                plot_figure_left.add_tools(HoverTool(tooltips="""
                <div>
                    <div>
                        <img src='@image' style='float: left; margin: 2px 2px 2px 2px' width="100" 
                height="100" />
                    </div>
                    <div>
                        <span style='font-size: 12px; color: #224499'>Label:</span>
                        <span style='font-size: 12px'>@label</span>
                    </div>
                </div>
                """))
                plot_figure_right = figure(
                title='TSNE projection 2',
                plot_width=500,
                plot_height=500,
                x_range=plot_figure_left.x_range, 
                y_range=plot_figure_left.y_range,
                tools=('pan, box_select, box_zoom,poly_select, wheel_zoom, reset, tap','lasso_select','save'),
                active_drag="lasso_select"
            )

                plot_figure_right.add_tools(HoverTool(tooltips="""
                <div>
                    <div>
                        <img src='@image' style='float: left; margin: 2px 2px 2px 2px' width="100" 
                height="100" />
                    </div>
                    <div>
                        <span style='font-size: 12px; color: #224499'>Label:</span>
                        <span style='font-size: 12px'>@label</span>
                    </div>
                </div>
                """))
                if n_components ==2:
                    for i,j in enumerate(labels_unique1):
                        datasource1=ColumnDataSource(digits_df1[digits_df1['label']==j])
                        plot_figure_left.circle(
                        'pc1',
                        'pc2',
                        source=datasource1,
                        color=color_mapping1.palette[i],#dict(field='label', transform=color_mapping),
                        line_alpha=0.6,
                        fill_alpha=0.6,
                        size=4,
                        legend=color_mapping1.factors[i]#'label'
                    )
                    plot_figure_left.legend.location = "top_left"
                    plot_figure_left.legend.click_policy="hide"
                    for i,j in enumerate(labels_unique2):
                        datasource2=ColumnDataSource(digits_df2[digits_df2['label']==j])
                        plot_figure_right.circle(
                        'pc1',
                        'pc2',
                        source=datasource2,
                        color=color_mapping2.palette[i],#dict(field='label', transform=color_mapping),
                        line_alpha=0.6,
                        fill_alpha=0.6,
                        size=4,
                        legend=color_mapping2.factors[i]#'label'
                    )
                    plot_figure_right.legend.location = "top_left"
                    plot_figure_right.legend.click_policy="hide"
                    output_file("interactive_legend.html", title="interactive_legend.py example")
                st.bokeh_chart(gridplot([[plot_figure_left,plot_figure_right]]))
            if num_components==2 and st.button('Display T-SNE'):
                if uploaded_file1 is not None:
                    if feature1 is not None:
                        TSNE_embedded1 = tsne.fit(feature1)
                        tsne1=np.array(TSNE_embedded1)
                if uploaded_file2 is not None:
                    if feature2 is not None:
                        TSNE_embedded2 = tsne.fit(feature2)
                        tsne2=np.array(TSNE_embedded2)
                if uploaded_file1 is not None and path1 is not None:
                    image_np1=save_image_to_array(path1)
                if uploaded_file2 is not None and path2 is not None:
                    image_np2=save_image_to_array(path2)
                plot_tsne_double_2d(image_np1,image_np2,tsne1,tsne2,labels1,labels2,n_components = num_components)