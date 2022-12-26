import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import torch
import pandas as pd
from skimage.segmentation import mark_boundaries
import os,sys
import lime
from lime import lime_image
from lime.lime_image import LimeImageExplainer
import matplotlib
def lime_image():
    st.header('Image explaination with LIME')
    myfile=st.sidebar.file_uploader('Choose a file(image list)')
    if myfile is not None:
        df=torch.load(myfile)
        path=df['path']
        inet_model = inc_net.InceptionV3()
        def transform_img_fn(path_list):
            out = []
            for img_path in path_list:
                img = keras.utils.load_img(img_path, target_size=(299, 299))
                x = keras.utils.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = inc_net.preprocess_input(x)
                out.append(x)
            return np.vstack(out)
        images=transform_img_fn(path)
        index=st.sidebar.selectbox('Choose the image to show',np.arange(len(path)))
        fig,ax=plt.subplots()
        im=ax.imshow(images[index]/2+0.5)
        st.write('''## original image''')
        st.pyplot(fig)
        preds = inet_model.predict(torch.unsqueeze(torch.tensor(images[index]),dim=0).numpy())
        df=pd.DataFrame(decode_predictions(preds)[0],columns=['no','class','prob'])
        st.write('''## predict''')
        st.dataframe(df.iloc[:,1:])
        top_index=st.sidebar.selectbox('Choose the top labels',[1,2,3,4,5])
        explainer =LimeImageExplainer()
        explanation = explainer.explain_instance(images[index].astype('double'), inet_model.predict, top_labels=5, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[top_index-1], positive_only=True, num_features=5, hide_rest=True)
        fig,ax=plt.subplots()
        im=ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        st.write('''## LIME explaination 1
                    positive only and hide rest''')
        st.pyplot(fig)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[top_index-1], positive_only=True, num_features=5, hide_rest=False)
        fig,ax=plt.subplots()
        im=ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        st.write('''## LIME explaination 2
                    positive only''')
        st.pyplot(fig)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[top_index-1], positive_only=False, num_features=10, hide_rest=False)
        fig,ax=plt.subplots()
        im=ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        st.write('''## LIME explaination 3
                    positive and negative''')
        st.pyplot(fig)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[top_index-1], positive_only=False, num_features=1000, hide_rest=False,min_weight=0.1)
        fig,ax=plt.subplots()
        im=ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        st.write('''## LIME explaination 4
                    positive and negative and threshold=0.1''')
        st.pyplot(fig)

        ind =  explanation.top_labels[top_index-1]
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
        fig,ax=plt.subplots()
        im=ax.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
        cNorm = matplotlib.colors.Normalize(vmin=-heatmap.max(), vmax=heatmap.max())
        fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu',norm=cNorm),ax=ax)
        st.write('''## LIME explaination 5
                    heatmap''')
        st.pyplot(fig)