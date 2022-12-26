import streamlit as st
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_notebook,output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10,Inferno10,Paired3,Category10_10
#import h5py
from io import BytesIO
import io
from PIL import Image
import base64
import cv2
from skimage import io
import os
from openTSNE import TSNE
import torch
def save_image_to_array(path):
            img_list = []
            for dir_image in path:
                img = cv2.cvtColor(cv2.imread(dir_image), cv2.COLOR_BGR2RGB)
                img_list.append(img)
            img_np = np.array(img_list)
            return(img_np)
