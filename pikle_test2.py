import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from skimage import color
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import streamlit as st
from streamlit_image_comparison import image_comparison

from PIL import Image
import pathlib

import dill

def main(input_path,reference_path,n,window_size, pca_dim_gray, pca_dim_rgb,
          lighting_fix, use_homography, resize_factor):
    
    with open('pre_processing.pkl','rb') as f:
        pre_processing = dill.load(f)
        
    with open('cluster.pkl','rb') as f:
        cluster = dill.load(f)
        
    processing = pre_processing(input_path=input_path,
                  reference_path=reference_path,
                  resize_factor=resize_factor, 
                  use_homography=use_homography,
                  lighting_fix=lighting_fix)
    image_1, image2_registered, new_shape = processing.processing()
    
    clust = cluster(input_path=input_path,
                  n=n,
                  window_size=window_size, 
                  pca_dim_gray=pca_dim_gray,
                  pca_dim_rgb=pca_dim_rgb)
    clust.setParam(image_1,image2_registered,new_shape)
    
    return clust.clustering()
    

    
if __name__ == '__main__':
    input_path = "img/golden.jpg"
    reference_path = "img/diff1.jpg"
    n=10
    window_size = 5
    pca_dim_gray = 3
    pca_dim_rgb = 9
    
    lighting_fix = True
    use_homography = True
    resize_factor = float(1)
    
    # set page config
    st.set_page_config(page_title="James Webb Space Telescope vs Hubble Telescope Images", layout="centered")

    st.title("James Webb vs Hubble Telescope Pictures")
    st.markdown("# Select Image")
    
    img_dir_name = 'img'
    img_dir = pathlib.Path(img_dir_name)
    
    col1, col2 = st.columns(2)
    with col1:
        golden_img = st.file_uploader('Golden img', type=['jpg','png'],key=200)
        if golden_img is not None:
            #input_path = str(img_dir / golden_img.name)
            golden_img = Image.open(golden_img)
            #input_path = cv2.cvtColor(np.array(golden_img), cv2.COLOR_RGB2BGR)
            
    with col2:
        reference_img = st.file_uploader('Reference img:', type=['jpg','png'],key=201)
        if reference_img is not None:
            #reference_path = str(img_dir / reference_img.name)
            reference_img = Image.open(reference_img)
            #reference_path = cv2.cvtColor(np.array(reference_img), cv2.COLOR_RGB2BGR)
    
    
    #n = st.sidebar.slider("cluster class Number", [10,30])
    
    
    
    if golden_img is not None and reference_img is not None:
        if st.button('inspection'):
            result = main(input_path, reference_path, int(n), int(window_size),
                int(pca_dim_gray), int(pca_dim_rgb), bool(lighting_fix), bool(use_homography),
                float(resize_factor))
            
            golden_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            #cv2.imwrite('ACCEPTED_CLASSES6'+'.png', result)
            
        
        image_comparison(
            img1=golden_img,
            img2=reference_img,
            label1="Golden",
            label2="Reference"
        )
    
    
    # if bottom_image is not None:
    #     bottom_image.
    #     print(bottom_image)
    #     # image = cv2.imread(bottom_image)
    #     # color_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     # pil_src = Image.fromarray(color_cvt)
        
    #     #image = Image.open(bottom_image)
    #     #new_image = image.resize((600, 400))
    #     # st.image(pil_src)
    

    # result = main(input_path, reference_path, int(n), int(window_size),
    #      int(pca_dim_gray), int(pca_dim_rgb), bool(lighting_fix), bool(use_homography),
    #      float(resize_factor))
    
    
    
