import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image

if __name__ == "__main__":
    # set page config
    st.set_page_config(page_title="James Webb Space Telescope vs Hubble Telescope Images", layout="centered")

    st.title("James Webb vs Hubble Telescope Pictures")

    st.markdown("# Select Image")

    st.markdown("Golden img: ")
    bottom_image = st.file_uploader('', type=['jpg','png'],key=200)
    st.markdown("Reference img: ")
    bottom_image = st.file_uploader('', type=['jpg','png'],key=201)


    st.markdown("# Southern Nebula")

    # render image-comparison
    image_comparison(
        img1="https://www.webbcompare.com/img/hubble/southern_nebula_700.jpg",
        img2="https://www.webbcompare.com/img/webb/southern_nebula_700.jpg",
        label1="Hubble",
        label2="Webb"
    )



# bottom_image = st.file_uploader('', type='jpg', key=6)
# if bottom_image is not None:
#     image = Image.open(bottom_image)
#     new_image = image.resize((600, 400))
#     st.image(new_image)