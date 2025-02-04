import io
import os
import json

import streamlit as st
import numpy as np

from PIL import Image, ImageDraw
from dotenv import load_dotenv
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas

from pipeline import PipelineHandler

def draw_search_results(search_results):
    st.write("### Search Results")
                        
    col1, col2, col3 = st.columns([1, 3, 5])
    with col1:
        st.write("**Image**")
    with col2:
        st.write("**Score**")
    with col3:
        st.write("**Parent Image**")
    for img, score, parent_image in search_results:
        col1, col2, col3 = st.columns([1, 3, 5])
        with col1:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.image(buf)
        with col2:
            st.write(score)
        with col3:
            if parent_image is not None:
                buf2 = io.BytesIO()
                parent_image.save(buf2, format="PNG")
                st.image(buf2)
                
if __name__ == "__main__":
    # set some defaults
    CATEGORIES_FILE = "static/CATEGORIZATION.json"
    
    # Set everything up
    st.title("Image Search App")
    
    if "pipeline_handler" not in st.session_state:
        st.session_state.pipeline_handler = PipelineHandler(CATEGORIES_FILE)
    
    if "category_filter" not in st.session_state:
        with open(CATEGORIES_FILE, "r") as f:
            categories = list(json.load(f).keys())
        categories = ["Detect", "All"] + categories
        st.session_state.filter_category = st.selectbox(
            "Choose category or let the model detect it",
            categories,
        )
    
    st.session_state.full_images_search_only = st.radio(
            "Include single product images only",
            ["Yes", "No"],
            key="full_im_only"
        )
        
    # Make utility for file upload
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        thumbnail = image.copy()
        
        thumbnail.thumbnail((640, 640))
        
        pipeline_handler = st.session_state.pipeline_handler
        
        if not pipeline_handler.image_exists(image):
            with st.spinner("Processing image..."):
                pipeline_handler.set_active_image(image, thumbnail.height, thumbnail.width)
                pipeline_handler.roi_detection()
        
        draw = ImageDraw.Draw(image)
        for x_center, y_center in pipeline_handler.centers:
            draw.ellipse((x_center - 7, y_center - 7, x_center + 7, y_center + 7), fill='red')
            
        # Create canvas (ONLY rectangle mode)
        canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red fill
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=image,
                update_streamlit=True,
                drawing_mode="rect",  # LOCKED to rectangle mode
                key="bbox_canvas",
                height=thumbnail.height,
                width=thumbnail.width
            )
        
        if pipeline_handler.json_data != canvas_result.json_data:
            pipeline_handler.json_data = canvas_result.json_data

            if canvas_result.json_data is not None and canvas_result.json_data['objects']:
                with st.spinner("Searching..."):
                    results = pipeline_handler.perform_search(
                        st.session_state.filter_category,
                        st.session_state.full_im_only == "Yes"
                    )
                    if results is not None:
                        draw_search_results(results)
                    else:
                        st.error('Your bounding box is invalid, please try bigger bounding box or choose one of the proposed points', icon="🚨")

        