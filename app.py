import time
import random
import io
import os
import json
import ast

import streamlit as st

from PIL import Image, ImageDraw
from dotenv import load_dotenv
from streamlit_image_coordinates import streamlit_image_coordinates

from src.elastic_db import ElasticDB
from src.model import EmbeddingModel, DetectionModel

def perform_search(image: Image.Image, labels, box, is_single_objects_only):
    """Mock function to simulate searching and returning images with scores."""
    if box is not None:
        query_box_image = image.crop(box)
        query_vector = st.session_state.model.embed_images_from_list([query_box_image])[0].cpu().tolist()
        
        search_results = st.session_state.db.search(query_vector, labels, single_product_only=is_single_objects_only)
        
        results = []
        for res in search_results:
            image = Image.open(res['_source']['filepath'])
            bbox = res['_source']['bbox']

            
            if bbox != "None":
                bbox = ast.literal_eval(res['_source']['bbox'])
                local_image = image.crop(bbox)
                results.append((
                    local_image, res['_score'], image, 
                ))
            else:
                results.append((
                    image, res['_score'], None, 
                ))
    else:
        query_vector = st.session_state.model.embed_images_from_list([image])[0].cpu().tolist()
        search_results = st.session_state.db.search(query_vector, None, single_product_only=is_single_objects_only)
        
        results = []
        for res in search_results:
            image = Image.open(res['_source']['filepath'])
            bbox = res['_source']['bbox']

            
            if bbox != "None":
                bbox = ast.literal_eval(res['_source']['bbox'])
                local_image = image.crop(bbox)
                results.append((
                    local_image, res['_score'], image, 
                ))
            else:
                results.append((
                    image, res['_score'], None, 
                ))
    
    return results


def draw_points(image, boxes):
    """Draw central points of bounding boxes on the image."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    centers = []
    for box in boxes:
        x_center = (box[0] + box[2]) // 2
        y_center = (box[1] + box[3]) // 2
        centers.append((x_center, y_center))
        draw.ellipse((x_center - 5, y_center - 5, x_center + 5, y_center + 5), fill='red')
    return image, centers


if __name__ == "__main__":
    # Set everything up
    st.title("Image Search App")

    if "model" not in st.session_state:
        st.session_state.model = EmbeddingModel()
    
    if "db" not in st.session_state:
        load_dotenv()
        st.session_state.db = ElasticDB(os.getenv("ELASTIC_API_KEY"))

    if "detection_model" not in st.session_state:
        with open("static/CATEGORIZATION.json", "r") as f:
            st.session_state.detection_model =  DetectionModel(
                json.load(f)
            )
            
    st.radio(
        "Include single product images only",
        ["Yes", "No"],
        key="single_product_only",
        # on_change=lambda: setattr(st.session_state, 'single_product_only', st.session_state.single_product_only == "Yes")
    )

    use_full_image_button = st.button("Search on full image")
    # Make utility for file upload
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    # Reset session state if a new file is uploaded
    if uploaded_file:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.detection_results = None
            st.session_state.search_results = None

    if uploaded_file:
        image = Image.open(uploaded_file)

        # Run detection only if not already stored
        if st.session_state.detection_results is None:
            with st.spinner("Processing image..."):
                results_dict = st.session_state.detection_model.find_points_of_interest(image)
                st.session_state.detection_results = results_dict
        else:
            results_dict = st.session_state.detection_results

        # Draw detection points
        image_with_points, centers = draw_points(image, results_dict["boxes"])

        # Display image with clickable points
        coords = streamlit_image_coordinates(image_with_points, key="clickable_image")

        if coords:
            x, y = coords["x"], coords["y"]
            for i, box in enumerate(results_dict["boxes"]):
                x_center = (box[0] + box[2]) // 2
                y_center = (box[1] + box[3]) // 2
                if abs(x - x_center) < 35 and abs(y - y_center) < 35:
                    with st.spinner("Searching for similar images..."):
                        search_results = perform_search(image, results_dict['labels'][i], box, st.session_state.single_product_only == "Yes")
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
                                    st.image(buf, width=100)
                                with col2:
                                    st.write(score)
                                with col3:
                                    if parent_image is not None:
                                        buf2 = io.BytesIO()
                                        parent_image.save(buf2, format="PNG")
                                        st.image(buf2, width=100)
                    break
        elif use_full_image_button:
            with st.spinner("Searching for similar images..."):
                search_results = perform_search(image, None, None, st.session_state.single_product_only == "Yes")
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
                        st.image(buf, width=100)
                    with col2:
                        st.write(score)
                    with col3:
                        if parent_image is not None:
                            buf2 = io.BytesIO()
                            parent_image.save(buf2, format="PNG")
                            st.image(buf2, width=100)

