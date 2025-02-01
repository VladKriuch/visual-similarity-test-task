import time
import random
import io
import os

import streamlit as st

from PIL import Image
from dotenv import load_dotenv

from src.elastic_db import ElasticDB
from src.model import EmbeddingModel

def perform_search(image: Image.Image):
    """Mock function to simulate searching and returning images with scores."""
    query_vector = st.session_state.model.embed_images_from_list([image])[0].cpu().tolist()
    search_results = st.session_state.db.search(query_vector)
    
    results = [(Image.open(search_result['_source']['filepath']), search_result['_score']) for search_result in search_results]
    
    return results

if __name__ == "__main__":
    # Set everything up
    st.title("Image Search App")

    if "model" not in st.session_state:
        st.session_state.model = EmbeddingModel()
    
    if "db" not in st.session_state:
        load_dotenv()
        st.session_state.db = ElasticDB(os.getenv("ELASTIC_API_KEY"))

    # Make utility for file upload
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Processing image..."):
            results = perform_search(image)
        
        st.write("### Search Results")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("**Image**")
        with col2:
            st.write("**Score**")
        
        for img, score in results:
            col1, col2 = st.columns([1, 3])
            with col1:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.image(buf, width=100)
            with col2:
                st.write(score)
