"""Script for populating db with initial dataset using elasticsearch as db"""

from src.elastic_db import ElasticDB
from src.models.detection import DetectionModel
from src.models.embedding import EmbeddingModel
from src.models.ocr import TextDetectionModel

import os
import json
import numpy as np
import torch
import argparse

from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image

from imagededup.methods import PHash

def filter_duplicates(image_dir):
    phasher = PHash()
    
    encodings = phasher.encode_images(image_dir=image_dir)
    duplicates = phasher.find_duplicates(encoding_map=encodings)
    
    duplicates_orig = []
    duplicates_to_filter = []
    
    for key, item in duplicates.items():
        if key not in duplicates_to_filter:
            duplicates_orig.append(key)
            duplicates_to_filter.extend(item)
    
    return [key for key in duplicates if key not in duplicates_to_filter]
    

def main(static_folder, path_to_data):
    # some static vars
    BATCH_SIZE = 8
    MAIN_FOLDER_PATH = path_to_data
    
    # Init db
    load_dotenv()
    
    elastic_db = ElasticDB()
    index_created = elastic_db.create_index()
    if not index_created:
        return
    
    # Init Embedding model
    model = EmbeddingModel()
    
    # Init OCR model
    ocr_model = TextDetectionModel()
    
    # Load detection model
    with open(static_folder + "/CATEGORIZATION.json", "r") as f:
        categors = json.load(f)
        detection_model = DetectionModel(
            reclassification_dict=categors
        )
        categories = list(categors.keys())
    category_tokenized = model.tokenize_text(categories)
    
    # Algorithm for populating db
    image_paths = [path_to_data + "/" + filtered_impath for filtered_impath in filter_duplicates(MAIN_FOLDER_PATH)[:500]]
    # image_paths = detect_nsfw(image_paths)
    
    # Clear after
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), total=total_batches, desc="Populating db with image embeddings"):
        batch = image_paths[i:i+BATCH_SIZE]
        image_embeddings = model.embed_images_from_paths(batch)
        
        for j, filepath in enumerate(batch):
            categorized_bboxes = detection_model.find_points_of_interest(filepath)
            image = Image.open(filepath)
            
            if True:
                probs = model.get_category_probs(image, category_tokenized)
                target_category = categories[np.argmax(probs)]
                
                detected_text = ocr_model.get_text(image)
                detected_text = target_category + ": " + ' '.join(detected_text)
                try:
                    text_embedding_vector = model.embed_text_from_tokens(model.tokenize_text(
                                f"{target_category}: {detected_text}"
                            )).cpu()[0].detach().numpy()
                except RuntimeError:
                    text_embedding_vector = model.embed_text_from_tokens(model.tokenize_text(
                                f"{target_category}: {detected_text}"[:100]
                            )).cpu()[0].detach().numpy()
                
                vector_combination = np.concatenate((image_embeddings[j].cpu().detach().numpy(), text_embedding_vector))
                
                elastic_db.insert_index(
                    filepath=filepath,
                    image_vector=image_embeddings[j].cpu().tolist(),
                    category=target_category,
                    bbox="None",
                    is_single_product=False,
                    text_vector=text_embedding_vector.tolist(),
                    vector_combination=vector_combination.tolist()
                )
                for indx, label in enumerate(categorized_bboxes['labels']):
                    bbox = categorized_bboxes['boxes'][indx]
                    cropped_image = image.crop(bbox)
                    
                    local_image_embeddings = model.embed_images_from_list([cropped_image])[0].cpu()
                    detected_text = ocr_model.get_text(image)
                    detected_text = target_category + ": " + ' '.join(detected_text)
                    try:
                        text_embedding_vector = model.embed_text_from_tokens(model.tokenize_text(
                                    f"{target_category}: {detected_text}"
                                )).cpu()[0].detach().numpy()
                    except RuntimeError:
                        text_embedding_vector = model.embed_text_from_tokens(model.tokenize_text(
                                    f"{target_category}: {detected_text}"[:100]
                                )).cpu()[0].detach().numpy()
                
                    
                    vector_combination = np.concatenate((image_embeddings[j].cpu().detach().numpy(), text_embedding_vector))
                    
                    elastic_db.insert_index(
                        filepath=filepath,
                        image_vector=local_image_embeddings.tolist(),
                        category=label,
                        bbox=str(bbox),
                        is_single_product=len(categorized_bboxes['labels']) == 1,
                        text_vector=text_embedding_vector.tolist(),
                        vector_combination=vector_combination.tolist()
                    )
            

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process a JSON dictionary and a data file path.')
    parser.add_argument('static_folder', type=str, help='Path to the JSON dictionary file')
    parser.add_argument('data_path', type=str, help='Path to the data file')
    
    # Parse arguments
    args = parser.parse_args()

    main(parser.static_folder, parser.data_path)