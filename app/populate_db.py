"""Script for populating db with initial dataset using elasticsearch as db"""

from src.elastic_db import ElasticDB
from src.models.detection import DetectionModel
from src.models.embedding import EmbeddingModel

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
    # Exclude corrupted files
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
    
    
    # Load detection model
    with open(os.path.join(static_folder, "CATEGORIZATION.json"), "r") as f:
        categors = json.load(f)
        detection_model = DetectionModel(
            reclassification_dict=categors
        )
        categories = list(categors.keys())
    category_tokenized = model.tokenize_text(categories)
    
    # Algorithm for populating db
    image_paths = [os.path.join(path_to_data, filtered_impath) for filtered_impath in filter_duplicates(MAIN_FOLDER_PATH)]
    
    # Clear after
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), total=total_batches, desc="Populating db with image embeddings"):
        print(f"Processing Batch {i} from {total_batches}")
        batch = image_paths[i:i+BATCH_SIZE]
        image_embeddings = model.embed_images_from_paths(batch)
        
        for j, filepath in enumerate(batch):
            categorized_bboxes = detection_model.find_points_of_interest(filepath)
            image = Image.open(filepath)
            
            if True:
                probs = model.get_category_probs(image, category_tokenized)
                target_category = categories[np.argmax(probs)]
                
                elastic_db.insert_index(
                    filepath=filepath,
                    image_vector=image_embeddings[j].cpu().tolist(),
                    category=target_category,
                    bbox="None",
                    is_single_product=False,
                )
                for indx, label in enumerate(categorized_bboxes['labels']):
                    bbox = categorized_bboxes['boxes'][indx]
                    cropped_image = image.crop(bbox)
                    
                    local_image_embeddings = model.embed_images_from_list([cropped_image])[0].cpu()
                    
                    elastic_db.insert_index(
                        filepath=filepath,
                        image_vector=local_image_embeddings.tolist(),
                        category=label,
                        bbox=str(bbox),
                        is_single_product=len(categorized_bboxes['labels']) == 1,
                    )
            

if __name__ == "__main__":
    # Set up argument parser
    load_dotenv()

    main("./static", "/data")