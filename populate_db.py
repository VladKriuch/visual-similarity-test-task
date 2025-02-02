"""Script for populating db with initial dataset using elasticsearch as db"""

from src.model import EmbeddingModel, DetectionModel
from src.elastic_db import ElasticDB

import glob
import os
import json
import numpy as np

from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image

def main():
    # Init model
    model = EmbeddingModel()
    
    # Init db
    load_dotenv()
    elastic_api_key = os.getenv('ELASTIC_API_KEY')
    
    elastic_db = ElasticDB(elastic_api_key)
    elastic_db.create_index()
    
    # Load detection model
    with open("static/CATEGORIZATION.json", "r") as f:
        categors = json.load(f)
        detection_model = DetectionModel(
            categors
        )
        categories = list(categors.keys())
    category_tokenized = model.tokenize_text(categories)
    # some static vars
    BATCH_SIZE = 8
    MAIN_FOLDER_PATH = "test_task_data/test_task_data"
    
    # Algorithm for populating db
    image_paths = glob.glob(os.path.join(MAIN_FOLDER_PATH, "*.jpg"))
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), total=total_batches, desc="Populating db with image embeddings"):
        batch = image_paths[i:i+BATCH_SIZE]
        image_embeddings = model.embed_images_from_paths(batch)
        
        for j, filepath in enumerate(batch):
            categorized_bboxes = detection_model.find_points_of_interest(filepath)
            image = Image.open(filepath)
            
            if len(categorized_bboxes['labels']) == 1:
                for indx, label in enumerate(categorized_bboxes['labels']):
                    bbox = categorized_bboxes['boxes'][indx]
                    cropped_image = image.crop(bbox)
                    local_image_embeddings = model.embed_images_from_list([cropped_image])[0].cpu().tolist()
                    
                    elastic_db.insert_index(
                        filepath,
                        local_image_embeddings,
                        label,
                        str(bbox),
                        True
                    )
            else:
                probs = model.get_category_probs(image, category_tokenized)
                target_category = categories[np.argmax(probs)]
                elastic_db.insert_index(
                    filepath,
                    image_embeddings[j].cpu().tolist(),
                    target_category,
                    "None",
                    True
                )
                for indx, label in enumerate(categorized_bboxes['labels']):
                    bbox = categorized_bboxes['boxes'][indx]
                    cropped_image = image.crop(bbox)
                    local_image_embeddings = model.embed_images_from_list([cropped_image])[0].cpu().tolist()
                    
                    elastic_db.insert_index(
                        filepath,
                        local_image_embeddings,
                        label,
                        str(bbox),
                        False
                    )
            

if __name__ == "__main__":
    main()