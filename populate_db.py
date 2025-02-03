"""Script for populating db with initial dataset using elasticsearch as db"""

from src.model import EmbeddingModel, DetectionModel, TextDetectionModel
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
    try:
        elastic_db.create_index()
    except:
        pass
    # Init logo det
    text_det = TextDetectionModel()
    
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
            
            # if len(categorized_bboxes['labels']) == 1:
            #     for indx, label in enumerate(categorized_bboxes['labels']):
            #         bbox = categorized_bboxes['boxes'][indx]
            #         cropped_image = image.crop(bbox)
                    
            #         detected_logos = logo_det.detect_logos(cropped_image)
                    
                    
            #         local_image_embeddings = model.embed_images_from_list([cropped_image])[0].cpu().tolist()
                    
            #         elastic_db.insert_index(
            #             filepath,
            #             local_image_embeddings,
            #             label,
            #             str(bbox),
            #             True
            #         )
            if True:
                probs = model.get_category_probs(image, category_tokenized)
                target_category = categories[np.argmax(probs)]
                vector_combination = image_embeddings[j].cpu().tolist()
                
                detected_text = text_det.get_text(image)
                
                text_embedding_vector = None
                if detected_text:
                    detected_text = ' '.join(detected_text)
                    try:
                        text_embedding_vector = model.get_text_embeddings(model.tokenize_text(
                                f"{target_category}: {detected_text}"
                            )).cpu().tolist()[0]
                    except RuntimeError:
                        text_embedding_vector = model.get_text_embeddings(model.tokenize_text(
                                f"{target_category}: {detected_text}"[:100]
                            )).cpu().tolist()[0]
                    
                    # Trying avg
                    vector_combination =((np.array(vector_combination) + np.array(text_embedding_vector)) / 2).tolist()
                
                
                elastic_db.insert_index(
                    filepath=filepath,
                    image_vector=image_embeddings[j].cpu().tolist(),
                    category=target_category,
                    bbox="None",
                    is_single_product=False,
                    text_vector=text_embedding_vector,
                    vector_combination=vector_combination
                )
                for indx, label in enumerate(categorized_bboxes['labels']):
                    bbox = categorized_bboxes['boxes'][indx]
                    cropped_image = image.crop(bbox)
                    
                    local_image_embeddings = model.embed_images_from_list([cropped_image])[0].cpu().tolist()
                    vector_combination = local_image_embeddings
                    
                    detected_text = text_det.get_text(np.array(cropped_image))
                    
                    text_embedding_vector = None
                    if detected_text:
                        detected_text = ' '.join(detected_text)
                        try:
                            text_embedding_vector = model.get_text_embeddings(model.tokenize_text(
                                f"{label}: {detected_text}"
                            )).cpu().tolist()[0]
                        except RuntimeError:
                            text_embedding_vector = model.get_text_embeddings(model.tokenize_text(
                                f"{label}: {detected_text}"[:100]
                            )).cpu().tolist()[0]
                        # Trying avg
                        vector_combination =((np.array(vector_combination) + np.array(text_embedding_vector)) / 2).tolist()
                    # logo_bboxes = logo_det.detect_logos(cropped_image)
                    # logo_image_embedding = None
                    # if logo_bboxes:
                    #     logo_box = (
                    #         min(arr['box']['xmin'] for arr in logo_bboxes),
                    #         min(arr['box']['ymin'] for arr in logo_bboxes),
                    #         max(arr['box']['xmax'] for arr in logo_bboxes),
                    #         max(arr['box']['ymax'] for arr in logo_bboxes)
                    #     )
                    #     logo_cropped = image.crop(logo_box)
                    #     logo_image_embedding = model.embed_images_from_list([logo_cropped])[0].cpu().tolist()
                        
                    elastic_db.insert_index(
                        filepath=filepath,
                        image_vector=local_image_embeddings,
                        category=label,
                        bbox=str(bbox),
                        is_single_product=len(categorized_bboxes['labels']) == 1,
                        text_vector=text_embedding_vector,
                        vector_combination=vector_combination
                    )
            

if __name__ == "__main__":
    main()