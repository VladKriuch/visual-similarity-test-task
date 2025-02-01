"""Script for populating db with initial dataset using elasticsearch as db"""

from src.model import EmbeddingModel
from src.elastic_db import ElasticDB

import glob
import os

from dotenv import load_dotenv
from tqdm import tqdm

def main():
    # Init model
    model = EmbeddingModel()
    
    # Init db
    load_dotenv()
    elastic_api_key = os.getenv('ELASTIC_API_KEY')
    
    elastic_db = ElasticDB(elastic_api_key)
    elastic_db.create_index()
    
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
            elastic_db.insert_index(
                filepath,
                image_embeddings[j].cpu().tolist()
            )

if __name__ == "__main__":
    main()