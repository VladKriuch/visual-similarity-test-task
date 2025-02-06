import json
import os
import ast

import numpy as np

from src.models.detection import DetectionModel
from src.models.embedding import EmbeddingModel
from src.elastic_db import ElasticDB
from src.models.ocr import TextDetectionModel

from dotenv import load_dotenv

from PIL import Image

class PipelineHandler:    
    """Class for general pipeline handler"""
    def __init__(self, categories_filepath,):        
        load_dotenv()
        
        with open(categories_filepath, "r") as f:
            self.categories = json.load(f)
        self.detection_model = DetectionModel(reclassification_dict=self.categories)
        self.embedding_model = EmbeddingModel()
        self.ocr_model = TextDetectionModel()
        
        self.category_tokenized = self.embedding_model.tokenize_text(list(self.categories.keys()))
        self.elastic_db = ElasticDB(index_name="visual-search-test")
        
        self.state_reset()
    
    def state_reset(self):
        self.active_image = None
        self.h_ratio = None
        self.w_ratio = None
        self.canvas_height = None
        self.canvas_width = None
        
        self.json_data = None
        self.detected_regions = None
        self.centers = []
        
    def set_active_image(self, image, canvas_height, canvas_width):
        self.state_reset()
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        
        self.active_image = image.copy()       
        self.h_ratio = self.canvas_height / image.height
        self.w_ratio = self.canvas_width / image.width
        
    def image_exists(self, image):
        return self.active_image is not None and np.shape(image) == np.shape(self.active_image) and np.all(np.array(image) == np.array(self.active_image))
    
    def roi_detection(self):
        self.detected_regions = self.detection_model.find_points_of_interest(self.active_image)
        self.centers = []
        for box in self.detected_regions['boxes']:
            x_center = (box[0] + box[2]) // 2
            y_center = (box[1] + box[3]) // 2
            self.centers.append((x_center, y_center))

    def perform_search(self, category_option: str, full_images_search: bool):
        is_self_drawn = True
        # Scale and crop
        object_data = self.json_data['objects'][-1]
        min_x, min_y, max_x, max_y = object_data['left'] * (1 / self.w_ratio), object_data['top'] * (1 / self.h_ratio), (object_data['width'] + object_data['left']) * (1 / self.w_ratio), (object_data['height'] + object_data['top']) * (1 / self.h_ratio)
        
        target_bbox = None
        if (max_x - min_x) < 35 or (max_y - min_y) < 35:
            is_self_drawn = False 
            center_x = (max_x + min_x) // 2
            center_y = (max_y + min_y) // 2
            
            l2 = lambda cx, cy, tx, ty: np.sqrt((cx - tx) ** 2) + np.sqrt((cy - ty) ** 2)
            for i, center in enumerate(self.centers):
                if l2(center_x, center_y, center[0], center[1]) < 35:
                    target_bbox = i
                    break
                
        
        if is_self_drawn:
            im_cropped = self.active_image.crop((int(min_x), int(min_y), int(max_x), int(max_y)))
            
            self.im_cropped = im_cropped
            
            # Get image embedding
            image_vector = self.embedding_model.embed_images_from_list([im_cropped])[0].cpu()
            
            if category_option == "Detect":
                # Detect category
                probs = self.embedding_model.get_category_probs(im_cropped, self.category_tokenized)
                target_category = list(self.categories.keys())[np.argmax(probs)]
            else:
                target_category = category_option
            
            detected_text = self.ocr_model.get_text(np.array(im_cropped))
            if detected_text:
                try:
                    text_embedding_vector = self.embedding_model.embed_text_from_tokens(self.embedding_model.tokenize_text(
                                    f"{target_category}: {detected_text}"
                                )).cpu()[0].detach().numpy()
                except RuntimeError:
                    text_embedding_vector = self.embedding_model.embed_text_from_tokens(self.embedding_model.tokenize_text(
                                    f"{target_category}: {detected_text}"[:100]
                                )).cpu()[0].detach().numpy()

                vector_combination = np.concatenate((image_vector.detach().numpy(), text_embedding_vector))
            else:
                vector_combination = image_vector
            # Perform search
            search_results = self.elastic_db.search(
                vector_combination.tolist(),
                label=target_category,
                full_image_search=full_images_search
            )
            
            # Bring results to a suitable format
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
        elif target_bbox is not None:
            min_x, min_y, max_x, max_y = self.detected_regions['boxes'][i]
            im_cropped = self.active_image.crop((int(min_x), int(min_y), int(max_x), int(max_y)))
            
            self.im_cropped = im_cropped
            
            # Get image embedding
            image_vector = self.embedding_model.embed_images_from_list([im_cropped])[0].cpu()
            
            # Detect category
            if category_option == "Detect":
                target_category = self.detected_regions['labels'][i]
            else:
                target_category = category_option
            
            detected_text = self.ocr_model.get_text(np.array(im_cropped))
            if detected_text:
                try:
                    text_embedding_vector = self.embedding_model.embed_text_from_tokens(self.embedding_model.tokenize_text(
                                    f"{target_category}: {detected_text}"
                                )).cpu()[0].detach().numpy()
                except RuntimeError:
                    text_embedding_vector = self.embedding_model.embed_text_from_tokens(self.embedding_model.tokenize_text(
                                    f"{target_category}: {detected_text}"[:100]
                                )).cpu()[0].detach().numpy()

                vector_combination = np.concatenate((image_vector.detach().numpy(), text_embedding_vector))
            else:
                vector_combination = image_vector
            
            # Perform search
            search_results = self.elastic_db.search(
                vector_combination.tolist(),
                label=target_category,
                full_image_search=full_images_search
            )
            
            # Bring results to a suitable format
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
        return None
            

        