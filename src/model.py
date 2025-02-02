import torch
import clip
from PIL import Image
from ultralytics import YOLO
from copy import deepcopy

class DetectionModel:
    def __init__(self, classification_dict={}, model_type="yolov8x-oiv7", force_cpu=False):
        self.device = "cuda" if torch.cuda.is_available() and force_cpu is not True else "cpu"
        self.model = YOLO(model_type)
        self.classification_dict = classification_dict
    
    def find_points_of_interest(self, image):
        boxes = self.get_boxes(image)
        return self.reclassify_results(boxes)
        
    def get_boxes(self, image):
        results = self.model(image, conf=0.1)
        
        results_dict = {
            'labels': [],
            'boxes': [],
            'conf': []
        }
        for results in results:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                label = self.model.names[cls]
                
                results_dict['labels'].append(label)
                results_dict['conf'].append(conf)
                results_dict['boxes'].append((x1, y1, x2, y2))
        
        return results_dict
    
    def reclassify_results(self, results):
        results = deepcopy(results)
        updated_results = {
            'labels': [],
            'boxes': [],
            'conf': []
        }
        
        for indx, label in enumerate(results['labels']):
            for class_name, class_values in self.classification_dict.items():
                if label in class_values:
                    updated_results['labels'].append(class_name)
                    updated_results['boxes'].append(results['boxes'][indx])
                    updated_results['conf'].append(results['conf'][indx])
                    break
        return updated_results
        
class EmbeddingModel:
    def __init__(self, model_type="ViT-B/32", force_cpu=False):
        """Class for generating embeddings for images/texts

        Args:
            model_type (str, optional): Model type provided for clip. Defaults to "ViT-B/32".
            force_cpu (bool, optional): Whether to force cpu. If True, the device will be forced to use cpu. Defaults to False.
        """
        self.device = "cuda" if torch.cuda.is_available() and force_cpu is not True else "cpu"
        self.model, self.model_preprocess = clip.load(model_type, device=self.device)
        
    def embed_images_from_paths(self, img_paths: list):
        """Function for creating embeddings for img_paths
        Returns the type that self.model.encode_image returns

        Args:
            img_paths (list): List of paths to images
        """
        images = [self.model_preprocess(Image.open(impath)).unsqueeze(0).to(self.device) for impath in img_paths]
        images = torch.cat(images, dim=0)

        return self.get_image_embeddings(images)
    
    def embed_images_from_list(self, images: list):
        """Function for geenrating embeddings for a list of images

        Args:
            images (list): list of images
        """
        images = [self.model_preprocess(img).unsqueeze(0).to(self.device) for img in images]
        return self.get_image_embeddings(torch.cat(images, dim=0))
    
    def get_image_embeddings(self, images):
        """Function for generating embeddings for a tensor of images

        Returns:
            images (list): Tensor of images
        """
    
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        
        return image_features