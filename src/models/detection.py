import torch
from ultralytics import YOLO

from src.utils import custom_non_max_suppression

from copy import deepcopy

class DetectionModel:
    """Object detection model
    Main purpose is to find ROI objects on the image, and crop using detected results while avoiding
    noisy info. Hopefully, it'll help by providing more accurate results
    """
    def __init__(self, 
                 model_type = "yolov8x-oiv7", 
                 force_cpu = False, 
                 reclassification_dict = {},
                 conf=0.2,
                 iou=0.5):
        """Init method

        Args:
            model_type (str, optional):  Model to use for object detection. Here, the YOLOv8 pretrained on oiv7 is used as default. Defaults to "yolov8x-oiv7".
            force_cpu (bool, optional): Whether to force cpu inference. Defaults to False.
            reclassification_dict (dict, optional): Dict to reclassify results.
            Used as an option to lower the amount of classes so that categories would not hurt the perfomance. Defaults to {}.
            
            conf; iou - parameters used at inference of the model
        """
        self.device = "cuda" if torch.cuda.is_available() and force_cpu is not True else "cpu"
        self.model = YOLO(model_type, verbose=False)
        
        self.reclassification_dict = reclassification_dict
        
        self.conf = conf
        self.iou = iou
    
    def find_points_of_interest(self, image):
        """Local pipeline function.
        
        Involves
        1. Detecting initial bounding boxes
        2. Reclassifying results with custom classification targets
        3. Elaborating custom non-max suppression that works both interclasses and combines some of the bounding boxes.
        """
        boxes = self.get_boxes(image)
        classified_boxes = self.reclassify_results(boxes)
        
        if len(classified_boxes['labels']) > 0:
            return custom_non_max_suppression(classified_boxes['labels'], classified_boxes['boxes'], classified_boxes['conf'], distance_threshold=100)
        return classified_boxes
        
    def get_boxes(self, image):
        """Function that retrieves bounding boxes using YOLO and converts them to the needed format
        """
        results = self.model(image, conf=self.conf, iou=self.iou)
        
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
        """
        Method used for reclassifying results
        
        In default mode, it just converts specific results like Skirt, T-shirt to more generous like Clothing
        """
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