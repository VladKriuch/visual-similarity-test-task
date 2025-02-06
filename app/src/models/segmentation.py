import torch
from ultralytics import SAM


class SegmentationModel:
    """Object segmentation model
    Main purpose is to retrieve the masks on the image based on detected bounding boxes in order to 
    retrieve clearer representation of an object
    """
    def __init__(self, 
                 force_cpu = False,): 
        """Init method

        Args:
            force_cpu (bool, optional): Whether to force cpu inference. Defaults to False.
        """
        self.device = "cuda" if torch.cuda.is_available() and force_cpu is not True else "cpu"
        self.model = SAM(verbose=False)
        
    def get_segmented_parts(self, image, detected_rois):
        pass