import torch
import clip
from PIL import Image

class EmbeddingModel:
    def __init__(self, model_type="ViT-B/32", force_cpu=False):
        """Class for generating embeddings for images/texts

        Args:
            model_type (str, optional): Model type provided for clip. Defaults to "ViT-B/32".
            force_cpu (bool, optional): Whether to force cpu. If True, the device will be forced to use cpu. Defaults to False.
        """
        self.device = "cuda" if torch.cuda.is_available() and force_cpu is not True else "cpu"
        self.model, self.model_preprocess = clip.load(model_type, device=self.device)
        
    def embed_images(self, img_paths: list):
        """Function for creating embeddings for img_paths
        Returns the type that self.model.encode_image returns

        Args:
            img_paths (list): List of paths to images
        """
        images = [self.model_preprocess(Image.open(impath)).unsqueeze(0).to(self.device) for impath in img_paths]
        images = torch.cat(images, dim=0)
        
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        
        return image_features