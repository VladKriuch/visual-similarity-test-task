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
            
        return image_features / image_features.norm(dim=-1, keepdim=True)
    
    def tokenize_text(self, text_list):
        """Tokenizer for predicting text embeddings"""
        text = clip.tokenize(text_list).to(self.device)
        return text
    
    def embed_text_from_tokens(self, text_tokens):
        """Embeds text from token"""
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)
        
    def get_category_probs(self, image, text):
        """Function to generate probability of text correspondencies to the image"""
        logits_per_image, _ = self.model(self.model_preprocess(image).unsqueeze(0).to(self.device), text)
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
        
        return probs