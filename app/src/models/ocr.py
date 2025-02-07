import easyocr

class TextDetectionModel:
    """Text detection model based on easyocr
    
    The main disadvantage here is that EasyOCR is slow as hell
    """
    def __init__(self, languages: list = ['en'], force_cpu: bool = False, conf_level: float = 0.5):
        """Init

        Args:
            languages (list, optional): Defaults to ['en']. Languages to use in ocr search
            force_cpu (bool, optional): Defaults to False. Whether to force cpu inference
            conf_level(float, optional): Defaults to 0.5. Confidence level to filter results by 
        """
        self.reader = easyocr.Reader(languages, gpu=True)
        self.conf_level = conf_level
        
    def get_text(self, image) -> set[str]:
        """Applies ocr to the image and returns the result to user

        Args:
            image (PIL.Image | filepath | np.ndarray): Image/filepath to the image. Typically used as PIL.Image or np.ndarray

        Returns:
            set[str]: Unique texts found in the image. 
        """
        return set([text[1] for text in self.reader.readtext(image) if text[2] > self.conf_level])