from PIL import Image
from pathlib import Path
from typing import Tuple, Union
import cv2 , numpy as np , io , logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageValidator:
    MIN_WIDTH = 150
    MIN_HEIGHT = 150
    MAX_PEAK_HISTOGRAM = 0.4 
    MAX_SYMMETRY_DIFF = 60   
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        if strict_mode:
            self.MIN_WIDTH = 200
            self.MIN_HEIGHT = 200
            self.MAX_PEAK_HISTOGRAM = 0.35
            self.MAX_SYMMETRY_DIFF = 50
    
    def is_chest_xray(self, img_input: Union[bytes, str, Path, Image.Image]) -> Tuple[bool, str]:
        try:
            img = self.load_image(img_input)
            
            if img is None:
                return False, "Failed to decode image"
            h, w = img.shape[:2]
            if h < self.MIN_HEIGHT or w < self.MIN_WIDTH:
                return False, f"Image too small ({w}x{h}). Min: {self.MIN_WIDTH}x{self.MIN_HEIGHT}"
         
            is_valid, message = self.check_contrast(img)
            if not is_valid:
                return False, message
            
            is_valid, message = self._check_symmetry(img)
            if not is_valid:
                return False, message
            
            return True, "Valid chest X-ray"
            
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return False, f"Validation error: {str(e)}"
    
    def load_image(self, img_input: Union[bytes, str, Path, Image.Image]) -> np.ndarray:
        try:
            if isinstance(img_input, bytes):
                nparr = np.frombuffer(img_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
            elif isinstance(img_input, (str, Path)):
                img = cv2.imread(str(img_input), cv2.IMREAD_GRAYSCALE)
                
            elif isinstance(img_input, Image.Image):
                if img_input.mode != 'L':
                    img_input = img_input.convert('L')
                img = np.array(img_input)
                
            elif isinstance(img_input, np.ndarray):
                if len(img_input.shape) == 3:
                    img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
                else:
                    img = img_input
            else:
                raise TypeError(f"Unsupported input type: {type(img_input)}")
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    def check_contrast(self, img: np.ndarray) -> Tuple[bool, str]:
        try:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            
            peak_val = hist.max()
            if peak_val > self.MAX_PEAK_HISTOGRAM:
                return False, f"Low contrast or uniform image (peak: {peak_val:.2f})"
          
            std_dev = np.std(img)
            if std_dev < 20:
                return False, f"Image lacks intensity variation (std: {std_dev:.1f})"
            
            return True, "Contrast OK"
            
        except Exception as e:
            logger.error(f"Contrast check failed: {e}")
            return False, f"Contrast check error: {str(e)}"
    
    def _check_symmetry(self, img: np.ndarray) -> Tuple[bool, str]:
        try:
            h, w = img.shape
            mid = w // 2
            left_half = img[:, :mid]
            right_half = img[:, mid:w]
            right_half_flipped = cv2.flip(right_half, 1)
            if left_half.shape[1] != right_half_flipped.shape[1]:
                min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_half_flipped = right_half_flipped[:, :min_width]
                
            diff = cv2.absdiff(left_half, right_half_flipped)
            mean_diff = diff.mean()
            
            if mean_diff > self.MAX_SYMMETRY_DIFF:
                return False, f"Image not symmetric (diff: {mean_diff:.1f}). May not be chest X-ray"
            
            return True, "Symmetry OK"
            
        except Exception as e:
            logger.error(f"Symmetry check failed: {e}")
            return False, f"Symmetry check error: {str(e)}"
    
    def validate_batch(self, img_inputs: list) -> list:
        results = []
        for img_input in img_inputs:
            result = self.is_chest_xray(img_input)
            results.append(result)
        return results
    
    def get_validation_details(self, img_input: Union[bytes, str, Path, Image.Image]) -> dict:
        try:
            img = self.load_image(img_input)
            
            if img is None:
                return {"error": "Failed to load image"}
            
            h, w = img.shape
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            peak_val = hist.max()
            std_dev = np.std(img)

            mid = w // 2
            left_half = img[:, :mid]
            right_half = cv2.flip(img[:, mid:w], 1)
            
            if left_half.shape[1] != right_half.shape[1]:
                min_width = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
            
            diff = cv2.absdiff(left_half, right_half)
            mean_diff = diff.mean()
            
            is_valid, message = self.is_chest_xray(img_input)
            
            return {
                "is_valid": is_valid,
                "message": message,
                "dimensions": {"width": w, "height": h},
                "contrast": {
                    "histogram_peak": float(peak_val),
                    "std_dev": float(std_dev),
                    "threshold": self.MAX_PEAK_HISTOGRAM
                },
                "symmetry": {
                    "mean_diff": float(mean_diff),
                    "threshold": self.MAX_SYMMETRY_DIFF
                }
            }
            
        except Exception as e:
            return {"error": str(e)}

if __name__ == '__main__':
    validator = ImageValidator()
    
    test_image = "/home/trank/python/python3.13/CNN/Image/normal.jpeg"
    is_valid, message = validator.is_chest_xray(test_image)
    print(f"Valid: {is_valid}, Message: {message}")
    
    # strict mode
    strict_validator = ImageValidator(strict_mode=True)
    is_valid, message = strict_validator.is_chest_xray(test_image)
    print(f"Strict validation - Valid: {is_valid}, Message: {message}")
    
    # detailed validation
    details = validator.get_validation_details(test_image)
    print("\nDetailed validation:")
    for key, value in details.items():
        print(f"  {key}: {value}")
    
    # batch validation
    test_images = [
        "/home/trank/python/python3.13/CNN/Image/normal.jpeg",
        "/home/trank/python/python3.13/CNN/Image/person109_bacteria_523.jpeg",
        "/home/trank/python/python3.13/CNN/Image/person1008_virus_1691.jpeg"
        "/home/trank/python/python3.13/CNN/Image/test.jpeg"
    ]
    results = validator.validate_batch(test_images)
    print("\nBatch validation:")
    for img, (valid, msg) in zip(test_images, results):
        print(f"  {img}: {valid} - {msg}")