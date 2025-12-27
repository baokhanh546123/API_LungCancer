from PIL import Image
from torchvision.models import resnet18 , ResNet18_Weights
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from pathlib import Path
from typing import Union , Dict , Optional
#from filter_image_class import ImageValidator
from model.filter_image_class import ImageValidator
import numpy as np , logging , onnxruntime as ort , io , matplotlib.pyplot as plt , cv2 , torch , torch.nn as nn , torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ONNXInferenceModel:
    def __init__(
        self,
        onnx_path: str ,
        pt_path : str , 
        labels: list,
        input_size: int = 224,
        use_cuda: bool = True,
        threshold: float = 0.5
    ):
        assert len(labels) == 2, "Labels list must contain exactly two elements (e.g., ['NORMAL', 'PNEUMONIA'])."

        self.onnx_path = onnx_path
        self.pt_path = pt_path
        self.pytorch_model = None
        self.labels = labels
        self.input_size = input_size
        self.threshold = float(threshold)

        # 'strict_mode=True' applies stricter validation criteria.
        self.image_validator = ImageValidator(strict_mode=True)

        self.image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        providers = ["CPUExecutionProvider"]
        if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def load_model(self) -> ort.InferenceSession:
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            
            session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=opts,
                providers=['CPUExecutionProvider']
            )
            
            logger.info(f"ONNX model loaded: {self.onnx_path}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def load_model_pt(self) -> nn.Module:
        try:
            if self.pt_path is None:
                raise ValueError("PyTorch model path not provided")
            
            logger.info(f"Loading PyTorch model: {self.pt_path}")

            model = resnet18(weights=ResNet18_Weights.DEFAULT)

            num_features = model.fc.in_features 
            model.fc = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(0.5),
                nn.Linear(num_features, len(self.labels))
            )
            checkpoint = torch.load(self.pt_path , map_location=self.device , weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            logger.info(f"PyTorch model loaded: {self.pt_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise


    def _infer(self, input_np: np.ndarray) -> np.ndarray:
        """Performs raw inference using the ONNX model."""
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_np}
        )
        logits = outputs[0]

        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Invalid output shape from ONNX model: {logits.shape}. Expected (batch_size, 2).")

        return logits
    
    def convert_to_bytes(self , image_path: Union[Image.Image , str , Path , bytes]) -> bytes:
        try:
            if isinstance(image_path , bytes):
                return image_path
            elif isinstance(image_path , (str , Path)):
                with open(image_path , 'rb') as f :
                    return f.read()
            elif isinstance(image_path , Image.Image):
                buffer = io.BytesIO()
                image_path.save(buffer , format='JPEG')
                return buffer.getvalue()
            else:
                TypeError(f"Cannot convert {type(image_path)} to bytes")
                
        except Exception as e:
            logger.error(f"Failed to convert to bytes: {e}")
            raise
    
    def convert_to_pil_image(self , image_path: Union[Image.Image , str , Path , bytes , np.ndarray]) -> Image.Image:
        try:
            if isinstance(image_path , Image.Image):
                if image_path.mode != 'RGB':
                    return image_path.convert('RGB')
                return image_path
            elif isinstance(image_path , (str , Path)):
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    return img.convert('RGB')
                return img
            elif isinstance(image_path , bytes):
                img = Image.open(io.BytesIO(image_path))
                if img.mode != 'RGB':
                    return img.convert('RGB')
                return img
            elif isinstance(image_path , np.ndarray):
                if len(image_path.shape) == 2 :
                     img = Image.fromarray(image_path, mode='L')
                elif len(image_path.shape) == 3:  # Color
                    img = Image.fromarray(image_path)
                else:
                    raise ValueError(f"Invalid array shape: {image_path.shape}")
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img
            else:
                raise TypeError(f"Unsupported input type: {type(image_path)}")
                
        except Exception as e:
            logger.error(f"Failed to convert input to PIL Image: {e}")
            raise


    def predict(self, image_path: Union[Image.Image , str , Path , bytes]) -> Dict:
        try:
            try:
                img_bytes = self.convert_to_bytes(image_path)
                is_valid, message = self.image_validator.is_chest_xray(img_bytes)
            except Exception as e :
                if isinstance(image_path , Image.Image):
                    is_valid , message =  True, "Validation skipped for PIL Image"
                else:
                    raise e 
                
            if not is_valid:
                result = {
                    "Error" : message,
                    "validation_failed": True
                }
                return result
            
            pil_img = self.convert_to_pil_image(image_path)
            input_tensor = self.image_transforms(pil_img)
            input_np = input_tensor.unsqueeze(0).numpy().astype(np.float32)

            # Step 3: Perform inference using the ONNX model
            logits = self._infer(input_np)

            # Apply softmax to convert logits to probabilities for binary classification.
            # Subtracting max for numerical stability to prevent exp overflow.
            logits_stable = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits_stable)
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Extract probabilities for the two classes
            normal_prob = float(probabilities[0, 0])
            pneumonia_prob = float(probabilities[0, 1])

            # Determine the predicted class based on the threshold
            is_pneumonia = pneumonia_prob >= self.threshold
            pred_index = 1 if is_pneumonia else 0
            predicted_label = self.labels[pred_index]

            # Calculate prediction confidence
            confidence = max(normal_prob, pneumonia_prob)

            # Step 4: Format and return the results
            result = {
                "clinical_decision": predicted_label,
                "decision_score": pneumonia_prob, # Probability of the positive class (PNEUMONIA)
                "decision_threshold": self.threshold,
                "risk_probability": {
                    self.labels[0]: normal_prob,
                    self.labels[1]: pneumonia_prob
                },
                "prediction_confidence": confidence,
                "interpretation": (
                    f"Risk of {self.labels[1]} is {pneumonia_prob:.2%}. "
                    f"Threshold is {self.threshold:.2%}. "
                    f"Clinical decision: {predicted_label}."
                ),
                "ai_role": "Clinical Decision Support",
                "disclaimer": (
                    "This AI output is intended to support clinical decision-making. "
                    "Final diagnosis must be made by a qualified medical professional."
                )
            }

            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "Error": f"Prediction error: {str(e)}",
                "prediction_failed": True
            }
    
    def predict_batch(self , image_path: list) -> list:
        results = []
        for path in image_path:
            result = self.predict(path)
            results.append(result)
        return results

    def gradcam_for_img(self , model : nn.Module , image_path: Union[Image.Image , str , Path , bytes] , transform : transforms.Compose, 
                        target_layer : Optional[nn.Module] = None , method : str = 'gradcam' , validate_img : bool = False) -> Dict:
        try:
            if not isinstance(model , nn.Module):
                raise TypeError(
                "Grad-CAM requires a PyTorch nn.Module. "
                "ONNXRuntime models are not supported."
            )

            cam_map = {
                "gradcam": GradCAM,
                "gradcam++": GradCAMPlusPlus,
                "eigencam": EigenCAM
            }

            if method.lower() not in cam_map:
                raise ValueError(f"Unsupported CAM method: {method}")

            model = model.to(self.device).eval()

            if target_layer is None:
                try:
                    target_layer = model.layer4[-1].conv2
                except Exception:
                    raise ValueError(
                            "Cannot infer target_layer automatically. "
                            "Please pass a convolutional layer explicitly."
                        )
            
            if validate_img:
                try:
                    img_bytes = self.convert_to_bytes(image_path)
                    is_valid, message = self.image_validator.is_chest_xray(img_bytes)
                    if not is_valid:
                        return {
                            "success": False,
                            "error": message,
                            "validation_failed": True
                        }
                except Exception as e :
                     if not isinstance(image_path, Image.Image):
                        logger.warning(f"Image validation failed: {e}")
                        
            pil_img = self.convert_to_pil_image(image_path)
            orig_img_resized = pil_img.resize((224, 224), Image.BILINEAR)
            orig_img = np.array(orig_img_resized).astype(np.float32) / 255.0

            input_tensor = transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                
                normal_prob = probs[0, 0].item()
                pneumonia_prob = probs[0, 1].item()
                
                pred_class = 1 if pneumonia_prob >= self.threshold else 0
                pred_label = self.labels[pred_class]
                pred_confidence = probs[0, pred_class].item()
            
            logger.info(
                f"Grad-CAM prediction: {pred_label} "
                f"(confidence: {pred_confidence:.2%})"
            )
            
            cam_algorithm = cam_map[method.lower()]
    
            cam = cam_algorithm(
                model=model,
                target_layers=[target_layer]
            )
            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            if grayscale_cam.ndim == 3:
                grayscale_cam = grayscale_cam[0]

            grayscale_cam = np.maximum(grayscale_cam, 0)
            cam_max = grayscale_cam.max()
            if cam_max > 0:
                grayscale_cam = grayscale_cam / (cam_max + 1e-8)
            else:
                logger.warning("Grad-CAM produced all-zero heatmap")

            cam_resized = cv2.resize(
                grayscale_cam,
                (orig_img.shape[1], orig_img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            cam_overlay = show_cam_on_image(
                orig_img,
                cam_resized,
                use_rgb=True
            )

            return {
                "success": True,
                "predicted_class": pred_class,
                "predicted_label": pred_label,
                "prediction_confidence": pred_confidence,
                "probabilities": {
                    self.labels[0]: normal_prob,
                    self.labels[1]: pneumonia_prob
                },
                "grayscale_cam": cam_resized,
                "cam_overlay": cam_overlay,
                "method": method,
                "threshold": self.threshold
            }
        
        except TypeError as e:
            logger.error(f"Type error in Grad-CAM: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "TypeError"
            }
            
        except ValueError as e:
            logger.error(f"Value error in Grad-CAM: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "ValueError"
            }
            
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Grad-CAM error: {str(e)}",
                "error_type": type(e).__name__
            }

"""path = '/home/trank/python/python3.13/CNN/Image/person1008_virus_1691.jpeg'
onnx = '/home/trank/python/python3.13/CNN/model/resnet18_lung_finetuned.onnx'
pt = '/home/trank/python/python3.13/CNN/model/best_pneumonia_classifier.pt'
LABELS = ['NORMAL', 'PNEUMONIA']

model = ONNXInferenceModel(onnx_path=onnx , pt_path=pt , labels=LABELS , threshold=0.6)

print('*'*50)
result = model.predict(path)
if 'Error' in result:
    print(f"Error: {result['Error']}")
else:
    print(f"Prediction: {result.get('clinical_decision')}")
    print(f"Confidence: {result.get('prediction_confidence', 0):.2%}")
print('*'*50)
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
pt_model = model.load_model_pt()
gradcam = model.gradcam_for_img(model=pt_model , image_path=path , transform=infer_transform , method='eigencam' , validate_img=True)
if gradcam['success']:
    print(f"Grad-CAM Prediction: {gradcam.get('predicted_label')}")
    print(f"Confidence: {gradcam.get('prediction_confidence', 0):.2%}")
    
    plt.figure(figsize=(10, 5))
    plt.title("Grad-CAM Overlay")
    plt.imshow(gradcam['cam_overlay'])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("/home/trank/python/python3.13/CNN/Image/gradcam_output.png" , dpi = 150 , bbox_inches='tight')
else:
    print(f"Grad-CAM Error: {gradcam['error']}")"""