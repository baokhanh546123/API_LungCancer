from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K 
from pathlib import Path
#from filter_image_class import ImageValidator
from PIL import Image
from model.filter_image_class import ImageValidator
from typing import Dict , Tuple , Union , List , Optional
import onnx , onnxruntime as ort , random , numpy as np , tensorflow as tf , numpy as np , cv2 , matplotlib.pyplot as  plt , logging , io , scipy.ndimage as ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ONNXInferenceHandmakeModel:
    def __init__(self , onnx_path : Union[str,Path]  , tf_path : Optional[Union[str, Path]] , labels : List[str] ,
                input_size : Tuple[int , int] = (128,128) , use_cuda : bool = True 
                ,last_conv_layer_name : str = 'conv_4_2' , threshold : float = 0.75):
                assert len(labels) == 2 , "Labels list must contain exactly two elements (e.g., ['NORMAL', 'cancer'])."

                self.image_validator = ImageValidator(strict_mode=True)

                self.onnx_path = onnx_path
                self.tf_path = tf_path 
                self.labels = labels
                self.input_size = input_size
                self.use_cuda = use_cuda
                self.last_conv_layer_name = last_conv_layer_name

                self._tf_model = None 

                #self.image_transforms = self._augumetation()
                
                self.session = self._load_model()
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.threshold = threshold
    
    def _load_model(self) -> ort.InferenceSession:
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4 

            available = ort.get_available_providers()

            providers = ["CPUExecutionProvider"]
            if "CoreMLExecutionProvider" in available:
                providers.insert(0, "CoreMLExecutionProvider")
            if self.use_cuda and "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
            self.session = ort.InferenceSession(
                self.onnx_path,
                sess_options=opts,
                providers=providers
            )
            logger.info(f"ONNX model loaded: {self.onnx_path}")
            return self.session
        except Exception as e :
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _load_model_tf(self) -> tf.keras.Model:
        if self.tf_path is None:
            raise ValueError("TF model path not provided")
        try:
            
            best_model = tf.keras.models.load_model(self.tf_path)
            logger.info(f"Loading TF model: {self.tf_path}")
            return best_model
        except Exception as e :
            logger.error(f"Failed to load TF model: {e}")
            raise
    
    def __preprocessing(self , image_path , augment : bool = False):
        if isinstance(image_path , Image.Image):
            img_orig = image_path
            if img_orig.size != self.input_size:
                img_orig = img_orig.resize(self.input_size , Image.BILINEAR)
        else:
            img_orig = load_img(image_path , target_size=self.input_size)
        img_array = img_to_array(img_orig)
        if augment:
            angle = np.random.uniform(-10,10)
            img_array = ndimage.rotate(img_array , angle , reshape=False , mode = 'nearest')
            if np.random.rand() > 0.5:
                img_array = ndimage.gaussian_filter(img_array , sigma=1)

        img_array = img_array / 255.0
        img_tensor = np.transpose(img_array, (2, 0, 1))
        img_input = np.expand_dims(img_array, axis=0)
        img_input = img_input.astype(np.float32)
        return img_input 
    
    def _infer(self , input_np : np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_np}
        )
        return outputs
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
    
    def predict(self , img ) -> Dict:
        try:
            try:
                img_bytes = self.convert_to_bytes(img)
                is_valid , message = self.image_validator.is_chest_xray(img_bytes)
            except Exception as e :
                if isinstance(img , Image.Image):
                    is_valid , meassage = True , "Validation skip image"
                else:
                    raise e 
            if not is_valid:
                result = {
                    "Error" : message,
                    "validation_failed" : True
                }
            
            pil_img = self.convert_to_pil_image(img)
            input_np = self.__preprocessing(pil_img,augment=True)
            
            #input_np = self.__preprocessing(img)
            #input_np = input_tensor.unsqueeze(0).numpy().astype(np.float32)
            logits = self._infer(input_np)

            probabilities = logits[0]
            normal_prob = float(probabilities[0,0])
            cancer_prob = float(probabilities[0,1])

            #predict_class_idx = np.argmax(probabilities)
            predict_class_idx = 1 if cancer_prob >= self.threshold else 0 
            predicted_label = self.labels[predict_class_idx]
            confidence = float(probabilities[0,predict_class_idx])

            result = {
                    "clinical_decision": predicted_label,
                    "decision_score": cancer_prob,
                    "decision_threshold": self.threshold,
                    "risk_probability": {
                        self.labels[0]: round(normal_prob,8),
                        self.labels[1]: round(cancer_prob,8)
                    },
                    "prediction_confidence": confidence,
                    "interpretation": (
                        f"Risk of {self.labels[1]} is {cancer_prob:.2%}. "
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

    def _build_grad_model(self , model : tf.keras.Model) -> tf.keras.Model:
        layer_names = [layer.name for layer in model.layers]
        if self.last_conv_layer_name not in layer_names:
            raise ValueError(
                f"Layer '{self.last_conv_layer_name}' not found in the TF model. "
                f"Available layers: {layer_names}"
            )
        input_tensor = tf.keras.Input(shape=model.input_shape[1:])
 
        x = input_tensor
        intermediate_output = None
 
        for layer in model.layers:
            x = layer(x)
            if layer.name == self.last_conv_layer_name:
                intermediate_output = x
 
        final_output = x
 
        grad_model = Model(
            inputs=input_tensor,
            outputs=[intermediate_output, final_output],
        )
        logger.info("GradCAM gradient model built on layer: %s", self.last_conv_layer_name)
        return grad_model

    def _ensure_tf_loaded(self):
        if self._tf_model is None:
            self._tf_model = self._load_model_tf()
            self._grad_model = self._build_grad_model(self._tf_model)
    
    
    def _make_gradcam_heatmap(self , img_array : np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        if self._grad_model is None:
            self._grad_model = self._build_grad_model(self._tf_model)
        
        grad_model = self._grad_model
    
        with tf.GradientTape() as tape:
            # Watch the conv-layer outputs so gradients flow through them
            last_conv_layer_output, predictions = grad_model(img_array, training=False)
            tape.watch(last_conv_layer_output)
 
            # Threshold-aware class selection (mirrors predict())
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]
 
        grads = tape.gradient(class_score, last_conv_layer_output)  # (1, h, w, c)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))         # (c,)
 
        # Weighted sum of feature maps
        last_conv_layer_output = last_conv_layer_output[0]           # (h, w, c)
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
        heatmap = tf.squeeze(heatmap)                                 # (h, w)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + K.epsilon())
 
        pred_idx = int(pred_index.numpy())
        probs = predictions[0].numpy()
 
        return heatmap.numpy(), pred_idx, probs

    def gradcam_for_img(self, image , method : str = 'gradcam' ,alpha : float = 0.5 ) -> Dict:
        try:
            self._ensure_tf_loaded()
            pil_img = self.convert_to_pil_image(image)
            pil_resized = pil_img.resize(
                (self.input_size[1], self.input_size[0]), Image.BILINEAR
            )
            orig_img = np.array(pil_resized, dtype=np.float32) / 255.0  # (H, W, 3) [0,1]
            img_input = np.expand_dims(orig_img, axis=0)                 # (1, H, W, 3)
 
            # --- GradCAM computation -----------------------------------
            heatmap, pred_idx, probs = self._make_gradcam_heatmap(img_input)
 
            cam_resized = cv2.resize(
                heatmap,
                (orig_img.shape[1], orig_img.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
 
            # Guard against all-zero heatmaps
            cam_max = cam_resized.max()
            if cam_max > 0:
                cam_resized = cam_resized / (cam_max + 1e-8)
            else:
                logger.warning("GradCAM produced an all-zero heatmap.")
 
            # Build coloured overlay (BGR → RGB for web / matplotlib)
            heatmap_uint8 = np.uint8(255 * cam_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
 
            orig_uint8 = np.uint8(255 * orig_img)
            cam_overlay = cv2.addWeighted(orig_uint8, 1 - alpha, heatmap_color, alpha, 0)
 
            pred_label = self.labels[pred_idx]
            pred_confidence = float(probs[pred_idx])
            normal_prob = float(probs[0])
            pneumonia_prob = float(probs[1])
 
            logger.info(
                "GradCAM prediction: %s (confidence: %.2f%%)",
                pred_label,
                pred_confidence * 100,
            )
 
            return {
                "success": True,
                "predicted_class": pred_idx,
                "predicted_label": pred_label,
                "prediction_confidence": round(pred_confidence, 8),
                "probabilities": {
                    self.labels[0]: round(normal_prob, 8),
                    self.labels[1]: round(pneumonia_prob, 8),
                },
                "grayscale_cam": cam_resized,   # np.ndarray (H, W), float [0,1]
                "cam_overlay": cam_overlay,      # np.ndarray (H, W, 3), uint8 RGB
                "method": method,
                "threshold": self.threshold,
            }
 
        except (TypeError, ValueError) as exc:
            logger.error("GradCAM error [%s]: %s", type(exc).__name__, exc)
            return {"success": False, "error": str(exc), "error_type": type(exc).__name__}
 
        except (TypeError, ValueError) as exc:
            logger.error("GradCAM error [%s]: %s", type(exc).__name__, exc)
            return {"success": False, "error": str(exc), "error_type": type(exc).__name__}
 
        except Exception as exc:
            logger.error("GradCAM generation failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "error": f"GradCAM error: {exc}",
                "error_type": type(exc).__name__,
            }

if __name__ == '__main__':
    onnx_path = None
    tf_path = None
    img_path = None
    h = ONNXInferenceHandmakeModel(onnx_path=onnx_path , tf_path=tf_path , labels=['NORMAL', 'PNEUMONIA'])
    result = h.predict(img_path)
    print(result["risk_probability"])
    """gcam = h.grad_cam(img_path , alpha=0.5)
    if gcam['success']:
        print("GradCAM label      :", gcam["predicted_label"])
        print("GradCAM confidence :", f"{gcam['prediction_confidence']:.2%}")
 
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"GradCAM – {gcam['predicted_label']}", fontsize=14)
        axes[0].imshow(gcam["cam_overlay"])
        axes[0].set_title("Overlay")
        axes[0].axis("off")
        axes[1].imshow(gcam["grayscale_cam"], cmap="jet")
        axes[1].set_title("Heatmap (grayscale)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig("/home/trank/Documents/API_LungCancer/Image/gradcam_output.png", dpi=150, bbox_inches="tight")
        print("GradCAM saved to gradcam_output.png")
    else:
        print("GradCAM error:", gcam["error"])"""