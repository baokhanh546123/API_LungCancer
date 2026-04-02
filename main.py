from fastapi import FastAPI, HTTPException, Depends , Request , Form , Query , status , UploadFile , File
from datetime import datetime, timedelta , timezone
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse ,JSONResponse , RedirectResponse , Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dotenv import load_dotenv
from typing import Any , Annotated , List , Dict
from pathlib import Path
from event.detect_gpu import Detect
from event.detect_gradcam import GradCam
import logging , re , os ,random , sys , uvicorn , io , time , base64 , asyncio , numpy as np , threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def pre_run():
    try:
        detector = Detect()
        installer = GradCam()
        logging.info("Checking GPU compatibility...")
        detector.info()
        detector.install_library()

        installer.install_grad_cam()
        
        from model.dowload_model import download_models , check_folder
        MODEL_REPO = "Trank123/API_LungCancer"
        FILES_TO_DOWNLOAD = [
        "best_pneumonia_classifier.pt",
        "best_pneumonia_classifier_mobilenetv2.pt",
        "mobilenetv2_lung_finetuned.onnx",
        "mobilenetv2_lung_finetuned.onnx.data",
        "resnet18_lung_finetuned.onnx",
        "resnet18_lung_finetuned.onnx.data",
        "keras_cnn_xray.onnx",
        "model_fold1.h5"
        ]
        is_model_ready = check_folder(MODEL_REPO, FILES_TO_DOWNLOAD)
        if not is_model_ready:
            logging.info("Model files missing. Starting automatic download from HF...")
            download_models(
                repo_id=MODEL_REPO, 
                file_list=FILES_TO_DOWNLOAD, 
                target_subdir='model/models'
            )
            return True
        return True
    except Exception as e:
        logging.error(f"Error during pre_run: {e}")
        return False
    

app = FastAPI(title="Chest X-ray Classification API")

current_dir = Path(__file__).parent.resolve()

@app.middleware("http")
async def no_cache_middleware(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.mount("/static", StaticFiles(directory= current_dir / "static"), name="static")
templates = Jinja2Templates(directory= str(current_dir / "templates"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logging.error(f"Error in home route: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    
@app.get("/predict" , response_class=HTMLResponse)
async def test(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


async def file_to_image(upload_img: UploadFile = File(..., alias="upload-img")) -> Image.Image:
    if not upload_img.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await upload_img.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail={"error": "File too large", "message": "Max 10MB"}
            )
        
        img = Image.open(io.BytesIO(contents))

        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.filename = upload_img.filename
        return img
    except Exception as e:
        logging.error(f"Error parsing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")
    except HTTPException:
        raise

executor = ThreadPoolExecutor(max_workers=4)
def encode_img_to_base64(img_arr : np.ndarray , format : str = 'PNG') -> str:
    try:
        if img_arr.dtype != np.uint8:
            img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img_arr , mode='L')  if len(img_arr.shape) == 2 else Image.fromarray(img_arr , mode='RGB')    
        buffered = io.BytesIO()
        pil_img.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/{format.lower()};base64,{base64_str}"
    except Exception as e:
        logging.error(f"Failed to encode image: {e}")
        return ""

@app.post("/load_model") 
async def load_model(
    model: Annotated[str, Form()],
    image: Annotated[Image.Image , Depends(file_to_image)]
):
    try:
        onnx_path , pt_path = None , None

        LABELS = ['NORMAL', 'PNEUMONIA']

        if model == 'model-resnet':
            onnx_path = current_dir / 'model/models/resnet18_lung_finetuned.onnx'
            pt_path = current_dir / 'model/models/best_pneumonia_classifier.pt'
            if (not onnx_path.exists() or onnx_path is None) and (not pt_path.exists() or pt_path is None):
                raise HTTPException(status_code=500, detail="Failed to download model files")
                
            from model.restnet18_onnx_inference import ONNXInferenceModel
            start = time.perf_counter()
            restnet = ONNXInferenceModel(str(onnx_path) , str(pt_path) , LABELS , threshold=0.6)
            
            #predict
            result = restnet.predict(image)
            if 'Error' in result:
                return {"status": "error", "message": result['Error']}

            loop = asyncio.get_event_loop()
            grad_cam_res = await loop.run_in_executor(
                executor,
                partial(restnet.gradcam_for_img, image, restnet.image_transforms, method='gradcam')
            )
            if not grad_cam_res.get('success', False):
                error_msg = grad_cam_res.get('error', 'Unknown error')
                logging.error(f"Grad-CAM failed: {error_msg}")
                return {
                    "success": False,
                    "message": error_msg
                }
            
            overlay_base64 = encode_img_to_base64(grad_cam_res['cam_overlay'] , format='PNG')
            
            if not overlay_base64:
                raise ValueError("Failed to encode Grad-CAM images")
            
            end = time.perf_counter()
            delta = float(end - start) 

            return {
                "status": "success", 
                "model_used": model,
                "result": {
                    'prediction_label': result.get('clinical_decision'),
                    'decision_score': result.get('decision_score'),
                    'prediction_confidence': result.get('prediction_confidence'),
                    'raw_probabilities': result.get('risk_probability'),
                    'decision_threshold': result.get('decision_threshold'),
                    'interpretation': result.get('interpretation'),
                    "run_time": round(delta, 4)
                },
                "images" : {
                     "gradcam" : overlay_base64 if overlay_base64 is not None else 'HTTP 500 Error'
                }
            }

        elif model == 'model-mobinet':
            onnx_path = current_dir / 'model/models/mobilenetv2_lung_finetuned.onnx'
            pt_path = current_dir / 'model/models/best_pneumonia_classifier_mobilenetv2.pt'

            if (not onnx_path.exists() or onnx_path is None) and (not pt_path.exists() or pt_path is None):
                raise HTTPException(status_code=500, detail="Model file not found on server")
            
            from model.mobilenetv2_lung_inference import Mobinet_ONNXInferenceModel
            start = time.perf_counter()
            mobilenet = Mobinet_ONNXInferenceModel(str(onnx_path) , str(pt_path) , LABELS , threshold=0.75)
            #predict
            result = mobilenet.predict(image)
            if 'Error' in result:
                return {"status": "error", "message": result['Error']}
            loop = asyncio.get_event_loop()
            grad_cam_res = await loop.run_in_executor(
                executor,
                partial(mobilenet.gradcam_for_img, image, mobilenet.image_transforms, method='gradcam')
            )
            if not grad_cam_res.get('success', False):
                error_msg = grad_cam_res.get('error', 'Unknown error')
                logging.error(f"Grad-CAM failed: {error_msg}")
                return {
                    "success": False,
                    "message": error_msg
                }
            
            overlay_base64 = encode_img_to_base64(grad_cam_res['cam_overlay'] , format='PNG')
            if not overlay_base64:
                raise ValueError("Failed to encode Grad-CAM images")
            
            end = time.perf_counter()
            delta = float(end - start)
            return {
                "status": "success", 
                "model_used": model,
                "result": {
                    'prediction_label': result.get('clinical_decision'),
                    'decision_score': result.get('decision_score'),
                    'prediction_confidence': result.get('prediction_confidence'),
                    'raw_probabilities': result.get('risk_probability'),
                    'decision_threshold': result.get('decision_threshold'),
                    'interpretation': result.get('interpretation'),
                    "run_time": round(delta, 4)
                },
                "images" : {
                     "gradcam" : overlay_base64 if overlay_base64 is not None else 'HTTP 500 Error'
                }
            }
        elif model == "model-handmade":
            onnx_path = current_dir / 'model/models/keras_cnn_xray.onnx'
            #pt_path = current_dir / 'model/models/best_pneumonia_classifier_mobilenetv2.pt'

            #if (not onnx_path.exists() or onnx_path is None) and (not pt_path.exists() or pt_path is None):
            #    raise HTTPException(status_code=500, detail="Model file not found on server")
            
            from model.handmake_onnx_inference import Handmake_ONNXInferenceModel
            start = time.perf_counter()
            hm = Handmake_ONNXInferenceModel(str(onnx_path) , str(pt_path) , LABELS , threshold=0.75)
            #predict
            result = mobilenet.predict(image)
            if 'Error' in result:
                return {"status": "error", "message": result['Error']}
            loop = asyncio.get_event_loop()
            grad_cam_res = await loop.run_in_executor(
                executor,
                partial(mobilenet.gradcam_for_img, image, mobilenet.image_transforms, method='gradcam')
            )
            if not grad_cam_res.get('success', False):
                error_msg = grad_cam_res.get('error', 'Unknown error')
                logging.error(f"Grad-CAM failed: {error_msg}")
                return {
                    "success": False,
                    "message": error_msg
                }
            
            overlay_base64 = encode_img_to_base64(grad_cam_res['cam_overlay'] , format='PNG')
            if not overlay_base64:
                raise ValueError("Failed to encode Grad-CAM images")
            
            end = time.perf_counter()
            delta = float(end - start)
            return {
                "status": "success", 
                "model_used": model,
                "result": {
                    'prediction_label': result.get('clinical_decision'),
                    'decision_score': result.get('decision_score'),
                    'prediction_confidence': result.get('prediction_confidence'),
                    'raw_probabilities': result.get('risk_probability'),
                    'decision_threshold': result.get('decision_threshold'),
                    'interpretation': result.get('interpretation'),
                    "run_time": round(delta, 4)
                },
                "images" : {
                     "gradcam" : overlay_base64 if overlay_base64 is not None else 'HTTP 500 Error'
                }
            }
            
        else:
            raise HTTPException(status_code=400, detail="Invalid model selection")
    except HTTPException:
        raise
    except Exception as e :
        logging.error(f"Server Error: {str(e)}")
        return {"status": "failed", "error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}



if __name__ == "__main__":
    if pre_run():
        print("\n" + "="*50)
        print("[SUCCESS] Môi trường hợp lệ. Đang khởi động Server...")
        print("="*50 + "\n")
        uvicorn.run(app, host="127.0.0.11", port=8000 , reload = False)
    else:
        print("\n" + "!"*50)
        print("[FAILED] Thiếu thư viện hoặc phần cứng không đạt.")
        print("Server sẽ không khởi động.")
        print("!"*50 + "\n")
        sys.exit(1)

    



