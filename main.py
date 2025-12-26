from fastapi import FastAPI, HTTPException, Depends , Request , Form , Query , status , UploadFile , File
from datetime import datetime, timedelta , timezone
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse ,JSONResponse , RedirectResponse , Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
from connect.pysql import get_members_table , produre_add_members , produre_delete_members , produre_update_members , length_member_id
from connect.nosql import MongoDB
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import Any , Annotated , List 
from pathlib import Path
from event.detect_gpu import Detect
from event.detect_gradcam import GradCam
import logging , re , os ,random , sys , uvicorn , io

logging.basicConfig(level=logging.INFO)

current_dir = Path(__file__).parent
env_file = current_dir / 'connect/connection.env'
load_dotenv(env_file)


def pre_run():
    try:
        detector = Detect()
        installer = GradCam()
        
        check_gpu = detector.install_library(min_cores=128)
        if not check_gpu:
            return False
            
        check_gradcam = installer.install_grad_cam()
        return check_gradcam
    except Exception as e :
        logging.error(f"Lỗi trong quá trình pre_run: {e}")
        return False

app = FastAPI(title="Chest X-ray Classification API",description="ONNX-based chest X-ray classification",version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

nosql = MongoDB(uri=os.getenv('HOST_NSQL') , db_name=os.getenv('DB_NAME') , collection_name=os.getenv('COL_NAME'))
@asynccontextmanager
async def lifespan(app : FastAPI):
    try:
        await nosql.connect()
        logging.info('Mongo ready')
    except Exception as e :
        logging.error('Failed' , e )
    
    yield
    await nosql.close()
    logging.info('exit')


@app.middleware("http")
async def no_cache_middleware(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    members = [member for member in get_members_table(query='all')]
    return templates.TemplateResponse("index.html", {"request": request, "members": members})

    
@app.get("/predict" , response_class=HTMLResponse)
async def test(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


async def file_to_image(upload_img: UploadFile = File(..., alias="upload-img")) -> Image.Image:
    if not upload_img.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await upload_img.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        return img , contents
    except Exception as e:
        logging.error(f"Error parsing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

@app.post("/load_model") 
async def load_model(
    model: Annotated[str, Form()],
    image: Annotated[Image.Image , Depends(file_to_image)]
):
    try:

        onnx_path = current_dir / 'model/resnet18_lung_finetuned.onnx'
        if not onnx_path.exists():
            raise HTTPException(status_code=500, detail="Model file not found on server")
    
        
        LABELS = ['NORMAL', 'PNEUMONIA']

        if model == 'model-resnet':
            from model.restnet18_onnx_inference import ONNXInferenceModel
            restnet = ONNXInferenceModel(str(onnx_path) , LABELS , threshold=0.6)
            result = restnet.predict(img_bytes)

            if 'Error' in result:
                return {"status": "error", "message": result['Error']}

            return {
                "status": "success", 
                "model_used": model,
                "result": {
                    'prediction_label': result.get('clinical_decision'),
                    'decision_score': result.get('decision_score'),
                    'prediction_confidence': result.get('prediction_confidence'),
                    'raw_probabilities': result.get('risk_probability'),
                    'decision_threshold': result.get('decision_threshold'),
                    'interpretation': result.get('interpretation')
                }
            }
        else:
            return {"status": "error", "message": f"Model '{model}' is not implemented yet."}
    except Exception as e :
        logging.error(f"Server Error: {str(e)}")
        return {"status": "failed", "error": str(e)}




if __name__ == "__main__":
    if pre_run():
        print("\n" + "="*50)
        print("[SUCCESS] Môi trường hợp lệ. Đang khởi động Server...")
        print("="*50 + "\n")
        uvicorn.run(app, host="127.0.0.1", port=8000 , reload = False)
    else:
        print("\n" + "!"*50)
        print("[FAILED] Thiếu thư viện hoặc phần cứng không đạt.")
        print("Server sẽ không khởi động.")
        print("!"*50 + "\n")
        sys.exit(1)

    



