import os , logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

logging.basicConfig(level=logging.INFO)

def check_folder(repo_id, file_list):
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / 'model/models'
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for f in file_list:
        file_path = model_dir / f
        if not file_path.exists():
            missing.append(f)
    
    if missing:
        logging.info("Some model files are missing. Downloading...")
        download_models(repo_id, missing)
        return False
    else:
        logging.info("All model files are present.")
        return True

def download_models(repo_id, file_list, target_subdir='model/models'):
    base_dir = Path(__file__).resolve().parent.parent
    local_dir = base_dir / target_subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Start: {repo_id}")
    logging.info(f"Name File: {local_dir}")
    logging.info("-" * 50)

    downloaded_files = []

    for file_name in file_list:
        try:
            logging.info(f"Loading: {file_name}...")
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )

            logging.info(f"Successfully: {file_name}")
            downloaded_files.append(file_path)
            
        except EntryNotFoundError:
            logging.error(f"Error : File '{file_name}' is not exists in repo.")
        except RepositoryNotFoundError:
            logging.error(f"Error: Dont found Repo '{repo_id}'.")
        except Exception as e:
            logging.error(f"Error {file_name}: {e}")
    logging.info("-" * 50)
    logging.info(f"Success! Complete {len(downloaded_files)}/{len(file_list)} files.")
    return downloaded_files

# --- Cấu hình ---


# --- Thực thi ---
if __name__ == "__main__":
    #download_models(MODEL_REPO, FILES_TO_DOWNLOAD)
    MODEL_REPO = "Trank123/API_LungCancer"
    FILES_TO_DOWNLOAD = [
        "best_pneumonia_classifier.pt",
        "best_pneumonia_classifier_mobilenetv2.pt",
        "mobilenetv2_lung_finetuned.onnx",
        "mobilenetv2_lung_finetuned.onnx.data",
        "resnet18_lung_finetuned.onnx",
        "resnet18_lung_finetuned.onnx.data"
    ]
    check_folder(MODEL_REPO, FILES_TO_DOWNLOAD)