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

import logging
import time
from pathlib import Path
from huggingface_hub import hf_hub_download, EntryNotFoundError, RepositoryNotFoundError

def download_models(repo_id, file_list, target_subdir='model/models', retries=3):
    base_dir = Path(__file__).resolve().parent.parent
    local_dir = base_dir / target_subdir
    local_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Start: {repo_id}")
    logging.info(f"Name File: {local_dir}")
    logging.info("-" * 50)

    downloaded_files = []

    for file_name in file_list:
        for attempt in range(1, retries + 1):
            try:
                logging.info(f"Loading: {file_name} (attempt {attempt}/{retries})...")
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_name,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                logging.info(f"Successfully: {file_name}")
                downloaded_files.append(file_path)
                break  # thành công, thoát vòng lặp thử lại

            except EntryNotFoundError:
                logging.error(f"Error : File '{file_name}' does not exist in repo. Stop retrying.")
                break  # không cần thử lại

            except RepositoryNotFoundError:
                logging.error(f"Error: Repo '{repo_id}' not found. Stop retrying.")
                break

            except Exception as e:
                logging.error(f"Error downloading {file_name} (attempt {attempt}): {e}")
                if attempt == retries:
                    logging.error(f"Failed to download {file_name} after {retries} attempts.")
                else:
                    wait_time = 2 ** (attempt - 1)  # tăng dần thời gian chờ: 1s, 2s, 4s...
                    logging.info(f"Retrying {file_name} in {wait_time}s...")
                    time.sleep(wait_time)

    logging.info("-" * 50)
    logging.info(f"Success! Completed {len(downloaded_files)}/{len(file_list)} files.")
    return downloaded_files

if __name__ == "__main__":
    #download_models(MODEL_REPO, FILES_TO_DOWNLOAD)
    MODEL_REPO = "Trank123/API_LungCancer"
    FILES_TO_DOWNLOAD = [
        "best_pneumonia_classifier.pt",
        "best_pneumonia_classifier_mobilenetv2.pt",
        "mobilenetv2_lung_finetuned.onnx",
        "mobilenetv2_lung_finetuned.onnx.data",
        "resnet18_lung_finetuned.onnx",
        "resnet18_lung_finetuned.onnx.data",
        "keras_cnn_xray.onnx"
    ]
    check_folder(MODEL_REPO, FILES_TO_DOWNLOAD)
