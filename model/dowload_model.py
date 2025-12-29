import os
import logging
import requests
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Download multiple ML model files from Google Drive.
    """

    def __init__(
        self,
        file_map: Dict[str, str],
        download_dir: Path | str,
        chunk_size: int = 8192,
    ):
        """
        :param file_map: dict {filename: google_drive_file_id}
        :param download_dir: directory to save downloaded models
        :param chunk_size: download chunk size
        """
        self.file_map = file_map
        self.download_dir = Path(download_dir)
        self.chunk_size = chunk_size

        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _build_drive_url(self, file_id: str) -> str:
        return f"https://drive.google.com/uc?id={file_id}&export=download"

    def download_file(self, filename: str, file_id: str) -> None:
        save_path = self.download_dir / filename

        if save_path.exists():
            logger.info("Model already exists: %s (skip)", save_path)
            return

        url = self._build_drive_url(file_id)
        logger.info("Downloading %s ...", filename)

        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)

            logger.info("Downloaded successfully: %s", filename)

        except requests.RequestException as exc:
            logger.error("Failed to download %s: %s", filename, exc)
            raise

    def download_all(self) -> None:
        """
        Download all files in file_map
        """
        for filename, file_id in self.file_map.items():
            self.download_file(filename, file_id)


MODEL_FILES = {
    "best_pneumonia_classifier_mobilenetv2.pt": "FILE_ID_1",
    "best_pneumonia_classifier.pt": "FILE_ID_2",
    "mobilenetv2_lung_finetuned.onnx": "FILE_ID_3",
    "mobilenetv2_lung_finetuned.onnx.data": "FILE_ID_4",
    "resnet18_lung_finetuned.onnx": "FILE_ID_5",
    "resnet18_lung_finetuned.onnx.data": "FILE_ID_6",
}
from pathlib import Path
dowloader = ModelDownloader(
    file_map=MODEL_FILES,
    download_dir=Path(__file__).parent / "models",
)
dowloader.download_all()