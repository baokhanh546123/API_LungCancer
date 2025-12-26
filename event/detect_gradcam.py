import sys , subprocess , importlib.util
from event.detect_gpu import Detect

class GradCam(Detect):
    def __init__(self):
        super().__init__()

    def is_installed_grad_cam(self):
        return importlib.util.find_spec("pytorch_grad_cam") is not None
    
    def install_grad_cam(self, min_cores=128):
        torch_installed = super().install_library(min_cores)
        if not torch_installed:
            print('Error')
            return False
        if self.is_installed_grad_cam():
            print("[INFO] Grad-CAM đã được cài đặt sẵn.")
            return True
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "grad-cam"]
            subprocess.run(cmd, check=True)
            print("[SUCCESS] Cài đặt Grad-CAM hoàn tất.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Lỗi khi cài đặt Grad-CAM: {e}")
            return False

"""if __name__ == "__main__":
    installer = GradCam()
    
    # Chạy cài đặt toàn bộ
    success = installer.install_grad_cam(min_cores=128)
    
    if success:
        print("\n=== HOÀN TẤT TOÀN BỘ QUÁ TRÌNH ===")
    else:
        print("\n=== CÓ LỖI XẢY RA ===")
"""