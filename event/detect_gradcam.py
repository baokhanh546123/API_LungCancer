import sys , subprocess , importlib.util
from event.detect_gpu import Detect

class GradCam(Detect):
    def __init__(self):
        super().__init__()

    def is_installed_grad_cam(self):
        return importlib.util.find_spec("pytorch_grad_cam") is not None
    
    def install_grad_cam(self):
        torch_installed = super().install_library()
        if not torch_installed:
            print('Error')
            return False
        if self.is_installed_grad_cam():
            print("[GradCam have installed.")
            return True
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "grad-cam"]
            subprocess.run(cmd, check=True)
            print("GradCam have installed successly")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
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