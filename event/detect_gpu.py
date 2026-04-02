import subprocess , os , importlib.util , platform , re , json

class Detect():
    def __init__(self):
        self._gpu_info = None

    def info(self):
        subprocess.run(["chmod", "+x", "./event/pre_run.sh"], check=True)
    
        result = subprocess.run(
            ["./event/pre_run.sh"], 
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        
        self._gpu_info = json.loads(result.stdout.strip())
        return self._gpu_info
    
    def _is_torch_installed(self):
        return importlib.util.find_spec("torch") is not None
    
    def install_library(self):
        if self._is_torch_installed():
            print("PyTorch have already exists")
            return True

        info = self.info()
        os = info['os']
        vendor = info['vendor']
        chipname = info['chipset_name']
        gpu_name = info['gpu_name']
        gpu_count = info['gpu_count']
        vram = info['vram_mb']
        tier = info['tier']

        print("="*30)
        print(f"   • OS           : {os}")
        print(f"   • Vendor       : {vendor}")
        print(f"   • Chipset      : {chipname}")
        print(f"   • GPU Name     : {gpu_name}")
        print(f"   • VRAM         : {vram} MB")
        print(f"   • Tier         : {tier.upper()}")
        print("="*30)
        
        install_cmd = ["pip", "install", "torch", "torchvision", "torchaudio"]
        index_url = None
        message = ""
        
        if (os == 'macos' or os == 'darkwin') and vendor == 'apple':
            message = f'Apple Silicon {chipname}' 
        elif vendor == 'nvidia':
            if tier in ['medium' , 'strong']:
                index_url = "https://download.pytorch.org/whl/cu126"
                message = f"NVIDIA {gpu_name}"
            else:
                index_url = "https://download.pytorch.org/whl/cpu"
                message = f"{gpu_name} , Your GPU is weak , fallback CPU"
        else:
            index_url = "https://download.pytorch.org/whl/cpu"
        
        print(message)
        if index_url:
            install_cmd.extend(["--index-url", index_url])
        print(f"Processing command: {' '.join(install_cmd)}")
        try:
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            print("Pytorch have installed successly")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed install:")
            print(e.stderr if e.stderr else str(e))
            return False
        except Exception as e:
            print(f"Error : {e}")
            return False
            
if __name__== "__main__":
    detector = Detect()

    # Lấy thông tin GPU bất kỳ lúc nào
    gpu_info = detector.info()
    print(gpu_info)

    # Cài PyTorch (chỉ chạy 1 lần)
    detector.install_library()