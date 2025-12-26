import subprocess , os , importlib.util , platform , re 

class Detect:
    def __init__(self):
        self.__gpus = []
        self.system = platform.system()

    def get_info_gpu(self):
        self.__gpus = []  
        
        try:
            try:
                nvidia_out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                
                if nvidia_out:
                    for line in nvidia_out.split('\n'):
                        name, mem = line.split(',')
                        self.__gpus.append({
                            'name': name.strip(),
                            'vendor': 'nvidia',
                            'cores': 2048 if int(mem) > 4000 else 512, # Ước tính tạm thời
                            'vram': int(mem)
                        })
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            self._check_system_gpu()
            
        except Exception as e:
            print(f"Lỗi không xác định khi quét phần cứng: {e}")
            
        return self.__gpus

    def _check_system_gpu(self):
        try:
            if self.system == "Windows":
                cmd = "wmic path win32_VideoController get Name, AdapterRAM /format:list"
                out = subprocess.check_output(cmd, shell=True).decode()
                names = re.findall(r"Name=(.*)", out)
                if names:
                    for name in names:
                        if not any(g['name'] == name.strip() for g in self.__gpus):
                            self.__gpus.append({'name': name.strip(), 'vendor': 'common', 'cores': 256})

            elif self.system == "Darwin":
                cmd = "system_profiler SPDisplaysDataType"
                out = subprocess.check_output(cmd).decode()
                cores_match = re.search(r"Total Number of Cores:\s+(\d+)", out)
                name_match = re.search(r"Chipset Model:\s+(.*)", out)
                if name_match:
                    self.__gpus.append({
                        'name': name_match.group(1).strip(),
                        'vendor': 'apple',
                        'cores': int(cores_match.group(1)) if cores_match else 8
                    })

            elif self.system == "Linux":
                out = subprocess.check_output("lspci | grep -i vga", shell=True).decode()
                if out:
                    self.__gpus.append({'name': out.strip(), 'vendor': 'linux_generic', 'cores': 256})
        except Exception:
            pass

    def is_torch_installed(self) -> bool:
        return importlib.util.find_spec("torch") is not None
        

    def install_library(self, min_cores=128):
        try:
            if self.is_torch_installed():
                print("PyTorch đã được cài đặt sẵn. Bỏ qua.")
                return True

            gpus = self.get_info_gpu()
            index_url = ""
            
            if not gpus:
                print("[WARN] Không tìm thấy GPU. Sử dụng phiên bản CPU.")
                index_url = "https://download.pytorch.org/whl/cpu"
            else:
                primary = gpus[0]
                print(f"[INFO] Phát hiện: {primary['name']} ({primary['cores']} cores)")

                if primary['cores'] < min_cores:
                    print(f"[WARN] GPU quá yếu ({primary['cores']} < {min_cores}). Cài bản CPU.")
                    index_url = "https://download.pytorch.org/whl/cpu"
                else:
                    if primary['vendor'] == 'nvidia':
                        index_url = "https://download.pytorch.org/whl/cu126"
                    elif primary['vendor'] == 'apple':
                        index_url = ""
                    else:
                        index_url = "https://download.pytorch.org/whl/cpu"

            install_cmd = ["pip", "install", "torch", "torchvision", "torchaudio"]
            if index_url:
                install_cmd.extend(["--index-url", index_url])

            print(f"[Đang chạy: {' '.join(install_cmd)}")
            
            result = subprocess.run(install_cmd, check=True)
            return result.returncode == 0

        except Exception as e:
            print(f"[ERROR] Quá trình cài đặt thất bại: {e}")
            return False

"""if __name__ == "__main__":
    detector = Detect()
    success = detector.install_library(min_cores=128)
    if success:
        print("--- Hoàn tất thành công ---")
    else:
        print("--- Thất bại ---")"""