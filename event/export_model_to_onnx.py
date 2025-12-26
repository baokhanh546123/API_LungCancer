from torchvision import models , transforms
from pathlib import Path
from typing import List , Any , Tuple
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch , torch.nn as nn , os , cv2 , numpy as np , matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
labels = ['NORMAL', 'PNEUMONIA']

current_path = Path(__file__).parent.parent

model_path = current_path/'model/resnet18_lung_finetuned.pt'
model_export = current_path/'model/resnet18_lung_finetuned.onnx'
model_name = 'restnet18'
input_size = 224

def init_model(model_name : str , num_cls : int , feature_extract : bool = True) -> Tuple[nn.Module, int]:
  model_ft = None
  input_size = 224

  if model_name == 'restnet18':
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Dropout(0.5) , nn.Linear(num_ftrs, num_cls))
    if feature_extract:
      for name , param in model_ft.named_parameters():
        if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
          param.requires_grad = True
        else:
          param.requires_grad = False
  else:
    raise NotImplemented(f"Model '{model_name}' is not supported yet.")

  print(f"Initialized {model_name} with {num_classes} classes.")
  return model_ft , input_size


def export_onnx():
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained weights not found at: {model_path}. Please ensure your training script saved the file correctly.")

  model, onnx_input_size = init_model(model_name , num_classes)
  model = model.to(device)
  state_dict = torch.load(model_path , map_location=device)
  model.load_state_dict(state_dict)
  model.eval()

  dummy_input = torch.randn(1, 3, onnx_input_size, onnx_input_size, device=device)
  torch.onnx.export(model, dummy_input,model_export, export_params=True, opset_version=18, do_constant_folding=True, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

try:
  export_onnx()
except FileNotFoundError as e:
  print(e)
except Exception as e :
  print(e)