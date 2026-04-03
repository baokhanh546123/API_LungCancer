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

def load_model(model_name : str , num_cls : int , path : Path , feature_extract : bool = True ) -> nn.Module:
  model , _ = init_model(model_name , num_cls , feature_extract)
  if not path.is_file():
    raise FileNotFoundError(f"Model weights file not found at: {path}")

  state_dict = torch.load(path , map_location=device)
  model.load_state_dict(state_dict)
  model.to(device) # Move the model to the specified device
  model.eval()
  return model

def predict_img(model : nn.Module , img_path : Path , trans : transforms.Compose , labels : List[str]):
  try:
    img = Image.open(img_path).convert('RGB')
  except FileNotFoundError:
    print(f"Image not found at {img_path}")
    return None

  input_tensor = trans(img)
  input_batch = input_tensor.unsqueeze(0).to(device)

  with torch.no_grad():
    output = model(input_batch)
    prob = torch.nn.functional.softmax(output[0] , dim = 0 )
    pred_index = torch.argmax(prob).item()
    confidence = prob[pred_index].item() * 100
    pred_class = labels[pred_index]

  return pred_class , confidence , prob.cpu().numpy()
  

current_path = Path(__file__).parent.parent

img_path = current_path / 'event/normal.jpeg'
model_path = current_path / 'model/resnet18_lung_finetuned.pt'
model_name = 'restnet18'

try:
  if not model_path.is_file():
    raise FileNotFoundError(f"Model weights file not found at: {model_path}. Did you train and save it?")

  image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  
  infer_model = load_model(model_name , num_classes , model_path)
  pred_class , confidence , raw_probs = predict_img(infer_model , img_path , image_transforms , labels)
  if pred_class:
        print("\n--- Prediction Result ---")
        print(f"File: {img_path.name}")
        print(f"Prediction: {pred_class}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Raw Probabilities: NORMAL={raw_probs[0]:.4f}, PNEUMONIA={raw_probs[1]:.4f}")

except FileNotFoundError as e:
    print(f"SETUP ERROR: {e}")
except Exception as e:
    print(f"An unexpected error occurred during inference: {e}")


target_layer = infer_model.layer4[-1].conv2

def gradcam_for_image(model, img_path, transform, target_layer=None, method="gradcam"):
    if target_layer is None:
        target_layer = model.layer4[-1].conv2

    img_pil = Image.open(img_path).convert("RGB")
    original_img_np = np.array(img_pil).astype(np.float32) / 255.0
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.softmax(out, dim=1)
        pred_class = int(probs.argmax(dim=1).item())
        pred_score = float(probs.max().item())

    cam_method = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "eigencam": EigenCAM}[method.lower()]
    cam = cam_method(model=model, target_layers=[target_layer])

    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    if grayscale_cam.ndim == 3:
        gcam = grayscale_cam[0]
    else:
        gcam = grayscale_cam
    gcam = np.float32(gcam)
    gcam = (gcam - gcam.min()) / (gcam.max() - gcam.min() + 1e-8)

    heatmap = np.uint8(255 * gcam)
    heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colormap = cv2.cvtColor(heatmap_colormap, cv2.COLOR_BGR2RGB)
    resized_grayscale_cam = cv2.resize(gcam, (original_img_np.shape[1], original_img_np.shape[0]))
    cam_image = show_cam_on_image(original_img_np, resized_grayscale_cam, use_rgb=True)


    return {
        "grayscale_cam": gcam,
        "heatmap_colormap": heatmap_colormap,
        "cam_on_image": cam_image,
        "pred_class": pred_class,
        "pred_score": pred_score
    }

def show_side_by_side(orig_path, cam_res, figsize=(12,4)):
    orig = Image.open(orig_path).convert("RGB")
    #orig_tittle = orig_path.split('/')[-2]
    fig, axes = plt.subplots(1,3, figsize=figsize)
    axes[0].imshow(orig); axes[0].axis('off')
    axes[1].imshow(cam_res['heatmap_colormap']); axes[1].set_title("Heatmap"); axes[1].axis('off')
    axes[2].imshow(cam_res['cam_on_image']); axes[2].set_title(f"Overlay (pred {cam_res['pred_class']} {cam_res['pred_score']:.2f})"); axes[2].axis('off')
    #plt.show()
    plt.savefig(current_path/'event/test1.jpeg')
    plt.close()


res = gradcam_for_image(infer_model , img_path , image_transforms , method='eigencam')
show_side_by_side(img_path , res)


