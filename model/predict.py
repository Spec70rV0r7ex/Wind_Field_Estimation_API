import torch
import numpy as np
from model.resnet import M64RN4

def load_model(weights_path: str = "model/weights/m64rn4_paper_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M64RN4()

    model.load_state_dict(torch.load(weights_path, map_location = device))
    model.to(device)
    model.eval()

    return model, device

def predict_wind_directions(model, device, sar_patches: np.ndarray) -> np.ndarray:
    for i in range(sar_patches.shape[0]):
        mean_val = np.mean(sar_patches[i])
        std_val = np.std(sar_patches[i])
        sar_patches[i] = (sar_patches[i] - mean_val) / (std_val + 1e-8)

    tensor_patches = torch.tensor(sar_patches, dtype = torch.float32)
    if len(tensor_patches.shape) == 3:
        tensor_patches = tensor_patches.unsqueeze(1)
        
    tensor_patches = tensor_patches.to(device)

    with torch.no_grad():
        predictions = model(tensor_patches)
        angles_rad = torch.atan2(predictions[:, 0], predictions[:, 1]) 
        angles_deg = torch.rad2deg(angles_rad).cpu().numpy() % 360

    return angles_deg
