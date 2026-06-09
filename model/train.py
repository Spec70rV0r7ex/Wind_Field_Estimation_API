import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

SAVE_DIR = './weights'
os.makedirs(SAVE_DIR, exist_ok = True)
SAVE_PATH = os.path.join(SAVE_DIR, 'm64rn4_paper_weights.pth')

class ZanchettaLoss(nn.Module):
    def __init__(self):
        super(ZanchettaLoss, self).__init__()

    def forward(self, preds, targets):
        cos_diff = torch.sum(preds * targets, dim = 1)
        loss = 1.0 - torch.square(cos_diff)
        return torch.mean(loss)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class M64RN4(nn.Module):
    def __init__(self):
        super(M64RN4, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, 1))
        return F.normalize(x, p = 2, dim = 1)

class SARWindDataset(Dataset):
    # Updated default data_dir to point to your local data folder
    def __init__(self, data_dir = './data/SAR_Wind_Dataset_2024'):
        print("Loading 49x49 SAR Dataset...")
        self.data = np.load(os.path.join(data_dir, 'SAR_X_49_2024.npy'))
        self.labels = np.load(os.path.join(data_dir, 'WIND_Y_49_2024.npy'))

        for i in range(self.data.shape[0]):
            mean_val = np.mean(self.data[i])
            std_val = np.std(self.data[i])
            self.data[i] = (self.data[i] - mean_val) / (std_val + 1e-8)

        self.num_samples = self.data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.data[idx].copy()
        label = self.labels[idx]

        angle_rad = np.arctan2(label[0], label[1])
        angle_deg = np.degrees(angle_rad)

        if np.random.rand() > 0.5:
            img = np.flip(img, axis = 2).copy()
            angle_deg = -angle_deg

            img = np.flip(img, axis = 1).copy()
            angle_deg = 180.0 - angle_deg

        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k = k, axes = (1, 2)).copy()
            angle_deg = angle_deg + (k * 90.0)

        angle_deg = angle_deg % 360
        new_angle_rad = np.radians(angle_deg)
        new_label = np.array([np.sin(new_angle_rad), np.cos(new_angle_rad)], dtype = np.float32)

        return torch.tensor(img), torch.tensor(new_label)

def angular_error(preds, targets):
    pred_angles = torch.atan2(preds[:, 0], preds[:, 1])
    target_angles = torch.atan2(targets[:, 0], targets[:, 1])
    diff = torch.abs(pred_angles - target_angles)
    diff = torch.min(diff, 2 * np.pi - diff)
    diff = torch.where(diff > np.pi/2, np.pi - diff, diff)
    return torch.mean(torch.rad2deg(diff)).item()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training initialized on device: {device}")
    
    model = M64RN4().to(device)
    
    criterion = ZanchettaLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001) # type: ignore
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 7)
    
    full_dataset = SARWindDataset()
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 2)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False, num_workers = 2)
    
    epochs = 200
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_loss, train_ang_err = 0.0, 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_ang_err += angular_error(outputs, targets) * inputs.size(0)

        model.eval()
        val_loss, val_ang_err = 0.0, 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_ang_err += angular_error(outputs, targets) * inputs.size(0)

        train_loss = train_loss / train_size
        train_ang_err = train_ang_err / train_size
        val_loss = val_loss / val_size
        val_ang_err = val_ang_err / val_size

        scheduler.step(val_loss)

        print(f"Ep {epoch+1:03d} | Loss: {train_loss:.4f} / Val Loss: {val_loss:.4f} (Err: {val_ang_err:.1f}°) | Time: {time.time() - start_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 25:
            print(f"Early stopping triggered. No improvement for 25 epochs.")
            break

if __name__ == '__main__':
    train_model()
