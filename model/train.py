import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from resnet import M64RN4

class SARWindDataset(Dataset):
    def __init__(self):
        self.num_samples = 1000
        self.data = torch.randn(self.num_samples, 1, 20, 20) 

        directions_deg = np.random.uniform(0, 360, self.num_samples)
        directions_rad = np.radians(directions_deg)

        self.labels = torch.tensor(
            np.stack((np.sin(directions_rad), np.cos(directions_rad)), axis = 1), 
            dtype = torch.float32
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = M64RN4().to(device)
    dataset = SARWindDataset()
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr = 0.001) # type: ignore
    epochs = 50

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1} / {epochs}], Loss: {running_loss/len(dataloader):.4f}")

    os.makedirs('weights', exist_ok = True)
    save_path = 'weights/m64rn4_weights.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")

if __name__ == "__main__":
    train_model()