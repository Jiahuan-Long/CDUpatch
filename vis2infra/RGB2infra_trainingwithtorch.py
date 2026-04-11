# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------- CONFIGURATION --------
color_dir = r"C:\Users\a\Downloads\gradient_images\gradient_images\vis"  # 可见光图像文件夹
infra_dir = r"C:\Users\a\Downloads\gradient_images\gradient_images\infra"  # 红外（灰度）图像文件夹
predict_image_path = r"C:\Users\a\Downloads\gradient_images\predict_patch.png"  # 用于验证的图片
image_size = (500, 500)  # 图像尺寸
batch_size = 2048
learning_rate = 1e-4
epochs = 20
model_save_dir = 'epoch_models'
predicted_image_dir = 'epoch_images'
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(predicted_image_dir, exist_ok=True)

# -------- MODEL DEFINITION --------
class RGB2InfraNet(nn.Module):
    def __init__(self):
        super(RGB2InfraNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -------- DATA LOADING FUNCTION --------
def load_dataset(color_dir, infra_dir, size):
    X, y = [], []
    for fname in os.listdir(color_dir):
        color_path = os.path.join(color_dir, fname)
        infra_path = os.path.join(infra_dir, fname)
        if not os.path.exists(infra_path):
            continue

        color_img = Image.open(color_path).resize(size).convert('RGB')
        infra_img = Image.open(infra_path).resize(size).convert('L')

        color_array = np.asarray(color_img) / 255.0
        infra_array = np.asarray(infra_img) / 255.0

        X.append(color_array.reshape(-1, 3))
        y.append(infra_array.flatten())

    X = np.vstack(X)
    y = np.hstack(y)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().unsqueeze(1)

    return X_tensor, y_tensor

# -------- INFERENCE FUNCTION --------
def predict_and_save_image(model, image_path, size, save_path):
    img = Image.open(image_path).resize(size).convert('RGB')
    arr = np.asarray(img) / 255.0
    X = torch.from_numpy(arr.reshape(-1, 3)).float().to(next(model.parameters()).device)

    with torch.no_grad():
        pred = model(X).cpu().numpy()

    pred_img = (pred.reshape(size) * 255).clip(0, 255).astype(np.uint8)
    plt.imsave(save_path, pred_img, cmap='gray')

# -------- MAIN TRAINING FUNCTION --------
def main():
    # 加载数据
    print("Loading data...")
    X, y = load_dataset(color_dir, infra_dir, image_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size
    )

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGB2InfraNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证模型
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                criterion(model(xb.to(device)), yb.to(device)).item() for xb, yb in test_loader
            ) / len(test_loader)

        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}')

        # 每个epoch保存模型
        epoch_model_path = os.path.join(model_save_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), epoch_model_path)

        # 每个epoch进行预测并保存图片
        predicted_image_path = os.path.join(predicted_image_dir, f'pred_epoch_{epoch + 1}.png')
        predict_and_save_image(model, predict_image_path, image_size, predicted_image_path)

if __name__ == '__main__':
    main()
