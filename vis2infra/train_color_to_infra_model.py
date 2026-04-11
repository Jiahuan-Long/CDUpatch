import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib
import matplotlib.pyplot as plt

# -------- CONFIGURATION --------
color_dir = r"C:\Users\a\Downloads\gradient_images\gradient_images\vis"  # 可见光图像文件夹
infra_dir = r"C:\Users\a\Downloads\gradient_images\gradient_images\infra"  # 红外（灰度）图像文件夹
image_size = (500, 500)  # 使用全分辨率
model_save_path = 'color2infra_model.pkl'
predict_image_path = r"C:\Users\a\Downloads\gradient_images\predict_patch.png"
n_iterations = 10  # 迭代次数


# -------- DATA LOADING --------
def load_image_pairs(color_dir, infra_dir, size):
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

        X.append(color_array.reshape(-1, 3))  # RGB channels
        y.append(infra_array.flatten())  # 灰度目标

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


# -------- INFERENCE --------
def predict_infra(model, color_image_path, size):
    img = Image.open(color_image_path).resize(size).convert('RGB')
    arr = np.asarray(img) / 255.0
    X = arr.reshape(-1, 3)
    pred = model.predict(X)
    # 将预测结果转换为灰度图（注意这里直接返回 numpy 数组供 plt.imshow 使用）
    pred_img = (pred.reshape(size) * 255).clip(0, 255).astype(np.uint8)
    return pred_img


# -------- MODEL TRAINING --------
def train_model_iterative(X_train, X_test, y_train, y_test, n_iter):
    # 使用 warm_start=True 且 max_iter=1，每次调用 fit() 都只进行一次迭代
    model = MLPRegressor(hidden_layer_sizes=(64, 64), warm_start=True, max_iter=1, verbose=True, random_state=42)
    for iteration in range(n_iter):
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Iteration {iteration + 1}/{n_iter}: Test score (R^2): {score:.4f}")

        # 用当前模型预测指定图片，并进行可视化
        pred_img = predict_infra(model, predict_image_path, image_size)
        plt.figure()
        plt.imshow(pred_img, cmap='gray')
        plt.title(f"Iteration {iteration + 1}")
        plt.axis('off')
        plt.show()  # 如果在非交互环境下运行，可考虑使用 plt.pause(0.5) 后 plt.close()
    return model


# -------- MAIN WORKFLOW --------
if __name__ == '__main__':
    print("Loading data...")
    X, y = load_image_pairs(color_dir, infra_dir, image_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model iteratively and visualizing predictions...")
    model = train_model_iterative(X_train, X_test, y_train, y_test, n_iterations)
    joblib.dump(model, model_save_path)
    print("Model saved to", model_save_path)
