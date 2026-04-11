import joblib
from PIL import Image
import numpy as np

# 载入保存的模型
model = joblib.load('color2infra_model.pkl')

# 定义与训练时相同的图片尺寸
image_size = (500, 500)

# 定义推理时的预处理和预测函数（与训练代码中保持一致）
def predict_infra(model, color_image_path, size):
    img = Image.open(color_image_path).resize(size).convert('RGB')
    arr = np.asarray(img) / 255.0
    X = arr.reshape(-1, 3)
    pred = model.predict(X)
    pred_img = (pred.reshape(size) * 255).clip(0, 255).astype(np.uint8)
    gray_img = Image.fromarray(pred_img, mode='L')
    rgb_img = gray_img.convert('RGB')
    return rgb_img

# 指定需要进行推理的图片路径
predict_image_path = r"/home/admin1/jiahuan/advpatch/YOLOv3_Demo/patchTraining/training_run_20250315_175327/epoch_images/74_patch.png"

# 调用函数进行推理
result_image = predict_infra(model, predict_image_path, image_size)

# 显示或保存预测结果
result_image.show()            # 直接在系统图片查看器中显示
result_image.save('result.png')  # 保存预测结果到 result.png
