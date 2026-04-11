import os
import imageio
from natsort import natsorted  # 按数字排序文件名

def make_gif_from_masked_patch(image_folder, gif_path, suffix='_adv_patch_masked.png', duration=0.3):
    # 获取所有匹配的图像文件
    files = [f for f in os.listdir(image_folder) if f.endswith(suffix)]
    files = natsorted(files)  # 自然排序，如 1, 2, 10 而非 1, 10, 2

    images = []
    for file in files:
        img_path = os.path.join(image_folder, file)
        image = imageio.imread(img_path)
        images.append(image)

    imageio.mimsave(gif_path, images, duration=duration)
    print(f"✅ GIF 已保存到: {gif_path}")

# 示例调用
make_gif_from_masked_patch(
    image_folder='patchTraining/training_run_20250519_141258/epoch_images',
    gif_path='training_run_20250519_141258_purepatch.gif',
    suffix='_pure_patch.png',
    duration=0.1  # 每帧持续 0.4 秒
)