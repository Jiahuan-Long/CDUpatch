import os
import shutil

# 数据集根目录（请根据实际情况修改）
base_dir = r"/home/admin1/datasets/DroneVehicle"

# 三个主文件夹：test, train, val
main_folders = ["train"]


def copy_files(src_dir, dst_dir, suffix=""):
    """
    复制 src_dir 中的所有文件到 dst_dir。
    如果提供了 suffix，则在文件名（不含扩展名）后添加该后缀再复制。
    如果目标文件已存在，则加上 _copy 后缀以避免冲突。
    """
    if src_dir is None or not os.path.exists(src_dir):
        return
    for file in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file)
        if os.path.isfile(src_file):
            base, ext = os.path.splitext(file)
            new_name = base + suffix + ext
            dst_file = os.path.join(dst_dir, new_name)
            # 如果目标文件已存在，则进一步修改名称
            i = 1
            while os.path.exists(dst_file):
                dst_file = os.path.join(dst_dir, f"{base}{suffix}_copy{i}{ext}")
                i += 1
            shutil.copy(src_file, dst_file)


for main_folder in main_folders:
    main_path = os.path.join(base_dir, main_folder)

    # 初始化各文件夹路径
    img_folder = None
    imgr_folder = None
    label_folder = None
    labelr_folder = None

    # 遍历主文件夹下的子文件夹，判断文件夹名称
    for d in os.listdir(main_path):
        full_path = os.path.join(main_path, d)
        if os.path.isdir(full_path):
            d_lower = d.lower()
            # 图片文件夹：区分 img 和 imgr（imgr中复制时需添加后缀）
            if d_lower.endswith("img") and "imgr" not in d_lower:
                img_folder = full_path
            elif "imgr" in d_lower:
                imgr_folder = full_path
            # 标签文件夹：区分 label 和 labelr
            elif d_lower.endswith("label") and "labelr" not in d_lower:
                label_folder = full_path
            elif "labelr" in d_lower:
                labelr_folder = full_path

    # 新建合并后的文件夹（在当前主文件夹下）
    merged_img_dir = os.path.join(main_path, "merged_img")
    merged_label_dir = os.path.join(main_path, "merged_label")
    os.makedirs(merged_img_dir, exist_ok=True)
    os.makedirs(merged_label_dir, exist_ok=True)

    # 合并图片：img 文件夹直接复制，imgr 文件夹复制时添加后缀 "r"
    copy_files(img_folder, merged_img_dir, suffix="")
    copy_files(imgr_folder, merged_img_dir, suffix="r")

    # 合并标签：同样对 labelr 添加后缀 "r"
    copy_files(label_folder, merged_label_dir, suffix="")
    copy_files(labelr_folder, merged_label_dir, suffix="r")

    print(f"在 {main_path} 下，图片已合并到 {merged_img_dir}，标签已合并到 {merged_label_dir}。")
