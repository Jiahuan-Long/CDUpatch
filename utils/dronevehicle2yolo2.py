import os
import xml.etree.ElementTree as ET

# 定义类别及其对应的索引
classes = ["car", "truck", "bus", "van", "freight_car"]
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# 数据集根目录
dataset_root = r"/home/admin1/datasets/DroneVehicle"


def convert_annotation(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图像尺寸
    size = root.find("size")
    img_width = float(size.find("width").text)
    img_height = float(size.find("height").text)

    with open(txt_path, "w") as out_file:
        # 遍历每个目标
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in classes:
                continue  # 忽略不在目标类别内的目标

            polygon = obj.find("polygon")
            if polygon is None:
                continue  # 若没有 polygon 信息则跳过

            # 读取 polygon 中的四个点坐标
            xs = []
            ys = []
            for i in range(1, 5):
                x = float(polygon.find(f"x{i}").text)
                y = float(polygon.find(f"y{i}").text)
                xs.append(x)
                ys.append(y)

            # 计算最小外接矩形
            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)

            # 计算中心点坐标和宽高，并归一化
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            bbox_width = (xmax - xmin) / img_width
            bbox_height = (ymax - ymin) / img_height

            label = class_to_idx[cls_name]
            out_file.write(f"{label} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")


# 递归遍历数据集目录，查找 XML 文件
for root_dir, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.endswith(".xml"):
            xml_file = os.path.join(root_dir, file)
            txt_file = os.path.join(root_dir, os.path.splitext(file)[0] + ".txt")
            convert_annotation(xml_file, txt_file)
            print(f"已转换: {xml_file} -> {txt_file}")
