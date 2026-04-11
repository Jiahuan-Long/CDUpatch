"""" 
划分训练集和测试集
"""
import os
import shutil

imgdir = "D:/VSCode/Datasets/patchCars/trainset/imgs"
labels_dir = 'D:/VSCode/Datasets/patchCars/trainset/yolo-labels'
desdir = "D:/VSCode/Datasets/patchCars/testset"  #
num = 0
# Loop over cleane beelden  # 对干净样本进行loop
for imgfile in os.listdir(imgdir):  # 得到所有文件名
    print("new image")  # 对路径下的图片进行遍历
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
        name = os.path.splitext(imgfile)[0]  # image name w/o extension
        txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
        cleanname = name + ".png"

        txtpath = os.path.abspath(os.path.join(
            labels_dir, txtname))
        imgpath = os.path.abspath(os.path.join(imgdir, cleanname))  # 这里开始保存图片，
        des_txtpath = os.path.abspath(
            os.path.join(desdir, 'yolo-labels/', txtname))
        des_imgpath = os.path.abspath(
            os.path.join(desdir, 'imgs/', cleanname))  # 这里开始保存图片，

        shutil.move(txtpath, des_txtpath)
        shutil.move(imgpath, des_imgpath)

        if num > 198:
            break
        num += 1
print("ALL DONE!")