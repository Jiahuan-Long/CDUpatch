'''
可以计算recall和ASR
可以保存粘贴补丁后图片和检测图片
'''
import sys
import time
import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from utils_self import *
from load_data import PatchTransformer, PatchApplier, InriaDataset
# import json
# import copy
import utils_self
import load_data
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
import fnmatch

# #  # 配置地址
print("Setting everything up")
print("start training time : ", time.strftime('%Y-%m-%d %H:%M:%S'))
imgdir = 'D:/VSCode/Datasets/visDrone2019/VisDrone2019-DET-val/images'
labdir = "D:/VSCode/Datasets/visDrone2019/VisDrone2019-DET-val"

clean_labdir = labdir + '/' + 'labels'  # 原始标签路径

weightfile = "weights/yolov3.pt"
datayaml = "data/VisDrone.yaml"
imgsz = (640, 640)
device = select_device('')

patchfile = "patch2eval/v5/v5-combine-scale-1.png"  #

savedir = "patch2eval/v5/"  #

print("scale factor : ", load_data.SCALE)

print("patchfile : ", patchfile)
# print("test data : ", imgdir)  # 把测试补丁和测试数据打印记录
print("savedir : ", savedir)
# # 配置模型
model = DetectMultiBackend(weights = weightfile, device=device).eval()

patch_applier = PatchApplier().cuda()
patch_transformer = PatchTransformer().cuda()
# 超参设置
batch_size = 1

patch_img = utils_self.load_image_file(patchfile)
tf = transforms.ToTensor()
adv_patch_cpu = tf(patch_img)  # 补丁转换到3维tensor
adv_patch = adv_patch_cpu.cuda()

t0 = time.time()
print("Pre-setting Done!")

n_png_images = len(fnmatch.filter(os.listdir(imgdir), '*.png'))
n_jpg_images = len(fnmatch.filter(os.listdir(imgdir), '*.jpg'))
n_images = n_png_images + n_jpg_images  # 应该和n_images_clean一致
print("Total images : ", n_images)

if not os.path.exists(os.path.join(labdir, 'yolo-labels')):
    os.mkdir(os.path.join(labdir, 'yolo-labels'))

class_names = model.names
# Loop over cleane beelden  # 对干净样本进行loop
for imgfile in os.listdir(imgdir):  # 得到所有文件名
    # print("new image")  # 对路径下的图片进行遍历
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
        name = os.path.splitext(imgfile)[0]  # image name w/o extension
        # 将文件名分开
        txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
        txtpath = os.path.abspath(os.path.join(clean_labdir, txtname))

        # open beeld en pas aan naar yolo input size  # 打开图像并调整为yolo输入大小,对每一张图片进行操作
        imgfile = os.path.abspath(os.path.join(imgdir, imgfile))  # 拼接成绝对路径
        # print("image file path is ", imgfile)
        img = load_image_file(imgfile)  # 注意这里的不同

        w, h = img.size
        # print("original w = ", w, ", h = ", h)
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2   # dim_to_pad = 1 if w < h else 2  # 根据原始图片的宽高决定怎么填充
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new(
                    'RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new(
                    'RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))

        resize = transforms.Resize(imgsz)
        padded_img = resize(padded_img)

        # print("size after padding is ", padded_img.size)

        if os.path.getsize(txtpath):
            label = np.loadtxt(txtpath)
        else:
            label = np.ones([5])
        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        transform = transforms.ToTensor()
        padded_img = transform(padded_img).cuda()  # 变成了三通道tensor数据
        # padded_img = transform(padded_img_copy).cuda()
        img_fake_batch = padded_img.unsqueeze(0)  # [1, 3, 640 640]
        # lab_fake_batch = label.unsqueeze(0).cuda()  # 增维 [1,2,5]
        ###################### Adjust at 20240407 ######################
        #   检测
        boxes_cls = utils_self.do_detect_nms(model, img_fake_batch, conf_thresh=0.01, nms_thresh=0.4)
        boxes_cls = boxes_cls.cpu().numpy()
        boxes_padded_label = []  # 定义对特定类别的boxes
        # 如果检测结果中没有
        if 3 not in boxes_cls[:, 5]:
            continue
        for box in boxes_cls:
            cls_id = box[5]
            obj_conf = box[4]
            if (cls_id == 3):  # 怎么筛选？
                if obj_conf >= 0.4:
                    box = [box[-1], box[0], box[1], box[2], box[3]]
                    boxes_padded_label.append(box)   # 0.51040, 0.55054, 0.03355, 0.04826, 0.92339, 3.00000
        #  保存预测的检测框
        txtpath_clean_pre = os.path.abspath(os.path.join(
                    labdir, 'yolo-labels', txtname))
        textfile_clean_pre = open(txtpath_clean_pre, 'w+')  #   仅写入筛选后的数据
        for box in boxes_padded_label:
            #   此时boxex_cls的格式为[cls_id,x,y,w,h]
            obj_conf = box[4]
            if obj_conf > 0.4:
                textfile_clean_pre.write(f'{box[5]} {box[0]} {box[1]} {box[2]} {box[3]}\n')
        textfile_clean_pre.close()

        # lab_fake_batch = label.unsqueeze(0).cuda()  # 增维 [1,2,5]
        lab_fake_batch = torch.tensor(boxes_padded_label).unsqueeze(0).cuda()  # 增维 [1,2,5]

        # transformeer patch en voeg hem toe aan beeld  # 对补丁进行变换并添加到图片中
        try:
            adv_batch_t = patch_transformer(
                adv_patch, lab_fake_batch, imgsz[0], do_rotate=True, rand_loc=True)
        except:
            print(111)
        p_img_batch = patch_applier(
            img_fake_batch, adv_batch_t)  # [1,3,608,608]

        p_img = p_img_batch.squeeze(0)  # 降维，第一维压缩  [3,608,608]
        # p_img = F.interpolate(p_img,(darknet_model.height, darknet_model.width))  # 对补丁图片进行了插值
        p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())  # 得到添加补丁后图片
        
        properpatchedname = name + ".png"
        p_img_pil.save(os.path.join(
            savedir, 'proper_patched/', properpatchedname))  # 保存添加补丁图片
        
        #   检测
        boxes_cls = utils_self.do_detect_nms(model, p_img_pil, conf_thresh=0.01, nms_thresh=0.4)
        boxes_cls = boxes_cls.cpu()
        boxes_patched = []  # 定义对特定类别的boxes
        for box in boxes_cls:
            cls_id = box[5]
            obj_conf = box[4]
            if (cls_id == 3):  # 怎么筛选？
                if obj_conf >= 0.4:
                    boxes_patched.append(box)   # 0.51040, 0.55054, 0.03355, 0.04826, 0.92339, 3.00000
        '''增加作框图工作'''
        # patched_pre_name = name + "_pre.png"
        
        patched_pre_name = name + ".png"    #   不添加后缀
        patched_pre_dir = os.path.join(
            savedir, 'pre_patched/', patched_pre_name)

        plot_boxes(p_img_pil, boxes_patched, patched_pre_dir,
                   class_names=class_names)  # 这里画出来的预测框也是经过筛选后
        #   这里参考creation attack中的功能，将两部分置信度设置都保存下来
        txtpath_pre = os.path.abspath(os.path.join(
                    savedir, 'yolo-labels', txtname))
        textfile_pre = open(txtpath_pre, 'w+')  #   仅写入筛选后的数据
        
        txtpath_w_conf_pre = os.path.abspath(os.path.join(
                savedir, 'yolo-labels_w_conf', txtname))
        textfile_w_conf_pre = open(txtpath_w_conf_pre, 'w+')    #   写入所有数据，包括置信度信息
         
        for box in boxes_patched:
            #   此时boxex_cls的格式为[x,y,w,h,conf,cls_id]
            textfile_w_conf_pre.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} \n')
            obj_conf = box[4]
            if obj_conf > 0.4:
                textfile_pre.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} \n')
        textfile_pre.close()
        textfile_w_conf_pre.close()
        
t1 = time.time()
t_01 = (t1-t0)/60
print("Processing Done!")
print("Total Running Time : ", t_01, "minutes !")


if __name__ == '__main__':
    t0 = time.time()
    # pre_label_dir = savedir + '/' + 'yolo-labels' # 粘贴补丁后的预测结果
    pre_label_dir = os.path.abspath(os.path.join(savedir, 'yolo-labels')) # 粘贴补丁后的预测结果
    

    labdir_gro_tru = labdir + '/' + 'yolo-labels'

    n_gro_tru_labels = len(fnmatch.filter(os.listdir(labdir_gro_tru), '*.txt'))
    n_pre_labels = len(fnmatch.filter(os.listdir(pre_label_dir), '*.txt'))

    print("total ground truth labels : ", n_gro_tru_labels)
    print("total predicted labels : ", n_pre_labels)
    length_gro_tru,_ = txt_len_read(labdir_gro_tru)
    print("length of ground truth labels : ", length_gro_tru)
    length_pre_label,_ = txt_len_read(pre_label_dir)
    print("length of predicted labels : ", length_pre_label)

    conf_thresh = 0.4  # threshold = 0.4  # 0.4和0.6差别较大
    precision, recall = utils_self.eval_list(
        pre_label_dir, labdir_gro_tru, conf_thresh)
    ASR = (length_gro_tru - length_pre_label) / length_gro_tru
    print("final precision : ", precision)
    print("final recall : ", recall)
    print("final ASR : ", ASR)
    t1 = time.time()
    t11 = (t1-t0)/60
    print('recall & precision Total running time: {:.4f} minutes'.format(t11))

