
from PIL import Image
import PIL.Image
import PIL.ImageOps
# from utils import *
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)

import math
import torch
from PIL import Image, ImageDraw, ImageFont

VisDrone_NAMES = "data/VisDrone.names"


def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return img

def txt_len_read(txtfile_list):
    # for instances calculate
    len_txt = 0
    len_ins_account = []
    for txtfile_label in os.listdir(txtfile_list):  # 得到所有文件名
        txtfile = os.path.abspath(os.path.join(txtfile_list, txtfile_label)) 
        if os.path.getsize(txtfile):
            myfile = open(txtfile)
            single_len = len(myfile.readlines())
            len_txt += single_len
            len_ins_account.append(single_len)
            
            # if (single_len ==6 or single_len == 7):
            # if (single_len > 5):
                
            #     print("label ID : ", txtfile)   #   打印长度大于5 labels文件。
            # 可以找到特定长度的文件，便于手动操作
            
    return len_txt, len_ins_account
    # return len_txt, len_ins_account


from torchvision import transforms
from PIL import Image
import torch

# patchfile_0 = "/mnt/sunjialiang/kitti_car/KITTI_DataSet_mix/easy_exp_1200_3/1880_patch.png"
# patchfile_1= "/mnt/sunjialiang/kitti_car/KITTI_DataSet_mix/easy_exp_1200_3/1810_patch.png"

def patch_MSE_calsulator(patchfile_0, patchfile_1):
# def patch_MSE_calsulator(adv_patch_0, adv_patch_1):
    
    # 计算两个补丁间的MSE
    patch_img_0 = Image.open(patchfile_0).convert('RGB')
    patch_img_1 = Image.open(patchfile_1).convert('RGB')

    tf = transforms.ToTensor()
    adv_patch_0 = tf(patch_img_0)  # 补丁转换到3维tensor
    adv_patch_1 = tf(patch_img_1)

    patch_diff = adv_patch_1 - adv_patch_0
    mse_F = torch.nn.MSELoss()
    patch_mse = mse_F(adv_patch_0, adv_patch_1)  # 计算两个补丁的MSE

    return patch_mse

def hist_draw(data_list, save_dir):
    '''绘制柱状图，根据每张图片统计的instances'''
    plt.bar(range(len(data_list)), data_list)
    plt.xlabel("number of instances")
    plt.ylabel("number of images")
    plt.savefig(save_dir)
    plt.show()


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # print("x[...,0] = ", x[...,0])
    #   这里因为首先对x进行了clone，所以应该是可以这么操作的，然后分别提取第1，2，3，4列
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def do_detect_nms(model, img, conf_thresh, nms_thresh, use_cuda=1):
    
    model.eval()
    
    if isinstance(img, Image.Image):
        """
        isinstance用来判断一个变量的类型，
        例如上面的条件是判断img是否是图片
        并对图片格式进行转换
        adv_patch = Image.open('data/horse.jpg').convert('RGB')  # 将这张图片作为patch进行测试
        adv_patch = adv_patch.resize((patch_size, patch_size))
        transform = transforms.ToTensor()
        adv_patch = transform(adv_patch).cuda()
        """
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        # trans to tensor，并且这个时候还对通道进行了分离，
        # print("img size is :", img.size())  # 608*608*3
        img = img.view(height, width, 3).transpose(
            0, 1).transpose(0, 2).contiguous()  # 对图片进行升维并变换，先[0, 1]，后[0,2]
        # contiguous：把tensor变成内存中连续分布的形式
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)  # 归一化
    elif type(img) == np.ndarray:  # cv2 image
        img = torch.from_numpy(img.transpose(
            2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == torch.Tensor:
        # 20240407 修改
        width, height = img.shape[-2:]

        #   2023-06-13补充
        if img.dim() == 3:
            img = img.unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()

    with torch.no_grad():
        outputs = model(img)
    #   NMS
    outputs_nms = non_max_suppression(outputs,conf_thres=conf_thresh, iou_thres=nms_thresh)
    '''
    outputs_nms: [n x 6]，n为instances数量。
    nms的参数，只指定conf_thresh和iou_thresh就可，其它的默认
    classes: 指定输出类别
    max_det: 最大检测数量
    '''
    #   这里进行归一化处理，便于后面的label保存和绘图
    outputs_data = outputs_nms[0]
    #   outputs_data: [x1,y1,x2,y2, conf, cls_id]
    #   1. 归一化
    #   2. 变成[x,y,w,h]
    # all_boxes = []
    # for box in outputs_data:
    #     box[0] = box[0] =
    #   [x1,y1,x2,y2] --> [x,y,w,h]
    outputs_xywh = xyxy2xywh(outputs_data)
    
    #   normalization
    outputs_norm = outputs_xywh.clone() if isinstance(outputs_xywh, torch.Tensor) else np.copy(outputs_xywh)
    
    outputs_norm[..., 0] = outputs_norm[...,0] / width
    outputs_norm[..., 1] = outputs_norm[...,1] / height
    outputs_norm[..., 2] = outputs_norm[...,2] / width
    outputs_norm[..., 3] = outputs_norm[...,3] / height
    
    
    return outputs_norm


def plot_boxes(img, boxes, savename=None, class_names=None):
    #   此时的boxes格式为[]
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [
                               0, 1, 0], [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='data/simhei.ttf',
                              size=np.floor(2e-2 * width + 0.5).astype('int32'))

    for box in boxes:
        # box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width  # box[1,2,3,4]中的值是归一化的，这里再还原
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height
        # x1 = box[0]
        # y1 = box[1]
        # x2 = box[2]
        # y2 = box[3]

        rgb = (255, 0, 0)
        if len(box) >= 5 and class_names:  #
            cls_conf = box[4]
            cls_id = int(box[5])
            # 同时输出类别id，str和置信度
            # print('[%i]%s: %f' % (cls_id, class_names[cls_id], cls_conf))
            # print('[%i]%s: %f' % (cls_id, class_names[cls_id], box[4]*box[5]))

            classes = len(class_names)
            offset = cls_id * 123457 % classes  # offset根据不同的类别设置颜色
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            score = box[4]  # 是使用那个score？
            # score = box[4] * box[5]    #   yolov3采用的就是两个乘积
            label = '{}{:.2f}'.format(class_names[cls_id], score)

            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            text_origin = np.array([x1, y1])

            '''
            # font = ImageFont.truetype(None, size=40, encoding="unic")
            draw.text((x1+40, y1), str(round(box[5].item(), 3)), fill=(0,0,205))  # 只是det_confs，并不是cls_confs
            #   round是显示小数点问题，输出的到底是cls_conf还是cls_conf*obj_conf?
            draw.text((x1, y1), class_names[cls_id], fill=(255,0,0))  # 更改字体颜色
            # draw.text((x1, y1), class_names[cls_id], fill=rgb)  # 标注文字
            '''
            draw.rectangle([tuple(text_origin), tuple(
                text_origin+label_size)], fill=(255, 0, 0))
            draw.text(text_origin, str(label, 'UTF-8'),
                      fill=(0, 0, 0), font=font)
        # draw.rectangle([x1, y1, x2, y2], outline=rgb)
        # 作者考虑到了给labels数据标注边框的情况，此时只标注边框，赞
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)
        # 想更改线条的粗细
    if savename:
        # print("save plot results to %s" % savename)  # 输出保存地址
        img.save(savename)
    return img


import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import torchvision


import struct  # get_image_size
import imghdr  # get_image_size
ANCHOR_PATH = "data/yolov3_anchors.txt"


def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)


def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch  # 可结合图形理解，此为交面积
    uarea = area1 + area2 - carea
    return carea/uarea  # 返回两个盒子的交并比


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes  # 如果没有一个框

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]  # 第5个元素是置信度

    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0  # 如果交并比大于阈值，则赋值为0
    return out_boxes  # 返回的仍是boxes信息，筛选后


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

# def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, img_size, only_objectness=0, validation=False):
    # anchor_step = len(anchors)//num_anchors
    '''多维度输入'''
    


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [
                               0, 1, 0], [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(round((box[0] - box[2]/2.0) * width))
        y1 = int(round((box[1] - box[3]/2.0) * height))
        x2 = int(round((box[0] + box[2]/2.0) * width))
        y2 = int(round((box[1] + box[3]/2.0) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(
                img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img



def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        # to avoid single truth problem
        # truths = truths.reshape(truths.size//6, 6)
        truths = truths.reshape(truths.size//5, 5)  # suv_patch_real_suv.py
        return truths
    else:
        return np.array([])

def read_truths_pre_7(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        # to avoid single truth problem
        # truths = truths.reshape(truths.size//6, 6)
        truths = truths.reshape(truths.size//7, 7)  # suv_patch_real_suv.py
        return truths
    else:
        return np.array([])

def read_truths_pre_6(lab_path):
    #   对于U版的yolov3,v5，仅会输出六列数据[x,y,w,h,obj_conf,id]
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        # to avoid single truth problem
        truths = truths.reshape(truths.size//6, 6)
        return truths
    else:
        return np.array([])
    
def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    # remove truths of which the width is smaller then the min_box_scale
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1],
                           truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def load_class_names(namesfile):  # 导入数据集的类名文件 # 用于加载coco.names文件夹
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(
        0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img


def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]

    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    """
    根据模型和图片、score阈值、nms阈值实现边界框的预测
    """
    model.eval()

    if isinstance(img, Image.Image):
        """
        isinstance用来判断一个变量的类型，
        例如上面的条件是判断img是否是图片
        并对图片格式进行转换
        adv_patch = Image.open('data/horse.jpg').convert('RGB')  # 将这张图片作为patch进行测试
        adv_patch = adv_patch.resize((patch_size, patch_size))
        transform = transforms.ToTensor()
        adv_patch = transform(adv_patch).cuda()
        """
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        # trans to tensor，并且这个时候还对通道进行了分离，
        # print("img size is :", img.size())  # 608*608*3
        img = img.view(height, width, 3).transpose(
            0, 1).transpose(0, 2).contiguous()  # 对图片进行升维并变换，先[0, 1]，后[0,2]
        # contiguous：把tensor变成内存中连续分布的形式
        #   H x W x c -->c x H x W
        #   2023-06-20，这里，为什么不直接view成（3，H，W），应该是有原因的
        #   view和transpose虽然可以实现维度一样，但是具体操作是不同的
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)  # 归一化
    elif type(img) == np.ndarray:  # cv2 image
        img = torch.from_numpy(img.transpose(
            2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()

    # Simen: dit doet een forward, vervangen voor duidelijkheid
    with torch.no_grad():
        outputs = model.forward(img)  # prediction

    outputs_extract = outputs[1]
    for i in range(3):
        #   这个for处理后，将数据格式从[batch,3,80,80,15]变换成[batch, 3*15, 80, 80]
        #   得到了符合格式的数据
        bs, na, ny, nx, no = outputs_extract[i].shape
        outputs_extract[i] = outputs_extract[i].permute(0,1,4,2,3).contiguous().view(bs, na*no, ny,nx)
                    
                        
    # get_region_boxes的输入和do_detect()的输入，变成了

    anchors = get_anchors(ANCHOR_PATH)
    num_anchors = len(anchors)
    class_names = load_class_names(VisDrone_NAMES)
    num_classes = len(class_names)
    # print("length of class_names : ", len(class_names))
    boxes_list = []
    for i in range(len(anchors)):
        # print("anchor_i : ", anchors[i])
        boxes = get_region_boxes(outputs_extract[i], conf_thresh, num_classes, anchors[i], num_anchors, (width, height))[0]
        #   这里的outputs_extract[i]数据格式为
        #   进行调整，得到的是非归一化的boxes
        boxes_list.append(boxes)

    all_boxes = []
    for box in boxes_list:
        for i in range(len(box)):
            box[i][0] = box[i][0] / width
            box[i][2] = box[i][2] / width
            box[i][1] = box[i][1] / height
            box[i][3] = box[i][3] / height
            all_boxes.append(box[i])
    boxes = nms(all_boxes, nms_thresh)  # 在这个函数内进一步对boxes进行了筛选
    #   nms返回的是个list

    return boxes
    '''
    outputs_extract = outputs[1]
    for i in range(3):
        #   这个for处理后，将数据格式从[batch,3,80,80,15]变换成[batch, 3*15, 80, 80]
        #   得到了符合格式的数据
        bs, na, ny, nx, no = outputs_extract[i].shape
        outputs_extract[i] = outputs_extract[i].permute(0,1,4,2,3).contiguous().view(bs, na*no, ny,nx)
    '''


def read_data_cfg(datacfg):  # 读取cfg文件
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)  # 深拷贝
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets


def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close()
    return count


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height


def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)

'''
def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.5, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates  # obj_conf第一次筛选

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = True  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output
'''

def eval_list(pre_patched_labels, gro_labels, conf_thresh):
    '''
    数据格式：
    pre_patched_labels: [x,y,w,h,obj_conf, id]
    gro_labels: [id, x, y, w, h]
    '''
    conf_thresh = conf_thresh  # 最终需要对这个值设置
    iou_thresh = 0.4  # AP跟iou阈值有关？
    total = 0.0
    proposals = 0.0
    correct = 0.0
    lineId = 0
    avg_iou = 0.0

    for imgfile in os.listdir(pre_patched_labels):  # 预测标签
        if imgfile.endswith('.txt'):
            pre_labeld_dir = os.path.abspath(os.path.join(pre_patched_labels, imgfile))
            lab_path = os.path.abspath(os.path.join(
                gro_labels, imgfile))
            lineId = lineId + 1  # 共有多少张图片

            truths = read_truths(lab_path)  # 真实标签  # 加载标签
            boxes_cls = read_truths_pre_6(pre_labeld_dir)

            total = total + truths.shape[0]  # ground_truth labels的个数

            for i in range(len(boxes_cls)):
                if (boxes_cls[i][4]) > conf_thresh:  
                    #   第五个元素为检测obj概率——U版只有六列，不包含类别概率
                    proposals = proposals+1  # 预测出来的检测框数量
                    # proposal即为预测为正值的情况，为TP+FP
            for i in range(truths.shape[0]):
                box_gt = [truths[i][1], truths[i][2],
                          truths[i][3], truths[i][4], 1.0]
                #   之前的ground truth标签虽然是[id, x,y,w,h]，然后这里重新定义了一个变量box_gt
                #   对顺序进行了调整
                best_iou = 0
                for j in range(len(boxes_cls)):
                    iou = bbox_iou(
                        box_gt, boxes_cls[j], x1y1x2y2=False)  # 计算IoU
                    best_iou = max(iou, best_iou)  # 最大的IoU
                if best_iou > iou_thresh:
                    avg_iou += best_iou
                    correct = correct+1  # correct为预测矩阵中的TP，为recall和precision中的分子

    precision = correct/(proposals + 1e-8)  # 对所有图片的结果 TP/(TP+FP)
    recall = correct/(total + 1e-8)  # TP/(TP+FN)
    # 简单理解为 recall越低越好
    fscore = 2.0*precision*recall/(precision+recall + 1e-6)
    # print("results in recall.py :")
    # print("%d IOU: %f, Recal: %f, Precision: %f, Fscore: %f\n" %
    #       (lineId-1, avg_iou/(correct + 1e-6), recall, precision, fscore))
    # print("total images = ", n_images)
    return precision, recall