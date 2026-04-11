"""
数据处理和补丁添加
Loss计算  # TGJ
"""
import matplotlib.pyplot as plt
import fnmatch
import math
import os
import sys
import time
import gc
import numpy as np
from numpy.core.shape_base import block
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import utils_self
from median_pool import MedianPool2d

# import edge_extractor_oneStage

ANCHOR_PATH = "data/yolov3_anchors.txt"
VisDrone_NAMES = "data/VisDrone.names"
IMG_SIZE = (640, 640)

SCALE = 2.
SCALE_FACTOR = 2.


def read_image(path):
    """
    读取一个已经训练好的补丁
    Read an input image to be used as a patch

    :param path: Path to the image to be read.
    :return: Returns the transformed patch as a pytorch Tensor.
    """
    patch_img = Image.open(path).convert('RGB')
    tf = transforms.ToTensor()

    adv_patch_cpu = tf(patch_img)
    return adv_patch_cpu


# def get_boxes_loss(output, anchors, num_anchors):
# torch.nn.MSEloss 有除以N


def bbox_reg(bbox_extractor):
    attack_bbox = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6]).cuda()
    # bbox_mse = []
    # for box in bbox_extractor:
    #     # print(len(box))
    #     box_mse = torch.nn.MSELoss()(attack_bbox, box) * len(box)
    #     bbox_mse.append(box_mse)
    # return bbox_mse
    # box_mse = torch.nn.MSELoss()(attack_bbox, bbox_extractor) * len(bbox_extractor)
    box_mse = torch.nn.MSELoss()(attack_bbox, bbox_extractor)
    return box_mse


class ClassLoss_CE(nn.Module):
    '''
    CE-loss似乎只能开展目标攻击？？
    '''

    def __init__(self, cls_id, num_cls):
        super(ClassLoss_CE, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls

    '''
    #   使用所有检测框构建CE-loss还不行
    # def forward(self, YOLOoutputs, target_ID):
    #    target_ID：待攻击的目标ID
    #     # target_ID = target_ID.cuda()
    #     output_cls = YOLOoutputs[:,:,5:]
    #     #   batch x 25200 x 10
    #     cls_confs = torch.sigmoid(output_cls)   #   sigmoid激活
    #     #   这里要不要进行筛选？好像没法筛选。。。
    #     batch_size = cls_confs.size(0)
    #     num_boxes = cls_confs.size(1)
    #     CE_loss = torch.nn.CrossEntropyLoss()

    #     target_label = torch.tensor([target_ID]).cuda()
    #     target_label = target_label.repeat(num_boxes)
    #     batch_loss = torch.zeros(batch_size)

    #     for i in range(batch_size):
    #         size_temp = cls_confs[i,:,:]
    #         ce_loss_per_image = CE_loss(cls_confs[i,:,:], target_label)
    #         batch_loss[i] = ce_loss_per_image

    #     cls_batch_loss = torch.mean(batch_loss)
    #     return cls_batch_loss
    '''

    def forward(self, YOLOoutputs, target_ID, max_labels):
        '''
        1. target_ID：待攻击的目标ID
        2. 这里的max_labels和max_len一致，用于筛选检测框
        '''
        # target_ID = target_ID.cuda()
        output_cls = YOLOoutputs[:, :, 5:].cuda()
        #   batch x 25200 x 10
        cls_confs = torch.sigmoid(output_cls)  # sigmoid激活
        #   这里要不要进行筛选？好像没法筛选。。。
        batch_size = cls_confs.size(0)
        # num_boxes = cls_confs.size(1)
        CE_loss = torch.nn.CrossEntropyLoss()
        #   对检测框进行筛选，原则：首先提取提取id=3的概率；然后进行sort；取前60index

        target_label = torch.tensor([target_ID]).cuda()
        target_label = target_label.repeat(max_labels)
        batch_loss = torch.zeros(batch_size)

        conf_for_class = cls_confs[:, :, self.cls_id]  # batch x 25200
        #   现在要把conf_for_class.shape=batchx25200变成batch * max_labels，并且留下的是indices
        _, indices = torch.topk(conf_for_class, k=max_labels, largest=True, sorted=True)
        #   仅需要得到indices即可，indices.shape = batch * max_labels

        conf_selected = torch.zeros(batch_size, max_labels, 10).cuda()  # 10是类别
        for i in range(batch_size):
            conf_selected[i, :, :] = cls_confs[i, indices[i, :], :]  # 期望size=max_labels x 10
            # temp = conf_selected[i,:,:]
            ce_loss_per_image = CE_loss(conf_selected[i, :, :], target_label)
            batch_loss[i] = ce_loss_per_image

        cls_batch_loss = torch.mean(batch_loss)
        #   取所有检测框的时候，还有可能是因为取了mean，取max会不会不同？？
        # cls_batch_loss,_ = torch.max(batch_loss,dim=0)
        return cls_batch_loss


class ClassLoss_MT(nn.Module):
    '''
    使用max-target loss，似乎可以开展目标、非目标攻击
    '''

    def __init__(self, cls_id, num_cls):
        super(ClassLoss_MT, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls

    def forward(self, YOLOoutputs, target_ID, max_labels):
        '''
        1. target_ID：待攻击的目标ID
        2. 这里的max_labels和max_len一致，用于筛选检测框
        '''
        # target_ID = target_ID.cuda()
        output_cls = YOLOoutputs[:, :, 5:].cuda()
        #   batch x 25200 x 10
        cls_confs = torch.sigmoid(output_cls)  # sigmoid激活
        #   这里要不要进行筛选？好像没法筛选。。。
        batch_size = cls_confs.size(0)
        # num_boxes = cls_confs.size(1)
        #   对检测框进行筛选，原则：首先提取提取id=3的概率；然后进行sort；取前60index

        if target_ID:
            '''目标攻击
               此时还得首先对检测框进行筛选
               因为如果不进行筛选，攻击的就不只是car类别
            '''
            # True
            # batch_loss = torch.zeros(batch_size)
            conf_for_class = cls_confs[:, :, self.cls_id]
            _, indices = torch.topk(conf_for_class, k=max_labels, largest=True, sorted=True)
            #   仅需要得到indices即可，indices.shape = batch * max_labels

            conf_selected = torch.zeros(batch_size, max_labels, 10).cuda()  # 10是类别
            for i in range(batch_size):
                conf_selected[i, :, :] = cls_confs[i, indices[i, :], :]  # 期望size=max_labels x 10

            #   在根据id进行筛选后，再对剩下的boxes构造loss
            conf_for_car = conf_selected[:, :, self.cls_id]
            conf_for_target_attack = conf_selected[:, :, target_ID]
            differ_conf = conf_for_car - conf_for_target_attack

            batch_loss_mean = torch.mean(differ_conf, dim=1)

            cls_batch_loss_sum = torch.sum(batch_loss_mean)

        else:
            '''非目标攻击'''
            conf_for_class = cls_confs[:, :, self.cls_id]  # batch x 25200
            #   此时用sum/mean还是max好？因为是要攻击所有类别，是否可以用sum？？用mean可能这个值很小
            # batch_loss = torch.sum(conf_for_class,1)
            batch_loss_mean = torch.mean(conf_for_class, 1)

            # cls_batch_loss = torch.mean(batch_loss)
            cls_batch_loss_sum = torch.sum(batch_loss_mean)
        return cls_batch_loss_sum


# class MaxProbExtractor(nn.Module):
#     """此部分计算YOLO的输出置信度。
#     MaxProbExtractor: extracts max class probability for class from YOLO output.
#     提取的是输出向量中最大的概率类别，因为最后是要进行优化，使得分类类别不是person
#     Module providing the functionality necessary to extract the max class probability for one class from YOLO output.
#
#     """
#
#     def __init__(self, cls_id, num_cls, config):
#         """
#         reduction: 'max' | 'mean' | 'topk'
#         topk: 仅在 reduction='topk' 时有效
#         """
#         super(MaxProbExtractor, self).__init__()
#         self.cls_id = cls_id  # 传入参数
#         self.num_cls = num_cls  # 传入参数
#         self.config = config
#
#     def forward(self, YOLOoutputs):
#
#         '''
#         现在这里的YOLOoutputs格式是[batch, (80**2+40**2+20**2)*3, 15]
#
#         '''
#         # 这部分代码实现的是使用sigmoid激活后loss
#         output_objectness = YOLOoutputs[:,:,4]
#         #   batch x 25200
#         output_cls_conf = YOLOoutputs[:,:,5:]
#         #   batch x 25200 x 10
#         max_obj_conf, _ = torch.max(output_objectness, dim=1)  #   这用来确定权重参数
#         normal_confs = torch.sigmoid(output_cls_conf)  # 对第二维进行softmax操作, 对类别概率进行softmax激活
#         confs_for_class = normal_confs[:, :, self.cls_id]  #
#         #   batch x 25200
#         max_cls_conf, _ = torch.max(confs_for_class, dim=1)  #   这用来确定权重参数
#
#         confs_if_object = self.config.loss_target(
#             output_objectness, confs_for_class)  # 最终使用的loss在这里确定[batch, 22743]
#         max_conf, max_conf_idx = torch.max(
#             confs_if_object, dim=1)  # 这个地方应不应该只取最大值还值得商榷
#
#
#         return max_conf

class MaxProbExtractor(nn.Module):
    def __init__(self, cls_id, num_cls, config, reduction='max', topk=5):
        """
        reduction: 'max' | 'mean' | 'topk'
        topk: 仅在 reduction='topk' 时有效
        """
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        self.reduction = reduction
        self.topk = topk

    def forward(self, YOLOoutputs):
        output_objectness = YOLOoutputs[:, :, 4]  # [B, N]
        output_cls_conf = YOLOoutputs[:, :, 5:]  # [B, N, num_cls]
        normal_confs = torch.sigmoid(output_cls_conf)  # [B, N, num_cls]
        confs_for_class = normal_confs[:, :, self.cls_id]  # [B, N]

        # Combine objectness + class confidence
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)  # [B, N]

        # ✅ 注意：这里的 if-else 缩进必须在 forward 函数内部！
        if self.reduction == 'max':
            final_conf, _ = torch.max(confs_if_object, dim=1)  # [B]
        elif self.reduction == 'mean':
            final_conf = torch.mean(confs_if_object, dim=1)  # [B]
        elif self.reduction == 'topk':
            topk_vals, _ = torch.topk(confs_if_object, self.topk, dim=1)  # [B, topk]
            final_conf = torch.mean(topk_vals, dim=1)  # [B]
        else:
            raise ValueError(f"Unknown reduction mode: {self.reduction}")

        return final_conf  # shape: [B]


# class NPSCalculator(nn.Module):
#     """NPSCalculator: calculates the non-printability score of a patch.
#
#     Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
#
#     """
#
#     def __init__(self, printability_file, patch_side):
#         super(NPSCalculator, self).__init__()
#         self.printability_array = nn.Parameter(self.get_printability_array(
#             printability_file, patch_side), requires_grad=False)
#         # 获得可打印分数数组，可参考"adv_patch.py"理解
#         #  nn.Parameter函数？将一个不可训练的类型转换称可训练的类型
#
#     def forward(self, adv_patch):
#         # calculate euclidian distance between colors in patch and colors in printability_array
#         # square root of sum of squared difference
#         color_dist = (adv_patch - self.printability_array + 0.000001)
#         color_dist = color_dist ** 2
#         color_dist = torch.sum(color_dist, 1) + 0.000001
#         color_dist = torch.sqrt(color_dist)
#         # only work with the min distance
#         # test: change prod for min (find distance to closest color)
#         color_dist_prod = torch.min(color_dist, 0)[0]
#         # calculate the nps by summing over all pixels
#         nps_score = torch.sum(color_dist_prod, 0)
#         nps_score = torch.sum(nps_score, 0)
#         return nps_score / torch.numel(adv_patch)
#
#     def get_printability_array(self, printability_file, side):
#         printability_list = []
#
#         # read in printability triplets and put them in a list
#         with open(printability_file) as f:
#             for line in f:
#                 printability_list.append(line.split(","))
#
#         printability_array = []
#         for printability_triplet in printability_list:
#             printability_imgs = []
#             red, green, blue = printability_triplet
#             printability_imgs.append(np.full((side, side), red))
#             printability_imgs.append(np.full((side, side), green))
#             printability_imgs.append(np.full((side, side), blue))
#             printability_array.append(printability_imgs)
#
#         printability_array = np.asarray(printability_array)
#         #   将list转换成array，[30,3,224,224]
#         printability_array = np.float32(printability_array)
#         pa = torch.from_numpy(printability_array)
#         return printability_array
class NPSCalculator(nn.Module):
    """NPSCalculator: calculates the non-printability score of a patch.
    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(
            self.get_printability_array(printability_file, patch_side),
            requires_grad=False
        )

    def forward(self, adv_patch):
        """
        Calculate Non-Printability Score (NPS)
        Compatible with rectangular patches (e.g., 500×600)
        """
        pa = self.printability_array

        # --- ✅ 自动匹配矩形尺寸 ---
        if pa.shape[-2:] != adv_patch.shape[-2:]:
            # printability_array: [num_colors, 3, H, W]
            # 逐个 resize，保持一致的通道顺序
            resized_list = []
            for p in pa:  # p shape: [3, H, W]
                p_resized = F.interpolate(
                    p.unsqueeze(0),  # [1, 3, H, W]
                    size=(adv_patch.shape[-2], adv_patch.shape[-1]),  # H, W 顺序确保一致
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                resized_list.append(p_resized)
            pa = torch.stack(resized_list, dim=0).to(adv_patch.device)
        # ---------------------------------------------

        # --- ✅ 保持原始逻辑 ---
        color_dist = (adv_patch - pa + 1e-6)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 1e-6
        color_dist = torch.sqrt(color_dist)
        color_dist_prod = torch.min(color_dist, 0)[0]
        nps_score = torch.sum(color_dist_prod)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(
            adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(
            adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class PatchTransformer_creation(nn.Module):

    def __init__(self):
        super(PatchTransformer_creation, self).__init__()

        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -180 / 180 * math.pi  # self.minangle = -20 / 180 * math.pi
        self.maxangle = 180 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def lab_transform(self, lab_batch_origin):
        '''
        对原始的lab_batch进行变换，只保留标签数据中size最大的目标
        input: lab_batch_origin
        output: lab_batch_select
        '''

        #   以下的标签是nx5，是经过0.4筛选得到
        lab_batch_select = torch.cuda.FloatTensor(
            lab_batch_origin.size(0), 1, 5).fill_(0)
        # shape = [batch, 1, 5]
        area_cal = lab_batch_origin[:, :, 3] * lab_batch_origin[:, :, 4]
        #   测试数据需要根据是5列还是6列对计算进行区分
        max_value, max_index = torch.max(area_cal, 1)  # 返回每张图片label面积的最大值及索引
        #   如果max_value = 1，则场景中没目标。
        min_value, min_index = torch.min(area_cal, 1)  # 找到面积最小的索引

        for i in range(lab_batch_origin.size(0)):
            if max_value[i] > 0.99:
                lab_batch_select[i, :, :] = torch.tensor(
                    [0.25, 0.25, 0.25, 0.25, 0.25]).cuda()
            else:
                temp_max = lab_batch_origin[i, max_index[i], :]
                temp_min = lab_batch_origin[i, min_index[i], :]
                #   取最大最小的平均
                lab_batch_select[i, :, :] = (temp_max + temp_min) / 2.
                #   直接取最大的作为参考
                # lab_batch_select[i, :, :] = lab_batch_origin[i, max_index[i], :]

        # 以下标签基于nx7得到，即经过0.01筛选得到的数据。
        '''
        lab_batch_select = torch.cuda.FloatTensor(
            lab_batch_origin.size(0), 1, 7).fill_(0)
        # shape = [batch, 1, 5]
        area_cal = lab_batch_origin[:, :, 2] * lab_batch_origin[:, :, 3]
        max_value, max_index = torch.max(area_cal, 1)  # 返回每张图片label面积的最大值及索引
        #   如果max_value = 1，则场景中没目标。#    因为之前的是填充为了1
        min_value, min_index = torch.min(area_cal, 1)  # 找到面积最小的索引

        ##################################################################################
        #   下面是对标签进行筛选和判断，如果场景中无目标，
        ##################################################################################
        if len(lab_batch_origin[0]) == 1:
            lab_batch_select[0, :, :] = torch.tensor(
                    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]).cuda()
        else:
            for i in range(lab_batch_origin.size(0)):
                if max_value[i] > 0.99:
                    lab_batch_select[i, :, :] = torch.tensor(
                        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]).cuda()
                else:
                    temp_max = lab_batch_origin[i, max_index[i], :]
                    temp_min = lab_batch_origin[i, min_index[i], :]
                    lab_batch_select[i, :, :] = (temp_max + temp_min) / 2.
                    # lab_batch_select[i, :, :] = lab_batch_origin[i, max_index[i], :]
        '''
        return lab_batch_select
        # return lab_batch_select

    def forward(
            self,
            adv_patch,
            lab_batch,
            img_size,
            do_rotate=True,
            rand_loc=False):
        """
        inputs：3-channels adv_patch, 3维标签（已扩维），图片大小
        lab_batch = [batch, max_lab, 5]

        outputs：[batch, max_lab, 3, 608, 608]
        patch_center: [batch, 2]，每张图片给出一个坐标"""

        adv_patch = self.medianpooler(
            adv_patch.unsqueeze(0))  # 返回[1,3,224,224], 补丁size()是固定的

        pad = (img_size - adv_patch.size(-1)) / 2  # (640-224) / 2 = 208
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  5维，[1,1,3,224,224]
        adv_batch = adv_patch.expand(
            lab_batch.size(0), 1, -1, -1, -1)

        batch_size = torch.Size((lab_batch.size(0), 1))  # [batch, max_lab]
        #   调整成[batch, 1]

        contrast = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_contrast, self.max_contrast)
        #   [batch, max_lab]
        # [batch, max_lab, 1, 1, 1]
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #   这里是在(-1)维扩维

        contrast = contrast.expand(-1, -1, adv_batch.size(-3),
                                   adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()  # [batch, 1, 3, 224, 224]
        #   max_lab = 1

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3),
                                       adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()  # [1,2,3,300,300]
        # RGB图像
        # noise = torch.cuda.FloatTensor(
        #     adv_batch.size()).uniform_(-1, 1) * self.noise_factor  # [1,2,3,300,300]
        # 红外图像
        noise = torch.cuda.FloatTensor(B, L, 1, H, W).uniform_(-1, 1) * self.noise_factor
        noise = noise.expand(-1, -1, 3, -1, -1)  # 复制成3通道
        # 更改代码，添加随机均匀分布噪声
        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise # real_test时固定

        # [1,2,3,224,224]  #  [0.000001, 0.99999]
        adv_batch = torch.clamp(adv_batch, 0.0, 1.)
        msk_batch = torch.ones_like(adv_batch).cuda()
        # print('mask_batch siz : ', msk_batch.size())

        mypad = nn.ConstantPad2d(
            (int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)

        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)  # [batch, max_lab, 3, 608, 608]
        # anglesize = (lab_batch.size(0) * lab_batch.size(1))  #
        anglesize = lab_batch.size(0)
        # anglesize_ = lab_batch.size()[0]
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(
                self.minangle, self.maxangle)  # 2
            # 将tensor用从均匀分布中的值填充
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates，调整大小并缩放
        current_patch_size = adv_patch.size(-1)  # 224
        # lab_batch_scaled = torch.cuda.FloatTensor(
        #     lab_batch.size()).fill_(0)  # [batch, max_lab, 5]   #   全 0 tensor
        #   这里学习torch.cuda.FloatTensor()语法
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size(
            0), 1, 5).fill_(0)  # [batch, max_lab, 5]   #   全0tensor

        #   如果场景中没目标，即lab_batch_scaled = [1,1,1,1,1]
        #   以下要对lab_batch进行筛选，只保留最大size的lab
        lab_batch_selected = self.lab_transform(lab_batch)  # 调用函数对其进行变换

        lab_batch_scaled[:, :, 0] = lab_batch_selected[:,
                                    :, 0] * img_size  # [batch,1,5]
        lab_batch_scaled[:, :, 1] = lab_batch_selected[:,
                                    :, 1] * img_size  # 第2、3（1，2）列没用上

        lab_batch_scaled[:, :, 2] = lab_batch_selected[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch_selected[:, :, 3] * img_size

        # ---------------------------------------------------------------#
        #   定义
        # ---------------------------------------------------------------#
        pre_scale = SCALE_FACTOR  # 缩放因子
        #   获得缩放因子
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 2].mul(
            1 / pre_scale)) ** 2) + ((lab_batch_scaled[:, :, 3].mul(1 / pre_scale)) ** 2))  # [batch, 1]

        #####################################################################################
        #   隔断
        #####################################################################################

        #   这里的位置现在是随机
        target_x = torch.rand(lab_batch_scaled[:, :, 1].size()).cuda()
        target_x = target_x.view(np.prod(batch_size))
        #   这里的rand(size=)，对于x,y而言，size是一样的
        # target_x = torch.rand_like(lab_batch_scaled[:,:,1])
        #   位置随机的话，超出边界如何处理？
        #   认为设置边界，当超出边界时，直接取边界，边界设置为[0.2, 0.8]
        target_y = torch.rand(lab_batch_scaled[:, :, 2].size()).cuda()

        target_y = target_y.view(np.prod(batch_size))

        target_x = torch.max(target_x, torch.tensor(0.2).cuda())
        #   torch.max/min除了获得tensor中的最大、最小值外，还可以对值进行比较，从而返回最大或最小值
        #   例如torch.max(a, b)返回相应元素的最大值，即谁大保留谁
        target_y = torch.min(target_y, torch.tensor(0.8).cuda())
        #   通过上面的操作可将中心坐标约束在[0.2,0.8]内

        #####################################################################################
        #   这里的target_x, target_y就是现在补丁的归一化坐标了吧
        #####################################################################################
        patch_center_x = (target_x * img_size).view(-1, 1)
        patch_center_y = (target_y * img_size).view(-1, 1)

        patch_center_axis = torch.cat([patch_center_x, patch_center_y], 1)

        scale = target_size / current_patch_size  # 最终的缩放因子
        scale = scale.view(anglesize)

        s = adv_batch.size()  # [batch,1,3,608,608]
        adv_batch = adv_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]
        msk_batch = msk_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]

        tx = (-target_x + 0.5) * 2  # 两个数  # 这个2和theta/affine_grid机制相关
        ty = (-target_y + 0.5) * 2  # 和仿射变换的坐标相关，左上角为原点(0.,0.)
        #   向右、向下为负，这里的系数2和affine()函数定义相关
        #   creation attack时，位置要随机，就和lab中的位置信息不再相关

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)

        # theta = torch.zeros(anglesize, 2, 3)  # simpler # theta默认是一个2x3的矩阵
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, adv_batch.shape)  # 第二个参数设置图片大小
        # 同时形状不变（第二个参数）
        # grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]
        msk_batch_t = msk_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]

        # [batch,max_lab,3,608,608] # 000001
        adv_batch_t = torch.clamp(adv_batch_t, 0.0, 1.)
        adv_patch_mask = adv_batch_t * msk_batch_t

        return adv_patch_mask, patch_center_axis


class PatchTransformer(nn.Module):
    """用于对补丁进行各种变换
    本类在测试、训练中使用方式不一样
    PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.
    根据labels中的数据对补丁进行缩放，并将其填充到图像上

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()

        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -180 / 180 * math.pi  # self.minangle = -20 / 180 * math.pi
        self.maxangle = 180 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def forward(
            self,
            adv_patch,
            lab_batch,
            img_size,
            do_rotate=True,
            rand_loc=False,
    ):

        adv_patch = self.medianpooler(
            adv_patch.unsqueeze(0))  # 返回[1,3,224,224], 补丁size()是固定的

        pad = (img_size - adv_patch.size(-1)) / 2  # (608-300) / 2 = 154
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  5维，[1,1,3,300,300]
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size(
            (lab_batch.size(0), lab_batch.size(1)))  # [batch, max_lab]

        contrast = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3),
                                   adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()  # [1,2,3,300,300]

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(
            self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3),
                                       adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()  # [1,2,3,300,300]
        noise = torch.cuda.FloatTensor(
            adv_batch.size()).uniform_(-1, 1) * self.noise_factor  # [1,2,3,300,300]

        # [1,2,3,300,300]  #  [0.000001, 0.99999]
        adv_batch = torch.clamp(adv_batch, 0.0, 1.)
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)

        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)  # [1,2,3,1], 最后一个维度扩维
        cls_mask = cls_mask.expand(-1, -1, -1,
                                   adv_batch.size(3))  # [1,2,3,300]
        cls_mask = cls_mask.unsqueeze(-1)

        cls_mask = cls_mask.expand(-1, -1, -1, -1,
                                   adv_batch.size(4))  # [1,2,3,300,300]
        msk_batch_test = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
        # print("size of msk_batch_test :", msk_batch_test.size())  # debug
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
        mypad = nn.ConstantPad2d(
            (int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)

        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)  # [1,2,3,608,608]
        anglesize = (lab_batch.size(0) * lab_batch.size(1))  # 2，旋转角度数量和预测框个数一致
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(
                self.minangle, self.maxangle)  # 2
            # 将tensor用从均匀分布中的值填充
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates，调整大小并缩放
        current_patch_size = adv_patch.size(-1)  # 300
        lab_batch_scaled = torch.cuda.FloatTensor(
            lab_batch.size()).fill_(0)  # [1,2,5]全0tensor
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size  # [1,2,5]
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size  # 第2、3列没用上

        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size

        # ---------------------------------------------------------------#
        #   定义
        # ---------------------------------------------------------------#
        pre_scale = SCALE  # 缩放因子

        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(
            1 / pre_scale)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(1 / pre_scale)) ** 2))  # [1,2]

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # [1, 2]
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))

        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # w
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # h

        if (rand_loc):  # 位置随机
            off_x = targetoff_x * \
                    (torch.cuda.FloatTensor(targetoff_x.size()
                                            ).uniform_(-0.2, 0.2))  # 后面的随机太大
            target_x = target_x + off_x  # 加了随机性

            off_y = targetoff_y * \
                    (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.2, 0.2))
            # 位置比较关键，想让其偏下，试试参数
            target_y = target_y + off_y

        scale = target_size / current_patch_size  # 最终的缩放因子
        scale = scale.view(anglesize)

        s = adv_batch.size()  # [1,2,3,608,608]
        adv_batch = adv_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]
        msk_batch = msk_batch.view(
            s[0] * s[1], s[2], s[3], s[4])  # [batch*max_lab,3,608,608]

        tx = (-target_x + 0.5) * 2  # 两个数  # 这个2和theta/affine_grid机制相关
        ty = (-target_y + 0.5) * 2

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        # theta = torch.zeros(anglesize, 2, 3)  # simpler # theta默认是一个2x3的矩阵
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, adv_batch.shape)  # 第二个参数设置图片大小
        # 同时形状不变（第二个参数）
        # grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]
        msk_batch_t = msk_batch_t.view(
            s[0], s[1], s[2], s[3], s[4])  # [batch,max_lab,3,608,608]

        # [batch,max_lab,3,608,608] # 000001
        adv_batch_t = torch.clamp(adv_batch_t, 0.0, 1.)
        temp_val = adv_batch_t * msk_batch_t

        return adv_batch_t * msk_batch_t


class MaskedPatchTransformer(nn.Module):
    def __init__(self, scale_factor=2.0):
        super(MaskedPatchTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = 0.0
        self.maxangle = 0.0
        self.medianpooler = MedianPool2d(7, same=True)
        self.scale_factor = scale_factor #默认为4，很适合异形补丁

    def forward(self, adv_patch, lab_batch, img_size, do_rotate=False, rand_loc=False, input_mask=None):
        B, L = lab_batch.shape[0], lab_batch.shape[1]
        current_patch_size = adv_patch.shape[-1]

        # 放大到整图尺寸（图像大小）
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))  # [1,3,H,W]
        adv_patch = F.interpolate(adv_patch, size=(img_size, img_size), mode='bilinear', align_corners=True)
        adv_patch = adv_patch.unsqueeze(0)  # [1,1,3,H,W]
        adv_batch = adv_patch.expand(B, L, -1, -1, -1)  # [B,L,3,H,W]

        # 随机亮度/对比度/噪声
        contrast = torch.cuda.FloatTensor(B, L, 1, 1, 1).uniform_(self.min_contrast, self.max_contrast)
        brightness = torch.cuda.FloatTensor(B, L, 1, 1, 1).uniform_(self.min_brightness, self.max_brightness)
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # adv_batch = torch.clamp(adv_batch * contrast + brightness + noise, 0.0, 1.0)
        adv_batch = torch.clamp(adv_batch * contrast + brightness , 0.0, 1.0)


        # mask 同样处理为 img_size 尺寸
        if input_mask is not None:
            # input_mask = input_mask.expand(1, 1, current_patch_size, current_patch_size)
            H, W = input_mask.shape[-2:]
            input_mask = input_mask.expand(1, 1, H, W)
            input_mask = F.interpolate(input_mask, size=(img_size, img_size), mode='bilinear', align_corners=True)
            msk_batch = input_mask.expand(B, L, img_size, img_size).unsqueeze(2)
        else:
            msk_batch = torch.ones(B, L, 1, img_size, img_size).cuda()

        # 缩放
        scale = torch.sqrt((lab_batch[:, :, 3] * img_size / self.scale_factor) ** 2 +
                           (lab_batch[:, :, 4] * img_size / self.scale_factor) ** 2) / img_size
        scale = scale.view(B * L)

        # 中心位置
        target_x = lab_batch[:, :, 1].view(B * L)
        # 高度比例向上偏移
        bbox_height = lab_batch[:, :, 4].view(B * L)  # 每个检测框的高
        offset_ratio = 0.2  # 你可以调大或调小
        target_y = lab_batch[:, :, 2].view(B * L) - bbox_height * offset_ratio
        # target_y = lab_batch[:, :, 2].view(B * L)
        if rand_loc:
            offset_x = lab_batch[:, :, 3].view(B * L) * torch.cuda.FloatTensor(B * L).uniform_(-0.2, 0.2)
            offset_y = lab_batch[:, :, 4].view(B * L) * torch.cuda.FloatTensor(B * L).uniform_(-0.2, 0.2)
            target_x += offset_x
            target_y += offset_y

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        angle = torch.zeros(B * L).cuda()
        sin, cos = torch.sin(angle), torch.cos(angle)

        # 构造仿射矩阵
        theta = torch.zeros(B * L, 2, 3).cuda()
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        adv_batch = adv_batch.view(B * L, 3, img_size, img_size)
        msk_batch = msk_batch.view(B * L, 1, img_size, img_size)
        grid = F.affine_grid(theta, adv_batch.size(), align_corners=True)
        adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=True)
        msk_batch_t = F.grid_sample(msk_batch, grid, mode='nearest', align_corners=True)

        adv_batch_t = adv_batch_t.view(B, L, 3, img_size, img_size)
        msk_batch_t = msk_batch_t.view(B, L, 1, img_size, img_size)
        return adv_batch_t * msk_batch_t, msk_batch_t


class DifferentiablePolygonMask(nn.Module):
    def __init__(self, H, W, sharpness=100.0):
        super(DifferentiablePolygonMask, self).__init__()
        self.H = H
        self.W = W
        self.sharpness = sharpness

        # 创建网格坐标 [H, W, 2]
        y = torch.linspace(0, 1, H)
        x = torch.linspace(0, 1, W)
        yy, xx = torch.meshgrid(y, x)  # 默认就是 'ij' 索引方式
        self.register_buffer('grid', torch.stack([xx, yy], dim=-1))  # [H, W, 2]

    def forward(self, vertices):
        """
        vertices: Tensor of shape [N, 2], 归一化坐标（0~1）
        return: mask [1, 1, H, W] in [0,1] with soft edges
        """
        N = vertices.shape[0]
        grid = self.grid.view(-1, 1, 2)  # [H*W, 1, 2]
        v = vertices.unsqueeze(0)  # [1, N, 2]

        # 两两顶点构边：v0,v1; v1,v2; ... vN-1,v0
        v0 = v[:, :-1, :]  # [1, N-1, 2]
        v1 = v[:, 1:, :]
        v_last = v[:, 0:1, :]
        v0 = torch.cat([v0, v[:, -1:, :]], dim=1)  # v0 [1, N, 2]
        v1 = torch.cat([v1, v_last], dim=1)  # v1 [1, N, 2]

        # 叉乘判断点在边左侧
        e = v1 - v0  # 边向量 [1, N, 2]
        p = grid - v0  # 从顶点到点的向量 [H*W, N, 2]

        cross = e[..., 0] * p[..., 1] - e[..., 1] * p[..., 0]  # [H*W, N]
        inside_score = torch.sigmoid(self.sharpness * cross)  # [H*W, N]
        mask_flat = torch.prod(inside_score, dim=-1)  # [H*W]
        mask = mask_flat.view(self.H, self.W)
        return mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


class MaskedPatchApplier(nn.Module):
    def __init__(self):
        super(MaskedPatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch_vis, adv_batch_infra, transformed_masks):
        B = img_batch.shape[0]
        patched_batch = []

        for i in range(B):
            img = img_batch[i]  # [3, H, W]

            # 判断红外
            mean_r = torch.mean(img[0])
            mean_g = torch.mean(img[1])
            mean_b = torch.mean(img[2])
            mean_diffs = torch.tensor([mean_r - mean_g, mean_g - mean_b, mean_b - mean_r])
            is_infra = torch.max(torch.abs(mean_diffs)) < 0.01

            advs = torch.unbind(adv_batch_infra[i] if is_infra else adv_batch_vis[i], dim=0)  # [L, 3, H, W]
            masks = torch.unbind(transformed_masks[i], dim=0)  # [L, 1, H, W]

            for adv, mask in zip(advs, masks):
                # mask = (mask > 0.5).float().expand_as(adv)
                mask = mask.expand_as(adv)  # 将 [1, H, W] 扩展为 [3, H, W]
                img = img * (1 - mask) + adv * mask  # 只在 mask 区域贴补丁

                # 可视化mask
                # vis_mask = mask[0].detach().cpu().squeeze().numpy()  # [H,W]
                # plt.imshow(vis_mask, cmap='gray')
                # plt.colorbar()
                # plt.title(f"Mask at Epoch")
                # plt.savefig(f"epoch_mask.png")  # 保存图片到当前目录
                # plt.close()

            patched_batch.append(img)


        return torch.stack(patched_batch, dim=0)


# original
# class PatchApplier(nn.Module):
#     """向图片上添加补丁
#     PatchApplier: applies adversarial patches to images.
#
#     Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
#
#     """
#
#     def __init__(self):
#         super(PatchApplier, self).__init__()
#
#     def forward(self, img_batch, adv_batch):
#         #  将patch应用到图片上
#         # print('img_batch.shape', img_batch.shape)
#         # print('adv_batch.shape', adv_batch.shape)
#         # 输入[batch, 3, 608, 608], [1, max_lab, 3, 608, 608]
#         advs = torch.unbind(adv_batch, 1)
#         # 按第二维进行切片，即max_lab，最后advs是一个补丁对那个每个lab的集合
#         #  返回指定维度切片后的元组，
#         for adv in advs:
#             #
#             # adv==0时保留img_batch,否则保留adv（补丁处是补丁，其它地方是原图片）
#             # print("adv : ", adv)
#             img_batch = torch.where((adv == 0.), img_batch, adv)
#             '''
#             adv=[1,3,608,608]
#             说明：首先这里的img_batch不一定要求维度和adv一致，这应该是torch.where函数的问题
#             当img_batch也是[1,3,608,608],返回的也是[1,3,608,608]
#             当img_batch是[3,608,608]，返回的也是[1,3,608,608]
#
#             '''
#             # 对每一个框进行操作，如果没有框，对应的补丁全是0，所以也不影响
#             """
#             torch.where(condition,a,b)
#             合并a,b两个tensor，满足条件下保留a，否则是b（元素替换）
#             """
#         return img_batch


class PatchApplier(nn.Module):
    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch_vis, adv_batch_infra):
        """
        img_batch:       [B, 3, H, W]
        adv_batch_vis:   [B, max_lab, 3, H, W]
        adv_batch_infra: [B, max_lab, 3, H, W]
        返回：每张图贴好补丁后的图像 batch: [B, 3, H, W]
        """
        B = img_batch.shape[0]
        patched_batch = []

        for i in range(B):
            img = img_batch[i]  # [3, H, W]

            mean_r = torch.mean(img[0])
            mean_g = torch.mean(img[1])
            mean_b = torch.mean(img[2])
            mean_diffs = torch.tensor([mean_r - mean_g, mean_g - mean_b, mean_b - mean_r])
            is_infra = torch.max(torch.abs(mean_diffs)) < 0.01


            # 选择对应的补丁
            advs = torch.unbind(adv_batch_infra[i] if is_infra else adv_batch_vis[i], dim=0)  # 每个框一个补丁 [3,H,W]

            for adv in advs:
                img = torch.where(adv == 0., img, adv)  # 将非零补丁贴上去

            patched_batch.append(img)

        return torch.stack(patched_batch, dim=0)  # [B, 3, H, W]


'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''


class InriaDataset(Dataset):
    """读取数据集
    InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        """
        输入参数包括：图片地址、标签地址，最大标签，图片尺寸
        """
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))

        print("n_images = ", n_images, '\n', "n_labels = ", n_labels)  # test

        # assert n_images == n_labels, "Number of images and number of labels does't match"
        # 这个地方会做出判断，如果图片数量和标签数不一样则报错
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(
            img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []  # 填入路径
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace(
                '.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        # 分别得到image和lab的路径和名称
        self.max_n_labels = max_lab  # 最大标签数

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace(
            '.jpg', '.txt').replace('.png', '.txt')
        # 分别得到图片和lab的地址
        image = Image.open(img_path).convert('RGB')
        # check to see if label file contains data.
        if os.path.getsize(lab_path):
            label = np.loadtxt(lab_path)
        # 若lab为空，直接注释掉行不行？
        else:
            label = np.ones([5])
            # label = np.ones([0])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        # 这个地方对图片进行预处理，所以训练的时候直接使用处理过后的图片就行
        transform = transforms.ToTensor()  # 这个地方转换为tensor，
        image = transform(image)
        # 需要对image和lab进行pad
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w, h = img.size
        if w == h:
            padded_img = img  #
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
        # 根据输入图片对原始图片进行resize和padding
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)  # choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]  # 最终需要填充的labels数据
        if (pad_size > 0):
            # padded_lab = F.pad(lab, (0, 0, 0, pad_size),
            #                    value=1)  # 给labels中的其他数据填充为1
            # 下面行填充为1
            padded_lab = F.pad(lab, (0, 0, 0, pad_size),
                               value=1e-6)  # 给labels中的其他数据填充为1
        else:
            padded_lab = lab[:self.max_n_labels]
        return padded_lab
        '''
        torch.nn.functional.pad(input, pad, mode, value)
        pad：表示填充方式，分别表示左、右、上、下，此时是下填充
        '''