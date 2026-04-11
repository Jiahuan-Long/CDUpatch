import sys
import json  # 确保导入了json模块
from torch.utils.tensorboard import SummaryWriter
from utils.loss import ComputeLoss
import PIL
import load_data
from tqdm import tqdm
from torch.backends import cudnn
import load_data
from load_data import *
import gc  # java中的垃圾回收机制
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
import subprocess
import copy
import joblib
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import patch_config
import sys
import time
import utils_self
from models.common import DetectMultiBackend
from utils.general import check_img_size
import os
from torchvision import transforms
from PIL import Image
import utils_self
from utils.torch_utils import select_device
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from vis2infra.RGB2infra_trainingwithtorch import RGB2InfraNet

device = select_device('')

'''visdrone = [pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor]'''
'''dronevehicle = [ car truck bus van freight_car]'''

# ----------------------- 全局训练超参数 -----------------------

TV_FACTOR = 2.5
NPS_FACTOR = 0.01
DETECT_FACTOR = 1
CLS_ID = 0              # 要攻击的类别，例如 person = 0，car = 3
NUM_CLS = 5            # 模型总类别数
TOPK_NUM = 20            # 仅在 reduction='topk' 时生效
REDUCTION_STRATEGY = 'static-max'  # 可选: 'dynamic', 'static-mean', 'static-topk', 'static-max'

device = torch.device('cuda:0')  # 显卡序号信息
GPU_device = torch.cuda.get_device_properties(device)  # 显卡信息
# 输出显卡参数
print("GPUs : ", GPU_device)


def convert_vis_patch_to_infra_tensor(patch_tensor, model, patch_size):
    patch_tensor = patch_tensor.unsqueeze(0)  # [1,3,H,W]
    B, C, H, W = patch_tensor.shape
    x = patch_tensor.permute(0, 2, 3, 1).reshape(-1, 3)  # [H*W, 3]
    pred_gray = model(x).reshape(B, 1, H, W)  # [1, 1, H, W]
    rgb_tensor = pred_gray.repeat(1, 3, 1, 1)  # 转为三通道 [1, 3, H, W]
    return rgb_tensor.squeeze(0)  # [3, H, W]

def get_prob_extractor(epoch, config):
    if REDUCTION_STRATEGY == 'dynamic':
        if epoch < 100:
            mode = 'mean'
        elif epoch < 200:
            mode = 'topk'
        else:
            mode = 'max'
    elif REDUCTION_STRATEGY == 'static-topk':
        mode = 'topk'
    elif REDUCTION_STRATEGY == 'static-mean':
        mode = 'mean'
    elif REDUCTION_STRATEGY == 'static-max':
        mode = 'max'
    else:
        raise ValueError(f"不支持的 REDUCTION_STRATEGY: {REDUCTION_STRATEGY}")

    extractor = MaxProbExtractor(
        cls_id=CLS_ID,
        num_cls=NUM_CLS,
        config=config,
        reduction=mode,
        topk=TOPK_NUM
    )
    return extractor, mode

class PatchTrainer(object):  # 继承object类

    def __init__(self, mode):
        '''参数配置'''
        self.config = patch_config.patch_configs[mode]()

        # 设置日志文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_folder = os.path.join('patchTraining', f'training_run_{timestamp}')
        os.makedirs(self.run_folder, exist_ok=True)

        # 保存输出到文件
        log_file = os.path.join(self.run_folder, "training_output.txt")
        sys.stdout = open(log_file, 'w')  # 将print输出重定向到文件

        # 输出重要的初始化参数
        print("训练模式 : ", mode)  # 打印出训练模式
        print(f"数据集路径 : {self.config.img_dir}, {self.config.lab_dir}")  # 数据集路径
        print(f"图片尺寸 : {self.config.img_size}, 补丁大小 : {self.config.patch_size}")
        print(f"权重文件 : {self.config.weightfile}")

        print("缩放因子 : ", load_data.SCALE)
        print("TV因子 : ", TV_FACTOR)
        print("NPS因子 : ", NPS_FACTOR)
        print("DETECT因子 : ", DETECT_FACTOR)

        print("\n--- 对抗训练核心参数 ---")
        print(f"攻击类别 CLS_ID           : {CLS_ID}")
        print(f"模型总类别数 NUM_CLS     : {NUM_CLS}")
        print(f"TOPK_NUM（仅topk生效）   : {TOPK_NUM}")
        print(f"Loss Reduction 策略      : {REDUCTION_STRATEGY}")
        print("--- ------------------ ---\n")

        print("训练模式 : ", mode)  # 打印出训练模式
        self.sigmoid = True
        print("使用 sigmoid = ", self.sigmoid)
        self.model = DetectMultiBackend(weights=self.config.weightfile, device=device).eval()

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        # self.prob_extractor = MaxProbExtractor(0, 5, self.config).cuda()
        # self.prob_extractor = MaxProbExtractor(3, 10, self.config).cuda()  # visdrone和dronevehicle的检测器不同


        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        # 根据时间戳为每次训练创建新的文件夹
        self.epoch_images_folder = os.path.join(self.run_folder, "epoch_images")
        os.makedirs(self.epoch_images_folder, exist_ok=True)

        # 为当前训练创建独立的TensorBoard日志文件夹
        self.log_dir = os.path.join(self.run_folder, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # TensorBoard 写入器初始化
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # 调用补丁映射模型
        self.mapping_model = RGB2InfraNet().cuda()
        self.mapping_model.load_state_dict(torch.load('vis2infra/RGB2infra_with_torch.pth'))
        self.mapping_model.eval()


    def train(self):
        """
        优化补丁以生成对抗样本。
        :return: 训练后的补丁
        """
        n_png_images = len(fnmatch.filter(os.listdir(self.config.img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(self.config.img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        print("训练集中的总图像数 : ", n_images)

        n_lab = len(fnmatch.filter(os.listdir(self.config.lab_dir), '*.txt'))
        length_txt_boxes, _ = utils_self.txt_len_read(self.config.lab_dir)
        print("训练集中的总实例数 : ", length_txt_boxes)
        print("训练集中的txt文件数 : ", n_lab)

        img_size = self.config.img_size
        print("补丁大小 : ", self.config.patch_size)

        batch_size = self.config.batch_size
        print("批次大小 = ", batch_size)

        max_n_epochs = 601
        max_lab = 60

        adv_patch_cpu = self.generate_patch("random")
        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=True),
            batch_size=batch_size, shuffle=True, num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'一个epoch有{len(train_loader)}个batch')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        print("原始学习率 : ", optimizer.param_groups[0]['lr'])
        scheduler = self.config.scheduler_factory(optimizer)

        for epoch in range(max_n_epochs):
            self.prob_extractor, mode = get_prob_extractor(epoch, self.config)

            ep_det_loss = 0
            ep_cls_loss = 0.
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            et0 = time.time()

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'正在运行第{epoch}个epoch',
                                                        total=self.epoch_length):
                with torch.autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    adv_patch = adv_patch_cpu.cuda()
                    infra_patch = convert_vis_patch_to_infra_tensor(adv_patch, self.mapping_model, self.config.patch_size)

                    adv_batch_vis = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    adv_batch_infra = self.patch_transformer(infra_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)


                    p_img_batch = self.patch_applier(img_batch, adv_batch_vis, adv_batch_infra)
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))

                    # # 可视化p_img_batch
                    # grid_img = make_grid(p_img_batch.cpu().detach().clone(), nrow=2, normalize=True, scale_each=True)
                    # # 将 tensor 转为 HWC 格式，并转换为 numpy 数组
                    # grid_np = grid_img.permute(1, 2, 0).numpy()
                    # plt.figure(figsize=(8, 8))
                    # plt.imshow(grid_np)
                    # plt.axis('off')
                    # plt.savefig("debug_patch.png")  # 保存图像


                    outputs = self.model(p_img_batch)

                    if self.sigmoid:
                        outputs_extract = outputs[0]
                    else:
                        outputs_extract = outputs[1]

                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)



                    max_prob = self.prob_extractor(outputs_extract)
                    det_loss = torch.mean(max_prob)
                    nps_loss = nps * NPS_FACTOR
                    tv_loss = tv * TV_FACTOR
                    det_loss = det_loss * DETECT_FACTOR

                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)

            scheduler.step(ep_loss)

            et1 = time.time()
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_nps_loss = ep_nps_loss / len(train_loader)
            ep_tv_loss = ep_tv_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)

            print(
                f'第{epoch}个epoch, 总损失: {ep_loss}, 检测损失: {ep_det_loss}, NPS损失: {ep_nps_loss}, TV损失: {ep_tv_loss}')
            print(f'第{epoch}个epoch时间: {et1 - et0}')


            self.writer.add_text("config/CLS_ID", str(CLS_ID))
            self.writer.add_text("config/NUM_CLS", str(NUM_CLS))
            self.writer.add_text("config/TOPK_NUM", str(TOPK_NUM))
            self.writer.add_text("config/REDUCTION_STRATEGY", REDUCTION_STRATEGY)
            if epoch % 10 == 0:
                print(f"[Epoch {epoch}] 当前使用的 loss reduction 模式：{mode}")
                self.writer.add_text("loss/reduction_mode", mode, epoch)


            # 保存补丁图像及可视化
            if epoch % 1 == 0:
                save_patch = adv_patch_cpu.cpu()  # 直接使用张量
                save_patch_name = f"{epoch}_patch.png"
                save_patch_image_path = os.path.join(self.epoch_images_folder, save_patch_name)

                # 如果需要，也可以保存为PNG文件
                save_patch_pil = transforms.ToPILImage()(save_patch)
                save_patch_pil.save(save_patch_image_path)

                # 将补丁图像写入TensorBoard
                self.writer.add_scalar('损失/总损失', ep_loss, epoch)
                self.writer.add_scalar('损失/检测损失', ep_det_loss, epoch)
                self.writer.add_scalar('损失/NPS损失', ep_nps_loss, epoch)
                self.writer.add_scalar('损失/TV损失', ep_tv_loss, epoch)
                self.writer.add_image('补丁图像', save_patch, epoch)


                # 新增：将粘贴补丁后的图像写入TensorBoard
                # p_img_batch 是整个批次的图像，可以使用 make_grid 制作成一个网格显示
                patched_grid = make_grid(p_img_batch.cpu().detach().clone(), nrow=2, normalize=True, scale_each=True)
                self.writer.add_image('补丁后的图像', patched_grid, epoch)

        return adv_patch_cpu

    def generate_patch(self, type):
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))
        return adv_patch_cpu

    def read_image(self, path):
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu




# 主程序
def main():
    argvs = "paper_obj"
    trainer = PatchTrainer(argvs)
    final_patch = trainer.train()


if __name__ == '__main__':
    print("开始训练时间 : ", time.strftime('%Y-%m-%d %H:%M:%S'))
    # print("缩放因子 : ", load_data.SCALE)
    # print("TV因子 : ", TV_FACTOR)
    # print("NPS因子 : ", NPS_FACTOR)
    t_begin = time.time()
    main()
    t_end = time.time()
    print('总训练时间: {:.4f}分钟'.format((t_end - t_begin) / 60))
