'''
2023-06-09
调试基于VisDrone数据集的补丁攻击
'''

from torch import optim
import torch


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "/home/admin1/datasets/mix-infra-vis/images"
        self.lab_dir = "/home/admin1/datasets/mix-infra-vis/labels"
        # self.img_dir = "/home/admin1/datasets/VisDrone/VisDrone2019-DET-val/images"
        # self.lab_dir = "/home/admin1/datasets/VisDrone/VisDrone2019-DET-val/labels"
        # self.img_dir = "/home/admin1/jiahuan/advpatch/YOLOv3_Demo/inria/pos"
        # self.lab_dir = "/home/admin1/jiahuan/advpatch/YOLOv3_Demo/inria/pos/yolo-labels"
        # self.img_dir = "/home/admin1/datasets/GT-frames/images"

        # self.lab_dir = "/home/admin1/datasets/GT-frames/labels"

        self.img_size = 640     #   图片大小
        # self.cfgfile = "data/VisDrone.yaml"  # "cfg/yolo.cfg" for origin
        # 似乎这里也可以用yolov2.cfg，这样的区别在于输入图片的大小
        # self.weightfile = "/home/admin1/jiahuan/advpatch/YOLOv3_Demo/weights/mix-infra-vis-epoch20.pt"
        # self.weightfile = "/home/admin1/jiahuan/advpatch/YOLOv3_Demo/weights/yolov3.pt"
        self.weightfile = "/home/admin1/datasets/advpath/YOLOv3v5-attack-25.10.26/runs/train/yolov5x-mix-infra-vis-v3-toudu-30epoch.pt"

        self.printfile = ("non_printability/30values.tx"
                          "t")
        self.patch_size = 600  #

        self.start_learning_rate = 0.03  # self.start_learning_rate = 0.03
        #   这里到底是0.1还是0.03？

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(
            x, 'min', patience=50)

        self.max_tv = 0

        self.batch_size = 4  # 这是基类，在下面被继承时会被重写

        self.loss_target = lambda obj, cls: obj  # 默认为两个参数的乘积

        self.target_loc = torch.tensor([0.,0.,0.01,0.01])


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    对高分辨率进行试验
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'


class Experiment3LowRes(Experiment1):
    """
    Lower res
    对低分辨率进行试验
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"


class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()
        
        self.batch_size = 8  # origin = 8  # 6/10/18/24
        self.patch_size = 500  # 放大，设置为500

        # self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls


class ObjectAndClass(BaseConfig):
    """
    obj_conf+cls_conf
    """

    def __init__(self):
       
        super().__init__()

        self.batch_size = 8 # origin = 8  # 6/10/18/24
        self.patch_size = 600  # 放大，设置为500

        self.patch_name = 'ObjectAndClass'
        self.max_tv = 0.165

        # self.loss_target = lambda obj, cls: (0.2*obj+0.8*cls)   #   权重参数？
        # self.loss_target = lambda obj, cls: (obj+0.001*cls)
        self.loss_target = lambda obj, cls: (obj*cls)


class ReproducePaperObj(BaseConfig):  # 都继承了BaseConfig类
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 2 # origin = 8  # 6/10/18/24
        self.patch_size = 600  # 放大，设置为500

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        # self.loss_target = lambda obj, cls: obj+cls
        self.loss_target = lambda obj, cls: obj
        # lambda函数的语法，冒号前是参数，后是返回值
        # 对于这种情况，只返回目标检测的概率，对应`load_data.py`中的86行


patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "obj_cls": ObjectAndClass,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj  # 再生成和文中最小化目标得分的补丁
}
# 定义的参数是个字典，通过键访问值