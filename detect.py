# YOLOv3 🚀 by Ultralytics, GPL-3.0 license

"""
专用于物理攻击实验
"""
"""
Run YOLOv3 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from torchvision import transforms
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
print("ROOT : ", ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#   这里是在操作路径

from models.common import DetectMultiBackend
from models.yolo import DetectionModel
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov3.pt',  # model path or triton URL
        source=ROOT / 'runs',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/VisDrone.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'TrainedModelTest/imgs_pre',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rte stride
):
    source = str(source)
    save_img     = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #   要不要保存标签的选择？
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'imgs_pre' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    
    #   直接使用models.yolo()中的输出
    # from models.experimental import attempt_download, attempt_load
    # model_detection = DetectionModel(cfg='models/yolov3.yaml').eval().cuda()
    
    # model_detection = attempt_load(weights)
    
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        ##################################################################################################
        #   这里返回的数据，将最大边变成了640，然后另一个边进行了同缩比，但是也会有填充，具体是如何操作的，还不是很清楚。
        #   例如，原图片大小为765*1360
        #   缩放后，765*(640/1360)X640 = 360 X 640
        ##################################################################################################
    vid_path, vid_writer = [None] * bs, [None] * bs
    # print("dataset : ", dataset)
    
    #   什么时候将数据resize成640*640大小？？
    #   注意！！！注意！！！在这个版本中，并没有将数据resize成固定的640x640大小，
    #   而是将长边设置为640，然后缩放短边，并且是32的整数倍（可能有填充）
    # Run inference
    # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    #   warmup似乎可以不要，这里输入的尺寸是imgsz，和输入图片无关
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # print("dt : ", dt)
    #   下面的输出是三部分，会不会和dt有关？
    num = 0
    for path, im, im0s, vid_cap, s in dataset:
        #   im: CHW格式
        #   im0s: HWC，且这个才是真实图片的尺寸
        # print("s : ", s)
        with dt[0]:
            # print("device : ", model.device)
            im = torch.from_numpy(im).to(model.device)
            #   转为torch。是啊，这里加载的图片格式应该是numpy格式的，然后转为tensor
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            #   上一行再是转为ToTensor
            # print("im single channel : ", im[0,:,0])
            #   通过上面的的测试，可以输出某些像素
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
                #   还可以这么升维？？

        # Inference
        # model_pred = model_detection(im)
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            #   确定一下这里im的数据格式
            # print("img shape : ", im.size(), '\n', im)
            # #   保存一下im试试，
            # savedir = "TrainedModelTest/detect_test"
            # savename = savedir + "/" + "test_detect.png"
            # img_save = im.squeeze(0)
            
            # img_pil = transforms.ToPILImage('RGB')(img_save)
            # img_pil.save(savename)
            #   im: C x H x W
            pred = model(im, augment=augment, visualize=visualize)
            '''
            2023-06-11——确定模型的输出
            1. 模型的输出包含两部分，是一个元组(item1,item2)；
            2. 其中第一部分是经过了detection head decode部分，[batch, 25200, 15]，同时cat到了一起，直接用于后面的NMS处理，以及作图等操作；
            3. 第二部分是模型的直接输出，是一个list，没有经过decode，同时也没有进行过cat，是一个三维尺度输出。正好是我们的要求输出部分！！！
            4. 不一样的是输出的格式，为:
                                        [[1x3x48x80x15]
                                         [1x3x24x40x15]
                                         [1x3x12x20x15]]
                这倒不存在大问题，后面进行`x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()`逆操作
                即先permute(0,1,4,2,3)——直接指定现在维度是原来的map维度即可。
                然后再view(bs, self.na * self.no, ny, nx)即可。
            '''
            #   这里的pred有两部分数据，奇怪？？
            #   这里的第一部分和第二部分输出有什么区别？？
            #   还是说第二部分没有激活？？
            #   也就是说，传给model的还是[0,1]tensor数据
            '''
            这里输出两部分：
            （1）大小为1x15120x15=1x(48*80+24*40+12*20)*3x15，也就是说，这里没有将图片进行resize到640x640
            而是直接基于缩放的图片进行预测，这里的的48=384/8，24=384/16，12=384/32，384为图片的高度，即H部分
            （2）第二部分输出为一个list：
                                        [[1x3x48x80x15]
                                         [1x3x24x40x15]
                                         [1x3x12x20x15]]
            按理来讲，这两部分应该是一样的啊，但是在下面的NMS处理中，其实使用了第一部分的输出。
            '''
            # 这里返回的数据，是否已经进行了归一化？

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            #   这里的classes是指定输出特定的类别
            #   这里只用了pred[0]，即输出的第一部分：[1x25200x15]（80x80+40x40+20x20）*3
            #   这里的输出格式为nx6
            #   其中的6为([x1, y1, x2, y2], conf, cls_id)
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                #   这里的p是什么？
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            #   这里又是im0，是im0s的copy
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #   这里是要归一化，gn是多大数据？
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                #   对前面四个数据[x1,y1,x2,y2]缩放到原图大小
                #   这里也放缩到原图大小，然后又基于原图归一化处理
                # Print results
                for c in det[:, 5].unique():
                    #   这里是根据.unique()对类别进行id排序，默认为升序
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                '''
                print("detection obj conf : ", det[:,4])
                for instance in det:
                    print(instance)
                    #   并且这里直接每一次都输出一个instance的检测结果不就行了吗？为什么要用下面的
                    for *xyxy, conf, cls in reversed(det):
                    这个格式？
                # Write results
                for *xx, yy, zz in det:
                    True
                '''
                # reversed的作用
                for *xyxy, conf, cls in reversed(det):
                    #   这里的reversed是对det进行整体取逆吗？
                    #   最后一行到第一行？？
                    #   确实是对所有数据进行逆序，但是为什么要这么处理呢？
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #   这里通过xyxy2xywh，又将[x1,y1,x2,y2] --> [x,y,w,h]
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # 然后这里给出了如果保存数据，数据保存的格式为[cls_id, [x,y,w,h], obj_conf]
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # if c ==3:
                            #   2023-07-09修改，仅绘制car类别
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    #   2023-07-03修改，即使是视频，也保存图片
                    #   str(save_dir / p.name)
                    img_name = str(num) + '_.png'
                    save_path_img = str(save_dir / 'imgs_pre' / img_name )
                    cv2.imwrite(save_path_img, im0)
                    num += 1
                    #   也可以每隔多少frame保存图片(if i // 30 ==0)
                    #   修改save_path即可
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #   修改
                            fps = 10
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 10, im0.shape[1], im0.shape[0]
                            #   保存图片的时候设置fps=1试试效果
                            # fps, w, h = 10, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)    #   这里是保存视频
                    #   （2023-07-03）这里尝试在视频检测过程中也保存图片。

        # Print time (inference-only)
        print("s : ",s)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default=ROOT / 'weights/yolov3.pt',
                        help='model path or triton URL')
    # parser.add_argument('--source', type=str, default='D:/Doc_staff/projects/visdrone_YOLO/Physical_attacks/creation/toy_test/imgs', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/VisDrone.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='D:/Doc_staff/projects/visdrone_YOLO/Physical_attacks/creation/toy_test/imgs_pre', help='save results to project/name')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=5, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=10, help='video frame-rate stride')
    # parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

#   这里有两个参数传入口，上面的似乎是继承下面的参数传递，下面的是dominant参数
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


'''
usage:

default:
    python detect.py --weights weights/yolov5s.pt
    default directory: data/images
    
single image:
    python detect.py --weights weights/yolov5s.pt --source data/images/bus.jpg
    
path/directory:
    python detect.py --weights weights/yolov5s.pt --source data/images/
    
path/directory:
    python detect.py --weights weights/yolov3.pt --save-txt --save-conf
    如上所示，--save-txt表示是否要保存预测的label文件，--save-conf表示是否要在label文件中保存obj_conf
    最后的格式为：
    [cls_id, [x,y,w,h], obj_conf]
'''