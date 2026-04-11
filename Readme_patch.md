## 补丁测试说明

1. 修改验证集路径：文件test_patch.py中line25-line26，修改成如下形式

```
imgdir = '/home/wdh/mnt/Dataset/VisDrone2019/VisDrone2019-DET-val/images'
labdir = "/home/wdh/mnt/Dataset/VisDrone2019/VisDrone2019-DET-val/testset"
```

2. 修改模型权重路径，line30

```
weightfile = "weights/yolov3.pt"   # visdrone上预训练的数据集
```

3. 修改数据配置文件，line30

```
datayaml = "data/VisDrone.yaml中的配置修改成如下形式
# path: D:/VSCode/Datasets/visDrone2019  # dataset root dir /home/wdh/mnt/Dataset/VisDrone2019
path: /home/wdh/mnt/Dataset/VisDrone2019  # dataset root dir 
train: VisDrone2020-DET-train/images  # train images (relative to 'path')  6471 images
val: VisDrone2019-DET-val/images  # val images (relative to 'path')  548 images
test: VisDrone2019-DET-test-dev/images  # test images (optional)  1610 images

```

4. 修改补丁路径，以及结果保存路径：line35，37

```
patchfile = "/home/wdh/ComparisonMethodPro/Yolov3_Demo/patch2eval/v5/v5-combine-scale-1.png"  # 任选一张补丁图片
savedir = "/home/wdh/ComparisonMethodPro/Yolov3_Demo/patch2eval/v5/"
```

5. 新建保存结果的相关文件：在savedir后，即放在38行开始

```
if not os.path.exists(os.path.join(savedir, 'proper_patched/')): os.mkdir(os.path.join(savedir, 'proper_patched/'))  # 保存添加补丁的图片目录
if not os.path.exists(os.path.join(savedir, 'pre_patched/')): os.mkdir(os.path.join(savedir, 'pre_patched/'))  # 保存添加补丁前的图片目录
if not os.path.exists(os.path.join(savedir, 'yolo-labels/')): os.mkdir(os.path.join(savedir, 'yolo-labels/'))  # 仅写入筛选后的数据
if not os.path.exists(os.path.join(savedir, 'yolo-labels_w_conf/')): os.mkdir(os.path.join(savedir, 'yolo-labels_w_conf/'))   # 写入所有数据，包括置信度信息
```

6. 运行以下命令，开始计算结果

```
\> conda activate ml
\> python test_patch.py
```

