# CDUPatch

## CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库是 ACM Multimedia（ACM MM）论文对应的代码实现：

**CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors**

CDUPatch 是一个面向研究的代码仓库，用于在 YOLO 系列目标检测器上训练和评估对抗补丁。项目基于 Ultralytics YOLOv3/YOLOv5 的代码结构，并在此基础上增加了适用于可见光与可见光/红外混合场景的补丁攻击流程。

## 项目功能

- 使用 `train_patch.py` 训练通用对抗补丁
- 使用 `load_data.py` 中的几何变换与贴图逻辑把补丁贴到目标上
- 使用 `test_patch.py` 评估攻击效果
- 保留 `vis2infra/` 目录，用于可见光补丁到红外风格补丁的映射

在这个项目里，被优化的对象不是检测器本身，而是补丁图像。检测模型作为固定目标，训练会尽量降低某个目标类别的目标置信度和或类别置信度。

## 关键文件

- `train_patch.py`：对抗补丁训练主脚本
- `test_patch.py`：评估脚本，可计算 precision、recall 和 ASR
- `patch_config.py`：数据路径、补丁大小、损失形式和训练配置
- `load_data.py`：补丁变换、补丁贴图、最大置信度提取、NPS 与 TV 等工具
- `vis2infra/RGB2infra_trainingwithtorch.py`：可见光到红外的映射辅助模块
- `Readme_patch.md`：原始补丁测试说明

## 目录结构

```text
CDUpatch/
├── train_patch.py
├── test_patch.py
├── patch_config.py
├── load_data.py
├── models/
├── utils/
├── vis2infra/
├── classify/
├── segment/
└── non_printability/
```

## 环境要求

- Python 3.8 及以上
- 强烈建议使用带 CUDA 的 PyTorch
- 安装依赖：

```bash
pip install -r requirements.txt
```

## 运行前准备

当前脚本中存在多个本地绝对路径，训练和评估前需要按你的环境修改：

- `patch_config.py`
- `test_patch.py`
- 如有需要，`data/` 下的数据集配置文件也需要同步调整

仓库不会跟踪较大的 YOLO 检测模型权重，例如 `yolov3.pt`、`yolov5x.pt` 等 `.pt` 文件。`vis2infra/` 目录及其中的映射模型文件会保留在仓库中。

## 发布状态

- 当前仓库已包含论文对应的主要代码。
- 相关数据集与预训练权重正在整理中，后续会补充上传。
- 较大的 YOLO 检测器权重目前仍默认由使用者在本地自行准备。

## 训练补丁

直接运行：

```bash
python train_patch.py
```

默认流程包括：

- 按 `patch_config.py` 加载检测模型权重
- 随机初始化补丁作为优化目标
- 根据目标框对补丁做缩放、旋转和贴图
- 计算检测损失、不可打印性损失和总变分损失
- 将训练日志、每轮补丁图像和 TensorBoard 日志保存到 `patchTraining/`

## 测试补丁

在验证集上评估补丁：

```bash
python test_patch.py
```

评估脚本可以：

- 把补丁贴到目标上
- 保存贴补丁后的图像和检测可视化结果
- 导出预测标签
- 计算 precision、recall 和 ASR

## 说明

- 这是研究性质代码，默认使用方式面向熟悉对抗补丁实验的用户。
- 当前实现默认依赖 CUDA，如需 CPU 环境运行，通常需要做少量修改。
- 数据集、大型 YOLO 模型权重、生成结果和临时实验文件不会纳入版本控制。

## 致谢

- 项目建立在 Ultralytics YOLOv3/YOLOv5 的代码结构之上
- 针对 VisDrone 与可见光/红外混合场景的对抗补丁研究进行了定制化修改

## 许可证

许可证信息见 `LICENSE`。如果你要再次分发本项目，请同时核对上游依赖及其许可证要求。
