import os
import torch
import torch.nn as nn
import numpy as np
import math
import PIL


def dir_exists(path):
    """
    检查路径是否存在，不存在则自动创建。

    Parameters
    ----------
    path : str
        需要检查或创建的文件夹路径。

    Notes
    -----
    常用于训练过程中的日志目录、模型保存目录创建。
    """
    if not os.path.exists(path):
        os.makedirs(path)


def initialize_weights(*models):
    """
    对传入的模型进行统一权重初始化。

    Parameters
    ----------
    *models : nn.Module
        任意数量的 PyTorch 模型。

    Notes
    -----
    - Conv2d 使用 Kaiming Normal 初始化（适合 ReLU）
    - BatchNorm2d 的 weight 初始化为 1，bias 初始化为 1e-4
    - Linear 层使用均值 0，方差很小的正态分布初始化权重，bias 置 0

    用于保证训练开始时的梯度稳定，提升模型收敛速度。
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    构造双线性插值的转置卷积（Deconv）初始化权重。

    Parameters
    ----------
    in_channels : int
        输入通道数。
    out_channels : int
        输出通道数。
    kernel_size : int
        卷积核大小。

    Returns
    -------
    torch.Tensor
        形状为 (in_channels, out_channels, k, k) 的双线性插值卷积核。

    Notes
    -----
    在语义分割 FCN / DeepLab 等模型中，
    常用于初始化上采样层（transposed convolution），
    使其初始行为等价于传统双线性插值，从而使训练更稳定。
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


def colorize_mask(mask, palette):
    """
    根据颜色映射表将 mask 转为带调色板的彩色图像。

    Parameters
    ----------
    mask : numpy.ndarray
        语义分割的 mask（H×W，每个像素为类别 ID）。
    palette : list[int]
        长度为 3×类别数的列表，表示 RGB palette。

    Returns
    -------
    PIL.Image
        转换好的带色彩图。

    Notes
    -----
    语义分割常用的可视化方法，将类别 ID 映射为对应颜色。
    若 palette 长度不足，会自动补 0。
    """
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def set_trainable_attr(m, b):
    """
    设置某个模块的参数是否可训练。

    Parameters
    ----------
    m : nn.Module
        需要设置的模块。
    b : bool
        True 表示可训练，False 表示冻结。

    Notes
    -----
    设置：
    - m.trainable 属性（自定义）
    - 所有参数的 requires_grad
    """
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    """
    对模型的所有子模块递归调用某个函数。

    Parameters
    ----------
    m : nn.Module 或 list/tuple
        模块或模块列表。
    f : callable
        对每个子模块执行的函数。

    Notes
    -----
    用于批量操作如冻结模型、修改属性等。
    """
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    """
    将模型或模块列表中所有层设为可训练或冻结。

    Parameters
    ----------
    l : nn.Module or list
        模型或模块集合。
    b : bool
        True 表示可训练，False 表示冻结。

    Notes
    -----
    常用于迁移学习场景，比如：
    - 冻结 backbone
    - 只训练 head（分类头）
    """
    apply_leaf(l, lambda m: set_trainable_attr(m, b))
