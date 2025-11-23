import torch
from easydict import EasyDict
import albumentations as A

cfg = EasyDict()  # 访问属性的方式去使用key-value 即通过 .key获得value


cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.max_epoch = 50  # 50

# batch size
cfg.train_bs = 1   # 32
cfg.valid_bs = 1   # 24
cfg.workers = 2  # 16


# 学习率
cfg.exp_lr = False   # 采用指数下降
cfg.lr_init = 0.01
cfg.factor = 0.1
cfg.milestones = [25, 45]
cfg.weight_decay = 5e-4
cfg.momentum = 0.9

cfg.is_warmup = False
cfg.hist_grad = False

cfg.log_interval = 10

cfg.bce_pos_weight = torch.tensor(0.75)  #  pos_weight=负样本数量 / 正样本数量

cfg.in_size = 512   # 输入尺寸最短边

norm_mean = (0.5, 0.5, 0.5)  # 比imagenet的mean效果好
norm_std = (0.5, 0.5, 0.5)

cfg.tf_train = A.Compose([
    A.Resize(width=cfg.in_size, height=cfg.in_size),
    A.Normalize(norm_mean, norm_std),
])

cfg.tf_valid = A.Compose([
    A.Resize(width=cfg.in_size, height=cfg.in_size),
    A.Normalize(norm_mean, norm_std),
])