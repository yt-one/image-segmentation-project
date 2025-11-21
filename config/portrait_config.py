import torch
import torchvision.transforms as transforms
from easydict import EasyDict

cfg = EasyDict()


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

cfg.log_interval = 10

cfg.bce_pos_weight = torch.tensor(0.75)  #  [36638126., 48661074.]) pose_weight=负样本数量 / 正样本数量

cfg.in_size = 512   # 输入尺寸最短边

norm_mean = (0.5, 0.5, 0.5)  # 比imagenet的mean效果好
norm_std = (0.5, 0.5, 0.5)

cfg.tf_train = transforms.Compose([
    transforms.Resize(width=cfg.in_size, height=cfg.in_size),
    transforms.Normalize(norm_mean, norm_std),
])

cfg.tf_valid = transforms.Compose([
    transforms.Resize(width=cfg.in_size, height=cfg.in_size),
    transforms.Normalize(norm_mean, norm_std),
])





