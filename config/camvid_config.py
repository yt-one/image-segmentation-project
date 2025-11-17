import torch
from easydict import EasyDict

cfg = EasyDict()  # 访问属性的方式去使用key-value 即通过 .key获得value

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.max_epoch = 150  # 150

cfg.crop_size = (360, 480)

# batch size
cfg.train_bs = 2  # 8
cfg.valid_bs = 1  # 4
cfg.workers = 1  # 16

# 学习率
cfg.lr_init = 0.1  # pretraied_model::0.1
cfg.factor = 0.1
cfg.milestones = [75, 130]
cfg.weight_decay = 1e-4
cfg.momentum = 0.9

cfg.log_interval = 10