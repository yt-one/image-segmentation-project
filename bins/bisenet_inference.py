import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import argparse
import torch
import cv2
import time
import numpy as np
from models.build_BiSeNet import BiSeNet
import albumentations as A

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--path_checkpoint', default=r"D:\Learn\Computer Science\deepLearning\visionPytorch\results\11-23_16-04\checkpoint_49.pkl",  # todo: 模型保存路径
                    help="path to your dataset")
parser.add_argument('--path_img', default=r"D:\Learn\Datasets\Portrait-dataset-2000\dataset\testing\00016.png",  # todo： 预测的图片路径
                    help="path to your dataset")
parser.add_argument('--data_root_dir', default=r"...",  # todo：数据根目录
                    help="path to your dataset")
args = parser.parse_args()

if __name__ == '__main__':

    path_img = args.path_img
    in_size = 512  # 224， 448 ， 336 ， 1024
    norm_mean = (0.5, 0.5, 0.5)  # 比imagenet的mean效果好
    norm_std = (0.5, 0.5, 0.5)
    # step1: 数据预处理

    img_bgr = cv2.imread(path_img)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_transform = A.Compose([
        A.Resize(width=in_size, height=in_size),
        A.Normalize(norm_mean, norm_std),
    ])

    transformed = img_transform(image=img_rgb)
    img_rgb = transformed['image']

    img_rgb = img_rgb.transpose((2, 0, 1))  # hwc --> chw
    img_tensor = torch.from_numpy(img_rgb).float().unsqueeze(0)
    img_tensor = img_tensor.to(device)


    # step2: 模型加载
    model = BiSeNet(num_classes=1, context_path="resnet101")
    checkpoint = torch.load(args.path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # step3: 推理
    with torch.no_grad():
        # 因为cpu/gpu是异步的，因此用cpu对GPU计时，应当等待gpu准备好再计时
        # 即使这个操作是等待GPU全部执行结束，CPU才可以读取时间信息
        torch.cuda.synchronize()
        s = time.time()
        outputs = model(img_tensor)
        torch.cuda.synchronize()  # 该操作是等待GPU全部执行结束，CPU才可以读取信息
        print(f"{time.time() - s:.4f}")
        outputs = torch.sigmoid(outputs).squeeze(1)
        pre_label = outputs.data.cpu().numpy()[0]


    # step4：后处理显示图像
    background = np.zeros_like(img_bgr, dtype=np.uint8)
    background[:] = 255
    alpha_bgr = pre_label
    alpha_bgr = cv2.cvtColor(alpha_bgr, cv2.COLOR_GRAY2BGR)
    h, w, c = img_bgr.shape
    alpha_bgr = cv2.resize(alpha_bgr, (w, h))
    # fusion
    result = np.uint8(img_bgr * alpha_bgr + background * (1 - alpha_bgr))
    out_img = np.concatenate([img_bgr, result], axis=1)
    cv2.imshow("result", out_img)
    cv2.waitKey()




