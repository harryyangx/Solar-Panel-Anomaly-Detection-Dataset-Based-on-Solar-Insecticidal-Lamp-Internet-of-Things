#coding:utf-8
import random
import numpy as np
import torch
from torch.backends.cudnn import deterministic

from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg')


# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/11/yolo11s_sim.yaml"
#数据集配置文件
data_yaml_path = '/home/pc/solar_panels_train/ultralytics_new20250625/datasets/datasets_3000/data.yaml'
#预训练模型
pre_model_name = '/home/pc/solar_panels_train/ultralytics_new20250625/runs/detect/11s_compare_s/weights/compare_s.pt'

torch.manual_seed(2025)
random.seed(2025)
np.random.seed(2025)


if __name__ == '__main__':
    #加载预训练模型
    #model = YOLO(model_yaml_path).load(pre_model_name)
    model = YOLO(model_yaml_path) #无迁移学习
    #训练模型
    results = model.train(data=data_yaml_path,
                          epochs=330,
                          batch=16,
                          name='11s_compare_HSPPFP_C2PSAF_oddall_CIoU',
                          optimizer='SGD',
                          cos_lr=True,
                          imgsz=640,
                          workers=8,
                          seed=2025,
                          deterministic=True,
                          #patience=50
                          )
