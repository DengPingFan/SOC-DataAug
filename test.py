import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from scipy import misc
import cv2
from model.ResNet_models import ResNet_Baseline
from data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = '/home/pz1/datasets/sod/RGB_Dataset/test/img/'
test_model = './models/Resnet/Model_29.pth'
model = ResNet_Baseline()
model.load_state_dict(torch.load(test_model))

model.cuda()
model.eval()

test_datasets = ['DUTS_Test','DUT','ECSSD','HKU-IS','SOD','PASCAL']

for dataset in test_datasets:
    save_path = './results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        res = model(image)
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res)
