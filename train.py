import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from model.ResNet_models import ResNet_Baseline
from data import get_loader
from tools import SaliencyStructureConsistency

from config import Config
from vis import visualize_pred, visualize_gt

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=20, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
opt = parser.parse_args()

config = Config()

print('Learning Rate: {}'.format(opt.lr))
model = ResNet_Baseline()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = config.image_root
gt_root = config.gt_root
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()

def generate_smoothed_gt(gts, epsilon=0.001):
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts

print("Let's go!")
for epoch in range(1, (opt.epoch+1)):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        atts = model(images)

        loss = CE(atts, gts)

        if config.label_smooth:
            loss = 0.5*(loss + CE(atts, generate_smoothed_gt(gts)))

        if config.lambda_loss_ss:
            images_scale = F.interpolate(images, scale_factor=0.3, mode='bilinear', align_corners=True)
            sal_scale = model(images_scale)
            sal_s = F.interpolate(atts, scale_factor=0.3, mode='bilinear', align_corners=True)
            loss_ss = SaliencyStructureConsistency(torch.sigmoid(sal_scale), torch.sigmoid(sal_s))

            loss += config.lambda_loss_ss * loss_ss
        loss.backward()
        optimizer.step()

        visualize_pred(torch.sigmoid(atts))
        visualize_gt(torch.sigmoid(gts))

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'Model' + '_%d' % epoch + '.pth')
