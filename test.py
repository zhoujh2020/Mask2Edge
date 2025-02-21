#!/user/bin/python
# -*- encoding: utf-8 -*-

import os
import torch
import torchvision
from PIL import Image
from os.path import join, isdir
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import cv2
import torch.nn.functional as F

def test(cfg, model, test_loader, save_dir):
    model.eval()
    dl = tqdm(test_loader)
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for image, pth in dl:
        dl.set_description("Single-scale test")
        image = image.cuda()
        _, _, H, W = image.shape
        filename = pth[0]
        with torch.no_grad():
            results = model(image)
        if cfg.side_edge:
            results_all = torch.zeros((len(results), 1, H, W))
            for i in range(len(results)):
                results_all[i, 0, :, :] = results[i]
            torchvision.utils.save_image((1 - results_all), join(save_dir, "%s.jpg" % filename))
        sio_save_dir = save_dir + '/mat/'
        if not isdir(sio_save_dir):
            os.makedirs(sio_save_dir)
        result = torch.squeeze(results[-1].detach()).cpu().numpy() 
        sio.savemat(sio_save_dir + filename + '.mat', {'predmap': result})
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(join(save_dir, "%s.png" % filename))
    
def multiscale_test(model, test_loader, save_dir):
    model.eval()
    dl = tqdm(test_loader)
    if not isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.4, 0.6, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.5]  # more big more small
    for image, pth in dl:
        dl.set_description("Multi-scale test")
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            result = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)
        ### rescale trick suggested by jiangjiang
        multi_fuse = (multi_fuse - multi_fuse.min()) / (multi_fuse.max() - multi_fuse.min())
        filename = pth[0]
        sio_save_dir = save_dir + '/mat/'
        if not isdir(sio_save_dir):
            os.makedirs(sio_save_dir)
        sio.savemat(sio_save_dir + filename + '.mat', {'predmap': multi_fuse})
        result_out_test = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_out_test.save(join(save_dir, "%s.png" % filename))
