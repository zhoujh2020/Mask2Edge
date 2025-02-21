#!/user/bin/python
# -*- encoding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F

def tracingloss(prediction, label, balanced_w=1.1):
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask==1).float()).float()
        num_negative = torch.sum((mask==0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0

    #print('bce')
    cost = torch.sum(torch.nn.functional.binary_cross_entropy(
                prediction.float(),label.float(), weight=mask, reduce=False))
    
    return cost
  
  
  
  
    
