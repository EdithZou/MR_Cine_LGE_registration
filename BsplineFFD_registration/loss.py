# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:02:39 2023

@author: SSKJ-TX
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# import scipy.ndimage
# import numpy as np



class Dice():
    """
    N-D dice for segmentation
    """
    def __init__(self):
        self.y_true_onehot_label3d = {}

    def dice(self, y_true, y_pred):
        # 单标签单value
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true.sum(dim=vol_axes) + y_pred.sum(dim=vol_axes)), min=1e-5) # bool
        # bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        # dice = torch.mean(top / bottom) # 鼓励大区域
        # dice = top / bottom # 区域一致
        dice = top.numpy() / bottom.numpy()
        return dice
    
    def loss_multi_value_in_onelabel3d(self, y_pred , y_true, labelorders):
        # 单标签文件中多value的情况
        num_classes = len(labelorders)
        dicescore = 0
        for order in labelorders:
            order = int(order)
            # k = str(order)
            # if k not in self.y_true_onehot_label3d:
                # self.y_true_onehot_label3d[k] = y_true == order
            y_true_onehot = y_true == order
            y_pred_onehot = y_pred == order
            dicescore += self.dice(y_true_onehot, y_pred_onehot)
            
        dicescore = dicescore / num_classes
        return dicescore#.to('cuda')
    
    
    
    
    
    
    
    