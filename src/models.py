# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:10:09 2024

@author: Hussain Ahmad Madni
"""

import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchsummary import summary
from torchvision.models import resnet50, Wide_ResNet50_2_Weights
import numpy as np
import torch.nn.functional as F

def load_wide_resnet_50(return_nodes:dict=None, verbose =False, size=(3,224,224)):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', Wide_ResNet50_2_Weights.IMAGENET1K_V1, force_reload=True, verbose=False) #weights=Wide_ResNet50_2_Weights.DEFAULT)
    #torch.hub.load('pytorch/vision:v0.14.1', 'wide_resnet50_2', weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1, force_reload=True, verbose=False)#weights=Wide_ResNet50_2_Weights.DEFAULT
    if not return_nodes is None:
        model = create_feature_extractor(model, return_nodes=return_nodes)
    if torch.cuda.is_available():
        model.cuda()
    if verbose:
        summary(model, size)
        for node in get_graph_node_names(model)[1]:
            if 'conv2' in node or True: 
                print(node)
    return model