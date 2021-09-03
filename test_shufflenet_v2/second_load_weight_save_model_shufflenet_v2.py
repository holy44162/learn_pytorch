from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import shufflenet_v2 # added by Holy 2109031500


if __name__ == "__main__":
    # hided by Holy 2109031500
    # model_ft = models.resnet18()
    model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True) # added by Holy 2109021500
    # model_ft = torch.hub.load('pytorch/vision:v0.9.1', 'shufflenet_v2_x0_5', pretrained=True) # added by Holy 2109030810
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    # Loading Model_ft Weights
    model_ft.load_state_dict(torch.load('model_ft_weights_shufflenet_v2.pth'))
    # model_ft.load_state_dict(torch.load('model_ft_weights_shufflenet_v2_x0_5.pth')) # added by Holy 2109030810
    # end of hide 2109031500

    # added by Holy 2109031500
    # num_classes = 1000
    # model_width = 0.5
    # model_ft = shufflenet_v2.Network(num_classes, model_width)
    # params = torch.load('model_ft_weights_shufflenet_v2_x0_5.pth', map_location=lambda storage, loc: storage)
    # model_ft.load_state_dict(params)

    # model_ft.num_classes = 2
    # end of addition 2109031500

    model_ft.eval()

    # Saving Model_ft with Shapes
    torch.save(model_ft, 'model_ft_shufflenet_v2.pth')
    # torch.save(model_ft, 'model_ft_shufflenet_v2_x0_5.pth') # added by Holy 2109030810

    """
    # Loading Model_conv Weights
    model_ft.load_state_dict(torch.load('model_conv_weights.pth'))
    # model_ft.eval()

    # Saving Model_conv with Shapes
    torch.save(model_ft, 'model_conv.pth')
    """