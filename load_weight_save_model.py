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


if __name__ == "__main__":
    model_ft = models.resnet18()
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    # Loading Model_ft Weights
    model_ft.load_state_dict(torch.load('model_ft_weights.pth'))
    model_ft.eval()

    # Saving Model_ft with Shapes
    torch.save(model_ft, 'model_ft.pth')

    # Loading Model_conv Weights
    model_ft.load_state_dict(torch.load('model_conv_weights.pth'))
    # model_ft.eval()

    # Saving Model_conv with Shapes
    torch.save(model_ft, 'model_conv.pth')