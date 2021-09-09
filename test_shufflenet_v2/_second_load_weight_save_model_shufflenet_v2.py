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
import shutil # added by Holy 2109041002

# added by Holy 2109041002
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min
# end of addition 2109041002

if __name__ == "__main__":
    # hided by Holy 2109031500
    # model_ft = models.resnet18()
    # model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True) # added by Holy 2109021500
    model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=False) # added by Holy 2109060810
    # model_ft = torch.hub.load('pytorch/vision:v0.9.1', 'shufflenet_v2_x0_5', pretrained=True) # added by Holy 2109030810
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    # added by Holy 2109080810
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    # end of addition 2109080810

    # added by Holy 2109080810
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    checkpoint_path = './checkpoint/current_checkpoint.pt'
    best_model_path = './best_model/best_model.pt'
    
    # load the saved checkpoint
    model_ft, optimizer_ft, start_epochs, valid_loss_min = load_ckp(best_model_path, model_ft, optimizer_ft)
    # end of addition 2109080810

    # Loading Model_ft Weights
    # model_ft.load_state_dict(torch.load('best_model.pt')) # hided by Holy 2109080810
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