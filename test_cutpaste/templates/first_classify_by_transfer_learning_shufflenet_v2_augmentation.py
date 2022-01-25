from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torchvision.datasets.folder import *
import glob


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


def train_model(model, criterion, optimizer, scheduler, start_epochs, n_epochs, valid_loss_min_input=None, checkpoint_path=None, best_model_path=None):    
    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }    

    since = time.time()
    
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input    

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    break_num = 0 
    break_num_threshold = 30 

    # added by Holy 2111301030
    for param_group in optimizer.param_groups:
        print('first learning rate: ', param_group['lr'])        
    # end of addition 2111301030
    
    # for epoch in range(num_epochs):
    for epoch in range(start_epochs, n_epochs+1): 
        # initialize variables to monitor training and validation loss
        valid_loss = 0.0        
        train_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0        

        print('Epoch {}/{}'.format(epoch, n_epochs)) 
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                # for inputs, labels in dataloaders[phase]:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}") 

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)                        

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    accuracy = torch.mean((preds == labels.data).float()).item()
                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                    
                if phase == 'train':
                    scheduler.step()
                    print('current learning rate: ', scheduler.get_last_lr()) # added by Holy 2111301030

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())                
                
                if phase == 'val':
                    valid_loss = epoch_loss
                    valid_acc = epoch_acc
                
                if phase == 'train':
                    train_loss = epoch_loss
                    train_acc = epoch_acc                

        print()
        
        # update our training history
        H["train_loss"].append(train_loss)
        H["train_acc"].append(train_acc.cpu().detach().numpy())
        H["val_loss"].append(valid_loss)
        H["val_acc"].append(valid_acc.cpu().detach().numpy())
        
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

            break_num = 0        
        else:
            break_num += 1
            if break_num > break_num_threshold:
                break        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plot.png')    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class NormalVsMessFolder(DatasetFolder):
    def __init__(self,root: str,transform = None,target_transform = None,loader = default_loader,is_valid_file= None):
        super(NormalVsMessFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # sample = self.loader(path)
        sample = cv2.imread(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


if __name__ == "__main__":
    INIT_LR = 1e-3
    BATCH_SIZE = 2**7    
    EPOCHS = 100
    STEP_SIZE = 7
    GAMMA = 0.1
    
    data_transforms = {
        'train': A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.HorizontalFlip(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(),
                A.Emboss(),
                A.Equalize(),
                A.FancyPCA(),
                A.GaussNoise(),
                A.GaussianBlur(),
                A.HueSaturationValue(),
                A.ISONoise(),
                A.ImageCompression(),
                A.MedianBlur(),
                A.MotionBlur(),
                A.MultiplicativeNoise(),
                A.Posterize(),
                A.RGBShift(),
                A.RandomBrightnessContrast(),
                A.RandomFog(),
                A.RandomGamma(),
                A.RandomRain(),
                A.RandomShadow(),
                A.RandomSnow(),
                A.RandomSunFlare(),
                A.RandomToneCurve(),
                A.Sharpen(),
                A.CoarseDropout(),
                A.Flip(),
                A.Perspective(),
                A.RandomGridShuffle(),
                A.ShiftScaleRotate(),
                A.VerticalFlip(),
                ], p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(224, 224),
            A.OneOf([            
                A.HorizontalFlip(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(),
                A.Emboss(),
                A.Equalize(),
                A.FancyPCA(),
                A.GaussNoise(),
                A.GaussianBlur(),
                A.HueSaturationValue(),
                A.ISONoise(),
                A.ImageCompression(),
                A.MedianBlur(),
                A.MotionBlur(),
                A.MultiplicativeNoise(),
                A.Posterize(),
                A.RGBShift(),
                A.RandomBrightnessContrast(),
                A.RandomFog(),
                A.RandomGamma(),
                A.RandomRain(),
                A.RandomShadow(),
                A.RandomSnow(),
                A.RandomSunFlare(),
                A.RandomToneCurve(),
                A.Sharpen(),
                A.CoarseDropout(),
                A.Flip(),
                A.Perspective(),
                A.RandomGridShuffle(),
                A.ShiftScaleRotate(),
                A.VerticalFlip(),
                ], p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),        
        ])
    } 
    data_dir = 'e:/dnn_data/z75_data' 
    
    image_datasets = {x: NormalVsMessFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']} 
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('using device: ', device) # added by Holy 2111291030
    
    pytorch_vision_version = 'pytorch/vision:v0.10.0'
    shufflenet_version = 'shufflenet_v2_x1_0'
    model_ft = torch.hub.load(pytorch_vision_version, shufflenet_version, pretrained=True) 
    
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)        

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=INIT_LR)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)
        
    start_epochs = 1
    end_epochs = start_epochs + EPOCHS
    checkpoint_path = './checkpoint/current_checkpoint.pt'
    best_model_path = './best_model/best_model.pt'

    # added by Holy 2111030810
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.mkdir(os.path.dirname(checkpoint_path))
    if not os.path.exists(os.path.dirname(best_model_path)):
        os.mkdir(os.path.dirname(best_model_path))
    # end of addition 2111030810

    resume_training = False
    if resume_training:        
        model_ft = torch.hub.load(pytorch_vision_version, shufflenet_version, pretrained=False)
        
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        
        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # optimizer_ft = optim.Adam(model_ft.parameters(), lr=INIT_LR) 

        # # Decay LR by a factor of 0.1 every 7 epochs
        # exp_lr_scheduler = lr_scheduler.StepLR(
        #     optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)

        ckp_path = os.path.dirname(checkpoint_path)
        file_type = '\*pt'
        pt_files = glob.glob(ckp_path + file_type)
        checkpoint_path_latest = max(pt_files, key=os.path.getmtime)        

        # load the saved checkpoint
        model_ft, optimizer_ft, start_epochs, valid_loss_min = load_ckp(checkpoint_path_latest, model_ft, optimizer_ft)
        end_epochs = start_epochs + EPOCHS
        print("model = ", model_ft)
        print("optimizer = ", optimizer_ft)
        print("start_epoch = ", start_epochs)
        print("valid_loss_min = ", valid_loss_min)
        print("valid_loss_min = {:.6f}".format(valid_loss_min))
        print(checkpoint_path_latest)

        # added by Holy 2110280810
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=INIT_LR) 

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)
        # end of addition 2110280810

        # valid_loss_min = np.Inf # added by Holy 2110070936

        # added by Holy 2110280810
        latest_best_model_path = 'latest_best_shufflenet_model.pth'
        if os.path.exists(latest_best_model_path):
            model_ft = torch.load(latest_best_model_path)
            print("load model: ", latest_best_model_path)
        # end of addition 2110280810
        
        # added by Holy 2111300810
        fine_tune = False
        if fine_tune:
            base_model_path = './base_model/base_shufflenet_model.pth'
            if os.path.exists(base_model_path):
                model_ft = torch.load(base_model_path)
                valid_loss_min = np.Inf
                print("load base model: ", base_model_path)
        # end of addition 2111300810

        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           start_epochs, end_epochs, valid_loss_min, checkpoint_path, best_model_path)
    else:
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           start_epochs, end_epochs, np.Inf, checkpoint_path, best_model_path)    
    
    # save model_ft
    torch.save(model_ft.state_dict(), 'model_ft_weights_shufflenet_v2.pth')
        
    model_ft.eval()

    # Saving Model_ft with Shapes
    torch.save(model_ft, 'model_ft_shufflenet_v2.pth')
