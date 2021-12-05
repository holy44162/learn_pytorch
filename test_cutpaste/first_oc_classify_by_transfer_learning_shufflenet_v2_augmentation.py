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

# added by Holy 2111121500
from torch.utils.data import Dataset
from pathlib import Path
import imutils
import random
import math

class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""
    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform
        
        if colorJitter is None:
            self.colorJitter = None
        else:
            # self.colorJitter = transforms.ColorJitter(brightness = colorJitter,
            #                                           contrast = colorJitter,
            #                                           saturation = colorJitter,
            #                                           hue = colorJitter)
            self.colorJitter = A.ColorJitter(brightness = colorJitter,
                                                      contrast = colorJitter,
                                                      saturation = colorJitter,
                                                      hue = colorJitter,
                                                      p = 1)
    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            # img = self.transform(img)
            # org_img = self.transform(org_img)
            # added by Holy 2111090810
            # img = numpy.array(img).astype('float32')
            img = self.transform(image=img)['image']
            # img = Image.fromarray(img)

            # org_img = numpy.array(org_img).astype('float32')
            org_img = self.transform(image=org_img)['image']
            # org_img = Image.fromarray(org_img)
            # end of addition 2111090810
        return org_img, img
    
class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.02,0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        # h = img.size[0]
        # w = img.size[1]
        h, w = img.shape[:2] # added by Holy 2111120810
        
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()
        
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))
        
        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        # box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        # patch = img.crop(box)
        patch = img[from_location_h:from_location_h + cut_h, from_location_w:from_location_w + cut_w] # roi = image[startY:endY, startX:endX] added by Holy 2111120810
        
        if self.colorJitter:
            # patch = self.colorJitter(patch)
            patch = self.colorJitter(image=patch)['image'] # added by Holy 2111120810
        
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        
        # insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        # augmented.paste(patch, insert_box)
        
        augmented[to_location_h:to_location_h + cut_h,to_location_w:to_location_w + cut_w] = patch # large_img[y_offset:y_end,x_offset:x_end] = small_img added by Holy 2111120810
        
        return super().__call__(img, augmented)

class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[2,16], height=[10,25], rotation=[-45,45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation
    
    def __call__(self, img):
        # h = img.size[0]
        # w = img.size[1]
        h, w = img.shape[:2] # added by Holy 2111120810
        
        # cut region
        # cut_w = random.uniform(*self.width)
        # cut_h = random.uniform(*self.height)

        # added by Holy 2111120810
        cut_w = int(random.uniform(*self.width))
        cut_h = int(random.uniform(*self.height))
        # end of addition 2111120810
        
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        # box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        # patch = img.crop(box)
        patch = img[from_location_h:from_location_h + cut_h, from_location_w:from_location_w + cut_w] # roi = image[startY:endY, startX:endX] added by Holy 2111120810
        
        if self.colorJitter:
            # patch = self.colorJitter(patch)
            patch = self.colorJitter(image=patch)['image'] # added by Holy 2111120810

        # rotate
        rot_deg = random.uniform(*self.rotation)
        # patch = Image.fromarray(patch) # added by Holy 2111120810
        # patch = patch.convert("RGBA").rotate(rot_deg,expand=True)

        # added by Holy 2111120810
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2RGBA)
        patch = imutils.rotate_bound(patch, -1*rot_deg)
        # end of addition 2111120810
        
        #paste
        # to_location_h = int(random.uniform(0, h - patch.size[0]))
        # to_location_w = int(random.uniform(0, w - patch.size[1]))

        # added by Holy 2111120810
        to_location_h = int(random.uniform(0, h - patch.shape[0]))
        to_location_w = int(random.uniform(0, w - patch.shape[1]))
        # end of addition 2111120810

        # mask = patch.split()[-1]
        # patch = patch.convert("RGB")
        
        # augmented = img.copy()
        # augmented = Image.fromarray(augmented) # added by Holy 2111120810
        # augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
        # augmented = numpy.array(augmented) # added by Holy 2111120810

        # added by Holy 2111120810
        mask = patch[:,:,-1]
        augmented = img.copy()
        for j in range(mask.shape[0]):
            for k in range(mask.shape[1]):
                if mask[j,k] == 255:
                    augmented[j+to_location_h,k+to_location_w,0] = patch[j,k,0]
                    augmented[j+to_location_h,k+to_location_w,1] = patch[j,k,1]
                    augmented[j+to_location_h,k+to_location_w,2] = patch[j,k,2]
        # end of addition 2111120810
        
        return super().__call__(img, augmented)
    
class CutPasteUnion(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)
    
    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)

class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)
    
    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)
        
        return org, cutpaste_normal, cutpaste_scar

class normal_mess_data(Dataset):
    """normal mess dataset."""

    def __init__(self, root_dir, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the dataset.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        
        # find test images
        if self.mode == "train":
            # self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.image_names = list((self.root_dir / "train" / "normal").glob("*.*")) # tested by Holy 2111020810
            # print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            #self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            # self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
            # self.imgs = Parallel(n_jobs=10)(delayed(read_and_resize)(file,size) for file in self.image_names) # added by Holy 2111120810
            # print(f"loaded {len(self.imgs)} images")
        else:
            #test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            self.image_names = list((self.root_dir / "val").glob(str(Path("*") / "*.*"))) # tested by Holy 2111020810
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            filename = self.image_names[idx]
            img = cv2.imread(filename.__str__())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                # img = self.transform(img)
                # added by Holy 2111090810
                # img = numpy.array(img)
                img = self.transform[0](image=img)['image']
                # img = Image.fromarray(img)
                img = self.transform[1](img)
                # end of addition 2111090810
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            # img = Image.open(filename)
            # img = img.resize((self.size,self.size)).convert("RGB")

            # added by Holy 2111120810
            img = cv2.imread(filename.__str__())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # end of addition 2111120810

            if self.transform is not None:
                # img = self.transform(img)
                img = self.transform(image=img)['image'] # added by Holy 2111120810
            return img, label != "normal"

class normal_mess_data_test(Dataset):
    """normal mess dataset."""

    def __init__(self, root_dir, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the dataset.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        
        # find test images
        if self.mode == "train":
            # self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.image_names = list((self.root_dir / "train" / "normal").glob("*.*")) # tested by Holy 2111020810
            # print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            #self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            # self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
            # self.imgs = Parallel(n_jobs=10)(delayed(read_and_resize)(file,size) for file in self.image_names) # added by Holy 2111120810
            # print(f"loaded {len(self.imgs)} images")
        else:
            #test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            self.image_names = list((self.root_dir / "val").glob(str(Path("*") / "*.*"))) # tested by Holy 2111020810
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            filename = self.image_names[idx]
            img = cv2.imread(filename.__str__())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                # img = self.transform(img)
                img = self.transform(image=img)['image'] # added by Holy 2111120810
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            # img = Image.open(filename)
            # img = img.resize((self.size,self.size)).convert("RGB")

            # added by Holy 2111120810
            img = cv2.imread(filename.__str__())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # end of addition 2111120810

            if self.transform is not None:
                # img = self.transform(img)
                img = self.transform(image=img)['image'] # added by Holy 2111120810
            return img, label != "normal"
# end of addition 2111121500

# added by Holy 2111161500
class normal_data(Dataset):
    """normal dataset."""

    def __init__(self, root_dir, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the dataset.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        
        # find test images
        if self.mode == "train":
            # self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.image_names = list((self.root_dir / "train" / "normal").glob("*.*")) # tested by Holy 2111020810
            # print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            #self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            # self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
            # self.imgs = Parallel(n_jobs=10)(delayed(read_and_resize)(file,size) for file in self.image_names) # added by Holy 2111120810
            # print(f"loaded {len(self.imgs)} images")
        else:
            #test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            # self.image_names = list((self.root_dir / "val").glob(str(Path("*") / "*.*"))) # tested by Holy 2111020810
            self.image_names = list((self.root_dir / "val" / "normal").glob("*.*"))
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):        
        filename = self.image_names[idx]
        img = cv2.imread(filename.__str__())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            # img = self.transform(img)
            # added by Holy 2111090810
            # img = numpy.array(img)
            img = self.transform[0](image=img)['image']
            # img = Image.fromarray(img)
            img = self.transform[1](img)
            # end of addition 2111090810
        return img        
# end of addition 2111161500

# added by Holy 2112011315
class normal_data_with_mess(Dataset):
    """normal dataset."""

    def __init__(self, root_dir, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the dataset.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        
        # find test images
        if self.mode == "train":
            # self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.image_names = list((self.root_dir / "train" / "normal").glob("*.*")) # added by Holy 2111020810
            self.image_names_mess = list((self.root_dir / "train" / "mess").glob("*.*")) # added by Holy 2112011315
            self.mess_length = len(self.image_names_mess) # added by Holy 2112011315
            # print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            #self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            # self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
            # self.imgs = Parallel(n_jobs=10)(delayed(read_and_resize)(file,size) for file in self.image_names) # added by Holy 2111120810
            # print(f"loaded {len(self.imgs)} images")
        else:
            #test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            # self.image_names = list((self.root_dir / "val").glob(str(Path("*") / "*.*"))) # tested by Holy 2111020810
            self.image_names = list((self.root_dir / "val" / "normal").glob("*.*"))
            self.image_names_mess = list((self.root_dir / "val" / "mess").glob("*.*")) # added by Holy 2112011315
            self.mess_length = len(self.image_names_mess) # added by Holy 2112011315
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):        
        filename = self.image_names[idx]
        img = cv2.imread(filename.__str__())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # added by Holy 2112011315
        filename_mess = self.image_names_mess[idx % self.mess_length]
        img_mess = cv2.imread(filename_mess.__str__())
        img_mess = cv2.cvtColor(img_mess, cv2.COLOR_BGR2RGB)
        # end of addition 2112011315

        if self.transform is not None:
            # img = self.transform(img)
            # added by Holy 2111090810
            # img = numpy.array(img)
            img = self.transform[0](image=img)['image']
            # img = Image.fromarray(img)
            img = self.transform[1](img)
            # end of addition 2111090810

            # added by Holy 2112011315
            img_mess = self.transform[0](image=img_mess)['image']
            img_mess = self.transform[2](image=img_mess)['image']
            # end of addition 2112011315
        img = (*img, img_mess) # added by Holy 2112011315
        return img
# end of addition 2112011315

# added by Holy 2111111500
class ProjectionNet_winding(nn.Module):
    def __init__(self, pretrained=True, head_layers=[512,512,512,512,512,512,512,512,128], num_classes=2):
        super(ProjectionNet_winding, self).__init__()
        #self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
        # self.resnet18 = resnet18(pretrained=pretrained)
        pytorch_vision_version = 'pytorch/vision:v0.10.0'
        shufflenet_version = 'shufflenet_v2_x1_0'
        self.shufflenet_v2 = torch.hub.load(pytorch_vision_version, shufflenet_version, pretrained=pretrained)

        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        # last_layer = 512
        last_layer = 1024
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons
        
        #the last layer without activation

        head = nn.Sequential(
            *sequential_layers
          )
        # self.resnet18.fc = nn.Identity()
        self.shufflenet_v2.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)
    
    def forward(self, x):
        # embeds = self.resnet18(x)
        embeds = self.shufflenet_v2(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits
    
    def freeze_shufflenet(self):
        # freeze full shufflenet
        # for param in self.resnet18.parameters():
        for param in self.shufflenet_v2.parameters():
            param.requires_grad = False
        
        #unfreeze head:
        # for param in self.resnet18.fc.parameters():
        for param in self.shufflenet_v2.fc.parameters():
            param.requires_grad = True
            
    def unfreeze(self):
        #unfreeze all:
        for param in self.parameters():
            param.requires_grad = True
# end of addition 2111111500

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
    break_num_threshold = 20 

    # added by Holy 2111301030
    for param_group in optimizer.param_groups:
        print('first learning rate: ', param_group['lr'])        
    # end of addition 2111301030

    # for epoch in range(num_epochs):
    for epoch in range(start_epochs, n_epochs+1):
        # added by Holy 2111151500
        if epoch-start_epochs == 20:
            model.unfreeze()
        # end of addition 2111151500

        batch_embeds = [] # added by Holy 2111161500

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
                # for inputs, labels in tepoch:
                for inputs in tepoch: # inputs(train): (3*128*3*224*224), inputs[0](val): (128*3*224*224), inputs[1](val): 128 added by Holy 2111130954
                    tepoch.set_description(f"Epoch {epoch}")

                    # added by Holy 2111151500
                    # if phase == 'train':
                    #     labels = torch.arange(len(inputs), device=device)
                    #     labels = labels.repeat_interleave(inputs[0].size(0))
                    #     inputs = torch.cat((inputs[0],inputs[1],inputs[2])).to(device)
                    # else:                        
                    #     labels = inputs[1].to(device)
                    #     inputs = inputs[0].to(device)
                    # end of addition 2111151500

                    # added by Holy 2111161500
                    labels = torch.arange(len(inputs), device=device)
                    labels = labels.repeat_interleave(inputs[0].size(0))
                    # inputs = torch.cat((inputs[0],inputs[1],inputs[2])).to(device)
                    inputs = torch.cat((inputs[0],inputs[1],inputs[2],inputs[3])).to(device) # added by Holy 2112011315
                    # end of addition 2111161500

                    # hided by Holy 2111151500
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)
                    # end of hide 2111151500

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # outputs = model(inputs)
                        # _, preds = torch.max(outputs, 1)
                        embeds, outputs = model(inputs) # added by Holy 2111151500
                        preds = torch.argmax(outputs,axis=1) # added by Holy 2111161500

                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                            # save embed for validation
                            batch_embeds.append(embeds.cpu().detach()) # added by Holy 2111161500

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
    BATCH_SIZE = 2**5
    EPOCHS = 100
    STEP_SIZE = 7
    GAMMA = 0.1
    
    # added by Holy 2111121500
    after_cutpaste_transform_A = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ToTensorV2()])
    train_transform_B2 = CutPaste3Way(transform = after_cutpaste_transform_A)
    # num_classes = 3
    num_classes = 4 # added by Holy 2112011315
    # end of addition 2111121500

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
                # A.RandomGridShuffle(), # hided by Holy 2112011315
                A.ShiftScaleRotate(),
                A.VerticalFlip(),
            ], p=1)
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
                # A.RandomGridShuffle(), # hided by Holy 2112011315
                A.ShiftScaleRotate(),
                A.VerticalFlip(),
            ], p=1)
        ])
    }
    
    # data_dir = 'e:/dnn_data/ZTC950V763_data'
    # data_dir = 'd:/data_seq/dnn_data/ZTC950V763_data'
    data_dir = 'd:/data_seq/dnn_data/w6013_data'
    
    # image_datasets = {x: NormalVsMessFolder(os.path.join(data_dir, x),
    #                                           data_transforms[x])
    #                   for x in ['train', 'val']} 
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
    #                                               shuffle=True, num_workers=4)
    #                for x in ['train', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # class_names = image_datasets['train'].classes

    # added by Holy 2111121500
    # train_data = normal_mess_data(data_dir, transform = [data_transforms['train'], train_transform_B2])
    # # 4640*3*3*224*224
    # test_data = normal_mess_data_test(data_dir, transform = data_transforms['val'], mode="test")
    # # 2593*2: part 1: 3*224*224, part 2: bool
    # image_datasets = {
    #     'train': train_data,
    #     'val': test_data}
    
    # added by Holy 2111161500
    # image_datasets = {x: normal_data(data_dir, transform = [data_transforms[x], train_transform_B2], mode=x)
    #                   for x in ['train', 'val']}
    # end of addition 2111161500

    # added by Holy 2112011315
    image_datasets = {x: normal_data_with_mess(data_dir, transform = [data_transforms[x], train_transform_B2, after_cutpaste_transform_A], mode=x)
                      for x in ['train', 'val']}
    # end of addition 2112011315

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    dataset_sizes = {x: num_classes*len(image_datasets[x]) for x in ['train', 'val']} # added by Holy 2111161500

    # tested by Holy 2111130954
    # for i_batch, sample_batched in enumerate(dataloaders['train']):
    #     print('i_batch: ', i_batch)
    #     print('sample_batched length: (3)', len(sample_batched))
    #     print(f'sample_batched[0] size: ({BATCH_SIZE}x3x224x224)', sample_batched[0].size())
    #     print(f'sample_batched[1] size: ({BATCH_SIZE}x3x224x224)', sample_batched[1].size())
    #     print(f'sample_batched[2] size: ({BATCH_SIZE}x3x224x224)', sample_batched[2].size())
    #     break
    # end of test 2111130954
    # end of addition 2111121500

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('using device: ', device) # added by Holy 2111291030
    
    # hided by Holy 2111130954
    # pytorch_vision_version = 'pytorch/vision:v0.10.0'
    # shufflenet_version = 'shufflenet_v2_x1_0'
    # model_ft = torch.hub.load(pytorch_vision_version, shufflenet_version, pretrained=True) 
    
    # num_ftrs = model_ft.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    # hided by Holy 2111130954

    # added by Holy 2111130954
    head_layer = 2
    head_layers = [512]*head_layer+[128]
    model_ft = ProjectionNet_winding(pretrained=True, head_layers=head_layers, num_classes=num_classes)
    # end of addition 2111130954

    model_ft = model_ft.to(device)

    model_ft.freeze_shufflenet() # added by Holy 2111130954

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=INIT_LR)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)
        
    start_epochs = 1
    end_epochs = start_epochs + EPOCHS
    checkpoint_path = './checkpoint/current_checkpoint.pt'
    best_model_path = './best_model/best_model.pt'

    # added by Holy 2111161500
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.mkdir(os.path.dirname(checkpoint_path))
    if not os.path.exists(os.path.dirname(best_model_path)):
        os.mkdir(os.path.dirname(best_model_path))
    # end of addition 2111161500

    resume_training = False
    if resume_training:        
        # model_ft = torch.hub.load(pytorch_vision_version, shufflenet_version, pretrained=False)
        
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, 2)

        # added by Holy 2111170810
        head_layer = 2
        head_layers = [512]*head_layer+[128]
        model_ft = ProjectionNet_winding(pretrained=False, head_layers=head_layers, num_classes=num_classes)
        # end of addition 2111170810
        
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
        latest_best_model_path = 'latest_best_model.pth'
        if os.path.exists(latest_best_model_path):
            model_ft = torch.load(latest_best_model_path)
            print("load model: ", latest_best_model_path)
        # end of addition 2110280810

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
