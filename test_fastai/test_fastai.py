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
# matplotlib.use("Agg") # hided by Holy 2112210810
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torchvision.datasets.folder import *
import glob

# added by Holy 2112210810
from fastai.vision.all import *
from fastai.data.all import *
from pathlib import Path
from torch.distributions.beta import Beta
from torchvision.models import shufflenet_v2_x1_0
from torchvision import transforms
from PIL import Image
from fastai.vision.learner import _default_meta, _resnet_meta
from fastai.vision.models.unet import _get_sz_change_idxs, UnetBlock, ResizeToOrig
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from fastai.data.core import DataLoaders
from fastai.optimizer import OptimWrapper
from functools import partial
from fastai.callback.schedule import Learner # To get `fit_one_cycle`, `lr_find`
from fastai.metrics import accuracy
import torch.nn.functional as F
import torch.onnx as onnx
import onnx as onnx_origin
from onnxsim import simplify
import onnxruntime as rt
# end of addition 2112210810


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

    arch = model_ft # added by Holy 2112271500
    
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

    # added by Holy 2112200810
    ###
    path = untar_data(URLs.PETS)
    files = get_image_files(path/"images")
    print('images number: ', len(files))

    def label_func(f): return f[0].isupper()

    # dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224), num_workers=0)

    dls = ImageDataLoaders.from_name_func(path,
                                          files,
                                          label_func,
                                          item_tfms=Resize(224),
                                          pin_memory=True,
                                          num_workers=0)

    learn = cnn_learner(dls, resnet34, metrics=error_rate)

    learn.fit_one_cycle(1, 1e-3)

    print('dls device: ', dls.device)

    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)
    # print('lrs type: ', type(lrs))

    # learn.fine_tune(1, lrs.valley)
    ###

    ###
    # pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
    #                  get_items=get_image_files,
    #                  splitter=RandomSplitter(),
    #                  get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
    #                  item_tfms=Resize(460),
    #                  batch_tfms=aug_transforms(size=224))
    
    # dls = pets.dataloaders(untar_data(URLs.PETS)/"images", num_workers=0)

    # learn = cnn_learner(dls, resnet34, metrics=error_rate)

    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)
    # print('lrs type: ', type(lrs))

    # learn.fine_tune(1, lrs.valley)
    ###

    ###
    # path = untar_data(URLs.PETS)

    # fnames = get_image_files(path/"images")

    # def label_func(fname):
    #     return "cat" if fname.name[0].isupper() else "dog"
    
    # dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
    #                    get_items=get_image_files,
    #                    get_y=label_func,
    #                    splitter=RandomSplitter(),
    #                    item_tfms=Resize(224))
    
    # dls = dblock.dataloaders(path/"images", num_workers=0)

    # dblock.summary(path/"images")

    # # dls.show_batch()
    # # plt.show()

    # learn = cnn_learner(dls, resnet34, metrics=error_rate)

    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)    

    # learn.fine_tune(1, lrs.valley)
    ###

    ###
    # path = Path('.')
    # path = untar_data(URLs.IMAGENETTE_160)

    # dls = ImageDataLoaders.from_folder(path, valid='val',
    #                                    item_tfms=RandomResizedCrop(128, min_scale=0.35), batch_tfms=Normalize.from_stats(*imagenet_stats), 
    #                                    num_workers=0)
    # # dls.show_batch()
    # # plt.show()

    # learn = cnn_learner(dls, resnet34, metrics=error_rate)

    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)    

    # learn.fine_tune(1, lrs.valley)
    ###

    ###
    # path = untar_data(URLs.IMAGENETTE_160)

    # lbl_dict = dict(
    #     n01440764='tench',
    #     n02102040='English springer',
    #     n02979186='cassette player',
    #     n03000684='chain saw',
    #     n03028079='church',
    #     n03394916='French horn',
    #     n03417042='garbage truck',
    #     n03425413='gas pump',
    #     n03445777='golf ball',
    #     n03888257='parachute'
    # )

    # def label_func(fname):
    #     return lbl_dict[parent_label(fname)]
    
    # # dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
    # #                    get_items=get_image_files,
    # #                    get_y=label_func,
    # #                    splitter=GrandparentSplitter(),
    # #                    item_tfms=RandomResizedCrop(128, min_scale=0.35),
    # #                    batch_tfms=Normalize.from_stats(*imagenet_stats))
    
    # # dls = dblock.dataloaders(path, num_workers=0)

    # imagenette = DataBlock(blocks=(ImageBlock, CategoryBlock),
    #                        get_items=get_image_files,
    #                        get_y=Pipeline(
    #                            [parent_label, lbl_dict.__getitem__]),
    #                        splitter=GrandparentSplitter(valid_name='val'),
    #                        item_tfms=RandomResizedCrop(128, min_scale=0.35),
    #                        batch_tfms=Normalize.from_stats(*imagenet_stats))

    # dls = imagenette.dataloaders(path, num_workers=0)

    # # dls.show_batch()
    # # plt.show()

    # learn = cnn_learner(dls, resnet34, metrics=error_rate)

    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)    

    # learn.fine_tune(1, lrs.valley)
    ###

    ###
    # source = untar_data(URLs.IMAGENETTE_160)
    # fnames = get_image_files(source)

    # lbl_dict = dict(
    #     n01440764='tench',
    #     n02102040='English springer',
    #     n02979186='cassette player',
    #     n03000684='chain saw',
    #     n03028079='church',
    #     n03394916='French horn',
    #     n03417042='garbage truck',
    #     n03425413='gas pump',
    #     n03445777='golf ball',
    #     n03888257='parachute'
    # )

    # tfm = Pipeline([parent_label, lbl_dict.__getitem__, Categorize(vocab = lbl_dict.values())])

    # splits = GrandparentSplitter(valid_name='val')(fnames)

    # dsets = Datasets(fnames, [[PILImage.create], [parent_label, lbl_dict.__getitem__, Categorize]], splits=splits)

    # print('dsets[0]: ', dsets[0])
    # dsets.show(dsets[0])
    # # plt.show()

    # item_tfms = [ToTensor, RandomResizedCrop(128, min_scale=0.35)]
    # batch_tfms = [IntToFloatTensor, Normalize.from_stats(*imagenet_stats)]

    # dls = dsets.dataloaders(after_item=item_tfms, after_batch=batch_tfms, bs=64, num_workers=0)
    # dls.show_batch()
    # # plt.show()

    # class LabelSmoothingCE(Module):
    #     def __init__(self, eps=0.1, reduction='mean'): self.eps,self.reduction = eps,reduction

    #     def forward(self, output, target):
    #         c = output.size()[-1]
    #         log_preds = F.log_softmax(output, dim=-1)
    #         if self.reduction=='sum': loss = -log_preds.sum()
    #         else:
    #             loss = -log_preds.sum(dim=-1) #We divide by that size at the return line so sum and not mean
    #             if self.reduction=='mean':  loss = loss.mean()
    #         return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target.long(), reduction=self.reduction)

    #     def activation(self, out): return F.softmax(out, dim=-1)
    #     def decodes(self, out):    return out.argmax(dim=-1)

    # # learn = cnn_learner(dls, resnet34, metrics=accuracy, pretrained=False)
    # # learn = Learner(dls, xresnet34(n_out=10), metrics=accuracy)
    # # learn = Learner(dls, xresnet34(n_out=10), loss_func=LabelSmoothingCE(), metrics=accuracy)

    # # pytorch_adamw = partial(OptimWrapper, opt=torch.optim.AdamW)
    # # learn = Learner(dls, xresnet18(), lr=1e-2, metrics=accuracy,
    # #                 loss_func=LabelSmoothingCrossEntropy(),
    # #                 opt_func=partial(pytorch_adamw, weight_decay=0.01, eps=1e-3))

    # pytorch_adamw = partial(OptimWrapper, opt=torch.optim.AdamW)
    # class Mixup(Callback):
    #     run_valid = False
        
    #     def __init__(self, alpha=0.4): self.distrib = Beta(tensor(alpha), tensor(alpha))
        
    #     def before_batch(self):
    #         self.t = self.distrib.sample((self.y.size(0),)).squeeze().to(self.x.device)
    #         shuffle = torch.randperm(self.y.size(0)).to(self.x.device)
    #         x1,self.y1 = self.x[shuffle],self.y[shuffle]
    #         self.learn.xb = (x1 * (1-self.t[:,None,None,None]) + self.x * self.t[:,None,None,None],)
        
    #     def after_loss(self):
    #         with NoneReduce(self.loss_func) as lf:
    #             loss = lf(self.pred,self.y1) * (1-self.t) + lf(self.pred,self.y) * self.t
    #         self.learn.loss = loss.mean()
    
    # learn = Learner(dls, xresnet18(), lr=1e-2, metrics=accuracy,
    #                 loss_func=LabelSmoothingCrossEntropy(), cbs=Mixup(),
    #                 opt_func=partial(pytorch_adamw, weight_decay=0.01, eps=1e-3))
    
    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)

    # learn.fit_one_cycle(5, lrs.valley)
    # learn.show_results()
    # plt.show()
    ###

    ###
    # path = Path(data_dir)
    
    # def label_func(fname):
    #     return parent_label(fname)
    
    # dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
    #                    get_items=get_image_files,
    #                    get_y=label_func,
    #                    splitter=GrandparentSplitter(valid_name='val'),
    #                    item_tfms=RandomResizedCrop(128, min_scale=0.35),
    #                    batch_tfms=Normalize.from_stats(*imagenet_stats))
    
    # dls = dblock.dataloaders(path, num_workers=0)
    
    # dls.show_batch()

    # learn = cnn_learner(dls, shufflenet_v2_x1_0, metrics=accuracy)

    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)    

    # learn.fine_tune(1, lrs.valley)
    # # learn.fit_one_cycle(2, lrs.valley)
    # learn.show_results()
    # plt.show()
    ###

    ###
    # source = untar_data(URLs.PETS)
    # items = get_image_files(source/"images")

    # img = PILImage.create(items[0])
    # img.show()

    # aug = A.ShiftScaleRotate(p=1)
    # def aug_tfm(img): 
    #     np_img = np.array(img)
    #     aug_img = aug(image=np_img)['image']
    #     return PILImage.create(aug_img)

    # # aug_tfm(img).show()

    # class AlbumentationsTransform(Transform):
    #     def __init__(self, aug): self.aug = aug
    #     def encodes(self, img: PILImage):
    #         aug_img = self.aug(image=np.array(img))['image']
    #         return PILImage.create(aug_img)
    
    # tfm = AlbumentationsTransform(A.ShiftScaleRotate(p=1))
    # a,b = tfm((img, 'dog'))
    # show_image(a, title=b)

    # class AlbumentationsTransform(DisplayedTransform):
    #     split_idx,order=0,2
    #     def __init__(self, train_aug): store_attr()
        
    #     def encodes(self, img: PILImage):
    #         aug_img = self.train_aug(image=np.array(img))['image']
    #         return PILImage.create(aug_img)

    # def get_train_aug(): return A.Compose([
    #     A.HueSaturationValue(
    #         hue_shift_limit=0.2,
    #         sat_shift_limit=0.2,
    #         val_shift_limit=0.2,
    #         p=0.5
    #     ),
    #     A.CoarseDropout(p=0.5),
    #     A.Cutout(p=0.5)
    # ])

    # item_tfms = [Resize(224), AlbumentationsTransform(get_train_aug())]

    # path = untar_data(URLs.PETS)/'images'

    # def is_cat(x): return x[0].isupper()
    # dls = ImageDataLoaders.from_name_func(
    #     path, get_image_files(path), valid_pct=0.2, seed=42,
    #     label_func=is_cat, item_tfms=item_tfms)
    
    # dls.train.show_batch(max_n=4)
    # dls.valid.show_batch(max_n=4)

    # class AlbumentationsTransform(RandTransform):
    #     "A transform handler for multiple `Albumentation` transforms"
    #     split_idx,order=None,2
    #     def __init__(self, train_aug, valid_aug): store_attr()
        
    #     def before_call(self, b, split_idx):
    #         self.idx = split_idx
        
    #     def encodes(self, img: PILImage):
    #         if self.idx == 0:
    #             aug_img = self.train_aug(image=np.array(img))['image']
    #         else:
    #             aug_img = self.valid_aug(image=np.array(img))['image']
    #         return PILImage.create(aug_img)

    # def get_train_aug(): return A.Compose([
    #     A.RandomResizedCrop(224, 224),
    #     A.Transpose(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.ShiftScaleRotate(p=0.5),
    #     A.HueSaturationValue(
    #         hue_shift_limit=0.2,
    #         sat_shift_limit=0.2,
    #         val_shift_limit=0.2,
    #         p=0.5),
    #     A.CoarseDropout(p=0.5),
    #     A.Cutout(p=0.5)
    # ])

    # def get_valid_aug(): return A.Compose([
    #     A.CenterCrop(224, 224, p=1.),
    #     A.Resize(224, 224)
    # ], p=1.)

    # item_tfms = [Resize(256), AlbumentationsTransform(get_train_aug(), get_valid_aug())]

    # dls = ImageDataLoaders.from_name_func(
    #     path, get_image_files(path), valid_pct=0.2, seed=42,
    #     label_func=is_cat, item_tfms=item_tfms)

    # dls.train.show_batch(max_n=4)
    # dls.valid.show_batch(max_n=4)

    # x,_ = dls.valid.one_batch()
    # print(x.shape)
    # plt.show()
    ###

    ###
    # path = Path(data_dir)

    data_transforms_fastai = {
        'train': A.Compose([            
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
                A.Resize(224, 224)
        ]),
        'val': A.Compose([
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
                A.Resize(224, 224)
        ])
    }
    
    # def label_func(fname):
    #     return parent_label(fname)
    
    # class AlbumentationsTransform(RandTransform):
    #     "A transform handler for multiple `Albumentation` transforms"
    #     split_idx,order=None,2
    #     def __init__(self, train_aug, valid_aug): store_attr()
        
    #     def before_call(self, b, split_idx):
    #         self.idx = split_idx
        
    #     def encodes(self, img: PILImage):
    #         if self.idx == 0:
    #             aug_img = self.train_aug(image=np.array(img))['image']
    #         else:
    #             aug_img = self.valid_aug(image=np.array(img))['image']
    #         return PILImage.create(aug_img)

    # def get_train_aug(): return data_transforms_fastai['train']

    # def get_valid_aug(): return data_transforms_fastai['val']

    # def get_train_aug_test(): return A.Resize(224, 224)

    # def get_valid_aug_test(): return A.Resize(224, 224)

    # def set_seed_fastai(dls,x=42):
    #     random.seed(x)
    #     dls.rng.seed(x)
    #     np.random.seed(x)
    #     torch.manual_seed(x)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)

    # # item_tfms = AlbumentationsTransform(get_train_aug(), get_valid_aug())
    # item_tfms = AlbumentationsTransform(get_train_aug_test(), get_valid_aug_test())
    
    # dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
    #                    get_items=get_image_files,
    #                    get_y=label_func,
    #                    splitter=GrandparentSplitter(valid_name='val'),
    #                    item_tfms=item_tfms,
    #                    batch_tfms=Normalize.from_stats(*imagenet_stats))
    
    # dls = dblock.dataloaders(path, num_workers=0)

    # set_seed_fastai(dls)
    
    # dls.train.show_batch(max_n=4)
    # dls.valid.show_batch(max_n=4)

    # # set_seed(42, reproducible=True)

    # learn = cnn_learner(dls, shufflenet_v2_x1_0, metrics=accuracy)

    # keep_path = learn.path
    # print('keep_path first: ', keep_path)

    # learn.path = Path('./best_model_fastai')
    # model_name = 'best_model_fastai'

    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)    

    # # learn.fine_tune(1, lrs.valley)
    # learn.fine_tune(0,
    #                 lrs.valley,
    #                 cbs=[EarlyStoppingCallback(monitor='accuracy', min_delta=0.01, patience=3),
    #                      SaveModelCallback(monitor='accuracy', min_delta=0.01, fname=model_name)])
    # # reset the model path
    # learn.path = keep_path

    # learn.export()

    # # # learn.fit_one_cycle(2, lrs.valley)
    # learn.show_results()
    # plt.show()

    # # def plot_top_losses_fix(interp, k, largest=True, **kwargs):
    # #     losses,idx = interp.top_losses(k, largest)
    # #     if not isinstance(interp.inputs, tuple): interp.inputs = (interp.inputs,)
    # #     if isinstance(interp.inputs[0], Tensor): inps = tuple(o[idx] for o in interp.inputs)
    # #     else: inps = interp.dl.create_batch(interp.dl.before_batch([tuple(o[i] for o in interp.inputs) for i in idx]))
    # #     b = inps + tuple(o[idx] for o in (interp.targs if is_listy(interp.targs) else (interp.targs,)))
    # #     x,y,its = interp.dl._pre_show_batch(b, max_n=k)
    # #     b_out = inps + tuple(o[idx] for o in (interp.decoded if is_listy(interp.decoded) else (interp.decoded,)))
    # #     x1,y1,outs = interp.dl._pre_show_batch(b_out, max_n=k)
    # #     if its is not None:
    # #         #plot_top_losses(x, y, its, outs.itemgot(slice(len(inps), None)), L(self.preds).itemgot(idx), losses,  **kwargs)
    # #         plot_top_losses(x, y, its, outs.itemgot(slice(len(inps), None)), interp.preds[idx], losses,  **kwargs)
    # #     #TODO: figure out if this is needed
    # #     #its None means that a batch knows how to show itself as a whole, so we pass x, x1
    # #     #else: show_results(x, x1, its, ctxs=ctxs, max_n=max_n, **kwargs)

    # # interp = Interpretation.from_learner(learn)
    # # # interp.plot_top_losses(9, figsize=(15,10))
    
    # # plot_top_losses_fix(interp, 9, nrows=3)

    # # plt.show()

    # test_img_pathname = 'd:/data_seq/gongqiWinding/Z75_DF-4105H-BD/211129/shrinkVideo/bigDatasets/test/imgs/img00022.jpg'

    # predict_result = learn.predict(test_img_pathname)
    # print('01: learn predict_result: ', predict_result)

    # learn.path = Path('./best_model_fastai')
    
    # learn.load(model_name, with_opt=False)

    # learn.path = keep_path

    # learn_inf = load_learner('export.pkl')
    
    # # predict_result = learn.predict(test_img_pathname)
    # predict_result_inf = learn_inf.predict(test_img_pathname)
    # print('02: learn_inf predict_result: ', predict_result_inf)

    # predict_result = learn.predict(test_img_pathname)
    # print('03: learn predict_result: ', predict_result)

    # IMAGE_SIZE = dls.one_batch()[0].shape[-2:] #dls is the dataloader you used for training
    # N_CLASSES = get_c(dls)
    # print('IMAGE_SIZE: ', IMAGE_SIZE, ' N_CLASSES: ', N_CLASSES)

    # pred_fastai = learn_inf.predict(test_img_pathname)
    # print('04: pred_fastai: ', pred_fastai)

    # predict_result_inf = learn_inf.predict(test_img_pathname)
    # print('05: learn_inf predict_result: ', predict_result_inf)

    # torch.save(learn_inf.model.state_dict(), "fasti_weights.pth")

    # model_arch1 = shufflenet_v2_x1_0(num_classes=2)
    # model_arch = learn_inf.model
    # checkpoint = torch.load('./fasti_weights.pth')
    # checkpoint1 = torch.load('./checkpoint/current_checkpoint_2112201717_w6013.pt')
    # checkpoint2 = torch.load('./best_model_fastai/models/best_model_fastai.pth')
    # model_arch.load_state_dict(checkpoint)
    # model_arch.eval()

    # # specify ImageNet mean and standard deviation
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]

    # imagenet_mean_tensor = torch.FloatTensor(imagenet_mean)
    # imagenet_std_tensor = torch.FloatTensor(imagenet_std)

    # transform_norm1 = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(imagenet_mean_tensor, imagenet_std_tensor)
    # ])

    # transform_norm = transforms.Normalize(imagenet_mean_tensor, imagenet_std_tensor)

    # img = cv2.imread(test_img_pathname)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # img = cv2.resize(img, (224,224))       

    # img = img.astype("float32") / 255.0
    # img -= imagenet_mean
    # img /= imagenet_std
    # img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    # img = torch.from_numpy(img)
    # # img = transform_norm(img)
    # # img -= imagenet_mean_tensor
    # # img /= imagenet_std_tensor
    # img = img.unsqueeze(0)
    # # img = img.to(device) 
    # output = model_arch(img)
    
    # val, cls = torch.max(output.data, 1)
    
    # # print("[pytorch]--->predicted class:", cls.item())
    # # print("[pytorch]--->predicted value:", val.item())
    # print("[pytorch]--->output.data:", output.data)

    # image = Image.open(test_img_pathname).convert("RGB")

    # image_tfm = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
    #                              0.229, 0.224, 0.225]),
    #     ]
    # )

    # x = image_tfm(image).unsqueeze_(0)

    # # inference on CPU
    # raw_out = model_arch(x)
    # print('raw_out.data: ', raw_out.data)
    ###

    ###
    # path = untar_data(URLs.CAMVID_TINY)

    # def acc_camvid(inp, targ, void_code=0):
    #     targ = targ.squeeze(1)
    #     mask = targ != void_code
    #     return (inp.argmax(dim=1)[mask] == targ[mask]).float().mean()

    # def get_y(o, path=path):
    #     return path / "labels" / f"{o.stem}_P{o.suffix}"
    
    # dls = SegmentationDataLoaders.from_label_func(
    #     path,
    #     bs=8,
    #     fnames=get_image_files(path / "images"),
    #     label_func=get_y,
    #     codes=np.loadtxt(path / "codes.txt", dtype=str),
    #     num_workers=0
    # )
    # print(dls.one_batch()[0].shape[-2:], get_c(dls))

    # learn = unet_learner(dls, resnet50, metrics=acc_camvid)

    # keep_path = learn.path
    # print('keep_path first: ', keep_path)

    # # learn.fine_tune(10)

    # learn.path = Path('.')
    # # learn.export("fastai_unet.pkl")

    # # learn.show_results()
    # # plt.show()

    # def acc_camvid(*_): pass
    # def get_y(*_): pass

    # learn = load_learner("fastai_unet.pkl")
    # torch.save(learn.model.state_dict(), "./fasti_unet_weights.pth")

    # model_arch = learn.model

    # image_path = "c:/Users/James/.fastai/data/camvid_tiny/images/Seq05VD_f01080.png"
    # pred_fastai = learn.predict(image_path)
    # print(pred_fastai[0].numpy())

    # class DynamicUnetDIY(SequentialEx):
    #     "Create a U-Net from a given architecture."

    #     def __init__(
    #         self,
    #         arch=resnet50,
    #         n_classes=32,
    #         img_size=(96, 128),
    #         blur=False,
    #         blur_final=True,
    #         y_range=None,
    #         last_cross=True,
    #         bottle=False,
    #         init=nn.init.kaiming_normal_,
    #         norm_type=None,
    #         self_attention=None,
    #         act_cls=defaults.activation,
    #         n_in=3,
    #         cut=None,
    #         **kwargs
    #     ):
    #         meta = model_meta.get(arch, _default_meta)
    #         encoder = create_body(
    #             arch, n_in, pretrained=False, cut=ifnone(cut, meta["cut"])
    #         )
    #         imsize = img_size

    #         sizes = model_sizes(encoder, size=imsize)
    #         sz_chg_idxs = list(reversed(_get_sz_change_idxs(sizes)))
    #         self.sfs = hook_outputs([encoder[i] for i in sz_chg_idxs], detach=False)
    #         x = dummy_eval(encoder, imsize).detach()

    #         ni = sizes[-1][1]
    #         middle_conv = nn.Sequential(
    #             ConvLayer(ni, ni * 2, act_cls=act_cls, norm_type=norm_type, **kwargs),
    #             ConvLayer(ni * 2, ni, act_cls=act_cls, norm_type=norm_type, **kwargs),
    #         ).eval()
    #         x = middle_conv(x)
    #         layers = [encoder, BatchNorm(ni), nn.ReLU(), middle_conv]

    #         for i, idx in enumerate(sz_chg_idxs):
    #             not_final = i != len(sz_chg_idxs) - 1
    #             up_in_c, x_in_c = int(x.shape[1]), int(sizes[idx][1])
    #             do_blur = blur and (not_final or blur_final)
    #             sa = self_attention and (i == len(sz_chg_idxs) - 3)
    #             unet_block = UnetBlock(
    #                 up_in_c,
    #                 x_in_c,
    #                 self.sfs[i],
    #                 final_div=not_final,
    #                 blur=do_blur,
    #                 self_attention=sa,
    #                 act_cls=act_cls,
    #                 init=init,
    #                 norm_type=norm_type,
    #                 **kwargs
    #             ).eval()
    #             layers.append(unet_block)
    #             x = unet_block(x)

    #         ni = x.shape[1]
    #         if imsize != sizes[0][-2:]:
    #             layers.append(PixelShuffle_ICNR(ni, act_cls=act_cls, norm_type=norm_type))
    #         layers.append(ResizeToOrig())
    #         if last_cross:
    #             layers.append(MergeLayer(dense=True))
    #             ni += in_channels(encoder)
    #             layers.append(
    #                 ResBlock(
    #                     1,
    #                     ni,
    #                     ni // 2 if bottle else ni,
    #                     act_cls=act_cls,
    #                     norm_type=norm_type,
    #                     **kwargs
    #                 )
    #             )
    #         layers += [
    #             ConvLayer(ni, n_classes, ks=1, act_cls=None, norm_type=norm_type, **kwargs)
    #         ]
    #         apply_init(nn.Sequential(layers[3], layers[-2]), init)
    #         # apply_init(nn.Sequential(layers[2]), init)
    #         if y_range is not None:
    #             layers.append(SigmoidRange(*y_range))
    #         super().__init__(*layers)

    #     def __del__(self):
    #         if hasattr(self, "sfs"):
    #             self.sfs.remove()

    # print(SequentialEx.mro())

    # model_torch_rep = DynamicUnetDIY()
    # state = torch.load("./fasti_unet_weights.pth")
    # model_torch_rep.load_state_dict(state)
    # model_torch_rep.eval()

    # model_arch.load_state_dict(state)
    # model_arch.eval()

    # image = Image.open(image_path).convert("RGB")
    # image_tfm = transforms.Compose(
    #     [
    #         transforms.Resize((96, 128)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # x = image_tfm(image).unsqueeze_(0)

    # # inference on CPU
    # raw_out = model_torch_rep(x)
    # print(raw_out.shape)

    # print('raw_out: ', raw_out)

    # pred_res = raw_out[0].argmax(dim=0).numpy().astype(np.uint8)
    # print(pred_res)

    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # img = cv2.resize(img, (128, 96))

    # img = img.astype("float32") / 255.0
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
    # img -= imagenet_mean
    # img /= imagenet_std
    # img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    # img = torch.from_numpy(img)
    # img = img.unsqueeze(0)
    # # img = img.to(device) 
    # output = model_torch_rep(img)
    # print('opencv result1: ', output.shape)

    # print('output: ', output)

    # pred_res1 = output[0].argmax(dim=0).numpy().astype(np.uint8)
    # print('opencv result2: ', pred_res1)

    # output_arch = model_arch(img)
    # print('output_arch1: ', output_arch.shape)

    # print('output_arch: ', output_arch)

    # pred_res_arch = output_arch[0].argmax(dim=0).numpy().astype(np.uint8)
    # print('output_arch2: ', pred_res_arch)

    # #####

    # class AlbumentationsTransform(RandTransform):
    #     "A transform handler for multiple `Albumentation` transforms"
    #     split_idx,order=None,2
    #     def __init__(self, train_aug, valid_aug): store_attr()
        
    #     def before_call(self, b, split_idx):
    #         self.idx = split_idx
        
    #     def encodes(self, img: PILImage):
    #         if self.idx == 0:
    #             aug_img = self.train_aug(image=np.array(img))['image']
    #         else:
    #             aug_img = self.valid_aug(image=np.array(img))['image']
    #         return PILImage.create(aug_img)
    
    # def get_train_aug_test(): return A.Resize(224, 224)

    # def get_valid_aug_test(): return A.Resize(224, 224)

    # def label_func(*_): pass
    
    # learn = load_learner("export.pkl")
    # torch.save(learn.model.state_dict(), "./export.pth")

    # model_arch = learn.model
    
    # pred_fastai = learn.predict(image_path)
    # print(pred_fastai)

    
    # state = torch.load("./export.pth")
    
    # model_arch.load_state_dict(state)
    # model_arch.eval()

    # image = Image.open(image_path).convert("RGB")
    # image_tfm = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # x = image_tfm(image).unsqueeze_(0)

    # # inference on CPU
    # raw_out = model_arch(x)
    # print(raw_out.shape)

    # print('raw_out: ', raw_out)    

    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # img = cv2.resize(img, (224, 224))

    # img = img.astype("float32") / 255.0
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
    # img -= imagenet_mean
    # img /= imagenet_std
    # img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    # img = torch.from_numpy(img)
    # img = img.unsqueeze(0)
    
    # output_arch = model_arch(img)
    # print('output_arch1: ', output_arch.shape)

    # print('output_arch: ', output_arch)
    ###

    ###
    # path = Path(data_dir)

    # test_img_pathname = 'd:/data_seq/gongqiWinding/Z75_DF-4105H-BD/211129/shrinkVideo/bigDatasets/test/imgs/img00022.jpg'
    
    # # # load FastAI learner
    # # class AlbumentationsTransform(RandTransform):
    # #     "A transform handler for multiple `Albumentation` transforms"
    # #     split_idx,order=None,2
    # #     def __init__(self, train_aug, valid_aug): store_attr()
        
    # #     def before_call(self, b, split_idx):
    # #         self.idx = split_idx
        
    # #     def encodes(self, img: PILImage):
    # #         if self.idx == 0:
    # #             aug_img = self.train_aug(image=np.array(img))['image']
    # #         else:
    # #             aug_img = self.valid_aug(image=np.array(img))['image']
    # #         return PILImage.create(aug_img)

    # def label_func(fname):
    #     return parent_label(fname)
    
    # dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
    #                    get_items=get_image_files,
    #                    get_y=label_func,
    #                    splitter=GrandparentSplitter(valid_name='val'),
    #                    item_tfms=Resize(224),
    #                    batch_tfms=aug_transforms(size=224))

    # dls = dblock.dataloaders(path, num_workers=0, verbose=True)

    # dls.show_batch(max_n=3)    

    # # learn = cnn_learner(dls, shufflenet_v2_x1_0, metrics=[error_rate, accuracy])
    # learn = cnn_learner(dls, resnet18, metrics=[error_rate, accuracy])

    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)    

    # # learn.fine_tune(1, lrs.valley)
    # learn.fine_tune(0, lrs.valley)

    # # plot training loss 
    # learn.recorder.plot_loss()

    # # reset the model path
    # learn.path = Path('.')

    # learn.export('export_211225.pkl')

    # plt.show()

    # learn = load_learner('./export_211225.pkl')
    
    # # not hot dog
    # print(learn.predict(test_img_pathname))

    # print(learn.dls.transform)

    # labels = learn.dls.vocab
    # print(labels)

    # pytorch_model = learn.model.eval() # gets the PyTorch model
    # softmax_layer = torch.nn.Softmax(dim=1) # define softmax
    # normalization_layer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization layer

    # # assembling the final model
    # final_model = nn.Sequential(
    #     normalization_layer,
    #     pytorch_model,
    #     softmax_layer
    # )

    # # loading an image and converting to tensor
    # def image_transform(path: str, size: int) -> torch.Tensor:
    #     '''Helper function to transform image.'''
    #     image = Image.open(path)

    #     # transformation pipeline
    #     transformation = transforms.Compose([
    #                 transforms.Resize([size,size]), # resizes image
    #                 transforms.ToTensor() # converts to image to tensor
    #             ])

    #     image_tensor = transformation(image).unsqueeze(0)
    #     print('Tensor shape: ', image_tensor.shape)

    #     return image_tensor
    
    # not_hot_dog_tensor = image_transform(test_img_pathname, 224)

    # with torch.no_grad():
    #     results = final_model(not_hot_dog_tensor)
    # print(labels[np.argmax(results.detach().numpy())], results.detach().numpy().astype(float))
    ###

    ###
    # PATH = Path('d:/data/hot_dog-not_hot_dog/data/')

    # size = 256

    # def label_func(fname):
    #     return parent_label(fname)
    
    # dogs = DataBlock(
    #     blocks=(ImageBlock, CategoryBlock), 
    #     get_items=get_image_files,
    #     splitter=GrandparentSplitter(valid_name='test'),                                  # splits dataset to train and test
    #     get_y=label_func,  # extracts label from image name
    #     item_tfms=Resize(256),                                      # applied to all items in a dataset
    #     batch_tfms=aug_transforms(size=256)                         # transformation applied to a batch
    #     )
    
    # dls = dogs.dataloaders(PATH, num_workers=0, verbose=False)

    # # dls.show_batch(max_n=3)
    # # plt.show()

    # # learn = cnn_learner(dls, resnet18, metrics=[error_rate, accuracy])
    # _default_meta = _resnet_meta
    # learn = cnn_learner(dls, shufflenet_v2_x1_0, metrics=[error_rate, accuracy])

    # arch = shufflenet_v2_x1_0

    # learn.fine_tune(8, 1e-3)

    # learn.recorder.plot_loss()

    # learn.path = Path('.')
    # # learn.export('hot_dog_model_resnet18_256_256.pkl')
    # learn.export('hot_dog_model_shufflenet_256_256.pkl')
    # plt.show()

    # # learn = load_learner('./hot_dog_model_resnet18_256_256.pkl')
    # learn = load_learner('./hot_dog_model_shufflenet_256_256.pkl')

    # test_img_pathname = 'd:/data/hot_dog-not_hot_dog/seefood/train/hot_dog/1013916.jpg'

    # print(learn.predict(test_img_pathname))

    # # print(learn.dls.transform)

    # labels = learn.dls.vocab
    # print(labels)

    # pytorch_model = learn.model.eval() # gets the PyTorch model
    # softmax_layer = torch.nn.Softmax(dim=1) # define softmax
    # normalization_layer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization layer

    # # assembling the final model
    # final_model = nn.Sequential(
    #     normalization_layer,
    #     pytorch_model,
    #     softmax_layer
    # )

    # def image_transform(path: str, size: int) -> torch.Tensor:
    #     '''Helper function to transform image.'''
    #     image = Image.open(path)

    #     # transformation pipeline
    #     transformation = transforms.Compose([
    #                 transforms.Resize([size,size]), # resizes image
    #                 transforms.ToTensor() # converts to image to tensor
    #             ])

    #     image_tensor = transformation(image).unsqueeze(0)
    #     print('Tensor shape: ', image_tensor.shape)

    #     return image_tensor
    
    # hot_dog_tensor = image_transform(test_img_pathname, 256)

    # with torch.no_grad():
    #     results = final_model(hot_dog_tensor)
    # print(labels[np.argmax(results.detach().numpy())], results.detach().numpy().astype(float))

    # with torch.no_grad():
    #     # image = Image.open(test_img_pathname).resize((size,size)).convert("RGB")
    #     # image_tensor = transforms.ToTensor()(image)
    #     # image_tensor = image_tensor.unsqueeze(0)
    #     image_tensor_normalized = normalization_layer(hot_dog_tensor)
    #     result = pytorch_model(image_tensor_normalized)
    #     result1 = result[0][0]
    #     result2 = result[0][1]
    #     pre1 = np.exp(result1.item())/(np.exp(result1.item())+np.exp(result2.item()))
    #     pre2 = np.exp(result2.item())/(np.exp(result1.item())+np.exp(result2.item()))
    #     softmax_out = softmax_layer(result)
    # print('test results(PIL image): ', labels[np.argmax((pre1,pre2))], (pre1,pre2))
    # print('test results(softmax)', labels[np.argmax(softmax_out.detach().numpy())], softmax_out.detach().numpy().astype(float))

    # img = cv2.imread(test_img_pathname)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # img = cv2.resize(img, (256, 256))

    # img = img.astype("float32") / 255.0
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
    # img_normalized = img - imagenet_mean
    # img_normalized /= imagenet_std
    # img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    # img = torch.from_numpy(img)
    # img = img.unsqueeze(0)
    
    # output_arch = final_model(img)    
    # print('output_arch: ', output_arch)

    # img_normalized = np.transpose(img_normalized, (2, 0, 1)).astype(np.float32)
    # img_normalized = torch.from_numpy(img_normalized)
    # img_normalized = img_normalized.unsqueeze(0)

    # with torch.no_grad():
    #     result = pytorch_model(img_normalized)
    #     result1 = result[0][0]
    #     result2 = result[0][1]
    #     pre1 = np.exp(result1.item())/(np.exp(result1.item())+np.exp(result2.item()))
    #     pre2 = np.exp(result2.item())/(np.exp(result1.item())+np.exp(result2.item()))        
    # print('test results(opencv): ', labels[np.argmax((pre1,pre2))], (pre1,pre2))
    ###

    ###
    # tfms = transforms.Compose([transforms.ToTensor(),
    #                            transforms.Normalize((0.1307,), (0.3081))
    #                            ])
    
    # train_dset = datasets.MNIST('d:/data/mnist/data', train=True, download=True, transform=tfms)
    # valid_dset = datasets.MNIST('d:/data/mnist/data', train=False, transform=tfms)

    # train_loader = DataLoader(train_dset, batch_size=256,
    #                           shuffle=True, num_workers=1, pin_memory=True)

    # test_loader = DataLoader(valid_dset, batch_size=512,
    #                          shuffle=False, num_workers=1, pin_memory=True)
    
    # dls = DataLoaders(train_loader, test_loader)

    # class Flatten(nn.Module):
    #     "Flattens an input"
    #     def forward(self, x): return x.view(x.size(0), -1)
    
    # class Net(nn.Sequential):
    #     def __init__(self):
    #         super().__init__(
    #             nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
    #             nn.Conv2d(32, 64, 3, 1), 
    #             # A head to the model
    #             nn.MaxPool2d(2), nn.Dropout2d(0.25),
    #             Flatten(), nn.Linear(9216, 128), nn.ReLU(),
    #             nn.Dropout2d(0.5), nn.Linear(128, 10), nn.LogSoftmax(dim=1)
    #         )
    
    # opt_func = partial(OptimWrapper, opt=optim.Adam)

    # # learn = Learner(dls, Net(), loss_func=F.nll_loss, opt_func=opt_func, metrics=accuracy)

    # # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # # print('lrs: ', lrs)    
    
    # # # learn.fit_one_cycle(n_epoch=1, lr_max=1e-2)
    # # learn.fit_one_cycle(n_epoch=1, lr_max=lrs.steep)

    # # learn.save('myModel', with_opt=False)
    # # plt.show()

    # new_net = Net()
    # net_dict = torch.load('models/myModel.pth') 
    # new_net.load_state_dict(net_dict)

    # data_path = untar_data(URLs.MNIST_SAMPLE)

    # single_image = data_path/'valid'/'3'/'8483.png'

    # im = Image.open(single_image)

    # print('pil image size: ', im.shape)

    # tfmd_im = tfms(im)

    # print('pil tfmd_im size: ', tfmd_im.shape)

    # tfmd_im = tfmd_im.unsqueeze(0)

    # print('pil tfmd_im unsqueeze(0) size: ', tfmd_im.shape)

    # with torch.no_grad():
    #     new_net.cuda()
    #     tfmd_im = tfmd_im.cuda()
    #     preds = new_net(tfmd_im)
    #     print(preds.argmax(dim=-1), preds)

    # img = cv2.imread(str(single_image))
    # print(img.shape)
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # img = cv2.resize(img, (256, 256))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # print('opencv gray img size: ', img.shape)

    # img = img.astype("float32") / 255.0
    # # imagenet_mean = [0.485, 0.456, 0.406]
    # # imagenet_std = [0.229, 0.224, 0.225]
    # imagenet_mean = [0.1307]
    # imagenet_std = [0.3081]
    # img_normalized = img - imagenet_mean
    # img_normalized /= imagenet_std
    # # img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    # img = np.expand_dims(img, axis=0)
    # print('opencv expand_dims gray img size: ', img.shape)
    # img = torch.from_numpy(img)
    # img = img.unsqueeze(0)
    # print('opencv unsqueeze(0) expand_dims gray img size: ', img.shape)
    
    # with torch.no_grad():
    #     img = img.to(device)
    #     result = new_net(img)
    #     print('opencv results: ', preds.argmax(dim=-1), preds)
    ###

    ###
    # num_ftrs = arch.fc.in_features
    # arch.fc = nn.Linear(num_ftrs, 10) 
    # tfms = transforms.Compose([transforms.ToTensor(),
    #                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #                            ])
    
    # train_dset = datasets.CIFAR10('d:/data', train=True, download=True, transform=tfms)
    # valid_dset = datasets.CIFAR10('d:/data', train=False, transform=tfms)

    # train_loader = DataLoader(train_dset, batch_size=64,
    #                           shuffle=True, num_workers=0, pin_memory=True)

    # test_loader = DataLoader(valid_dset, batch_size=64,
    #                          shuffle=False, num_workers=0, pin_memory=True)
    
    # dls = DataLoaders(train_loader, test_loader)
        
    # opt_func = partial(OptimWrapper, opt=optim.Adam)

    # learn = Learner(dls, arch, loss_func=F.cross_entropy, opt_func=opt_func, metrics=accuracy)

    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)

    # learn.fine_tune(1, lrs.valley)    
    
    # # # learn.fit_one_cycle(n_epoch=1, lr_max=1e-2)
    # # learn.fit_one_cycle(n_epoch=1, lr_max=lrs.steep)

    # # learn.save('myModel_arch', with_opt=False)
    # # plt.show()

    # new_net = arch
    # net_dict = torch.load('models/myModel_arch.pth') 
    # new_net.load_state_dict(net_dict)

    # data_path = untar_data(URLs.MNIST_SAMPLE)

    # single_image = Path('d:/data/cats_dogs/test_set/dogs/dog.4143.jpg')

    # im = Image.open(single_image)

    # print('pil image size: ', im.shape)

    # tfmd_im = tfms(im)

    # print('pil tfmd_im size: ', tfmd_im.shape)

    # tfmd_im = tfmd_im.unsqueeze(0)

    # print('pil tfmd_im unsqueeze(0) size: ', tfmd_im.shape)

    # with torch.no_grad():
    #     new_net.cuda()
    #     tfmd_im = tfmd_im.cuda()
    #     preds = new_net(tfmd_im)
    #     print(preds.argmax(dim=-1), preds)

    # img = cv2.imread(str(single_image))
    # print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # img = cv2.resize(img, (256, 256))    

    # print('opencv img size: ', img.shape)

    # img = img.astype("float32") / 255.0
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]    
    # img_normalized = img - imagenet_mean
    # img_normalized /= imagenet_std
    # img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    # print('opencv transpose img size: ', img.shape)
    # img = torch.from_numpy(img)
    # img = img.unsqueeze(0)
    # print('opencv unsqueeze(0) transpose img size: ', img.shape)
    
    # with torch.no_grad():
    #     img = img.to(device)
    #     result = new_net(img)
    #     print('opencv results: ', preds.argmax(dim=-1), preds)
    ###

    ###
    path = Path('d:/data/hot_dog-not_hot_dog/data/')

    size = 224

    normalization_layer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    softmax_layer = torch.nn.Softmax(dim=1)

    tfms = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])
    
    tfms_fastai = Resize(224)

    tfms_pytorch = transforms.Resize(224)

    class AlbumentationsTransform(RandTransform):
        "A transform handler for multiple `Albumentation` transforms"
        split_idx,order=None,2
        def __init__(self, train_aug, valid_aug): store_attr()
        
        def before_call(self, b, split_idx):
            self.idx = split_idx
        
        def encodes(self, img: PILImage):
            if self.idx == 0:
                aug_img = self.train_aug(image=np.array(img))['image']
            else:
                aug_img = self.valid_aug(image=np.array(img))['image']
            return PILImage.create(aug_img)

    def get_train_aug(): return data_transforms_fastai['train']

    def get_valid_aug(): return data_transforms_fastai['val']

    # def get_train_aug_test(): return A.Resize(224, 224)

    # def get_valid_aug_test(): return A.Resize(224, 224)

    # def get_train_aug_test(): return A.Compose([A.Resize(224, 224),
    #                                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    # def get_valid_aug_test(): return A.Compose([A.Resize(224, 224),
    #                                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    
    item_tfms = AlbumentationsTransform(get_train_aug(), get_valid_aug())
    # item_tfms = AlbumentationsTransform(get_train_aug_test(), get_valid_aug_test())

    # dls = ImageDataLoaders.from_folder(path, valid='test',
    #                                    item_tfms=RandomResizedCrop(128, min_scale=0.35), 
    #                                    batch_tfms=Normalize.from_stats(*imagenet_stats),
    #                                    num_workers=0)
    
    # dls = ImageDataLoaders.from_folder(path, valid='test',
    #                                    item_tfms=item_tfms, 
    #                                    batch_tfms=Normalize.from_stats(*imagenet_stats),
    #                                    num_workers=0)
    
    # dls = ImageDataLoaders.from_folder(path, valid='test',
    #                                    item_tfms=item_tfms, 
    #                                    batch_tfms=None,
    #                                    num_workers=0)
    
    # dls = ImageDataLoaders.from_folder(path, valid='test',
    #                                    item_tfms=item_tfms, 
    #                                    batch_tfms=Normalize.from_stats(*imagenet_stats),
    #                                    num_workers=0)

    # image_transform = transforms.Compose([transforms.Resize((224,224)), transforms.Lambda(lambda image: Image.fromarray(np.array(image).astype(np.float32)))])

    # dls = ImageDataLoaders.from_folder(path, valid='test',
    #                                 #    item_tfms=Resize(224), 
    #                                    item_tfms=image_transform, 
    #                                    shuffle=False,
    #                                 #    batch_tfms=transforms.ToTensor(),
    #                                 #    before_batch=IntToFloatTensor(),
    #                                    batch_tfms=None,
    #                                    num_workers=0,
    #                                    pin_memory=False)

    def label_func(fname):
        return parent_label(fname)
    
    dogs = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files,
        splitter=GrandparentSplitter(valid_name='test'),                                  # splits dataset to train and test
        get_y=label_func,  # extracts label from image name
        item_tfms=Resize(256),                                      # applied to all items in a dataset
        batch_tfms=aug_transforms(size=256)                         # transformation applied to a batch
        )
    
    dls = dogs.dataloaders(path,
                           pin_memory=False,
                           shuffle=False,
                           num_workers=0,
                           verbose=False)
    
    print('fastai device: ', dls.device)

    b = dls.one_batch()

    for batch_ndx, sample in enumerate(dls):
        print(batch_ndx)
        print(type(sample))
        for sample1 in enumerate(sample):
            print(type(sample1))
    
    def set_seed_fastai(dls,x=42):
        random.seed(x)
        dls.rng.seed(x)
        np.random.seed(x)
        torch.manual_seed(x)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)
    
    num_ftrs = arch.fc.in_features
    arch.fc = nn.Linear(num_ftrs, 2) 
    
    learn = Learner(dls, arch, loss_func=CrossEntropyLossFlat(), opt_func=Adam, metrics=accuracy)
    # learn = Learner(dls, arch, loss_func=CrossEntropyLossFlat(), opt_func=Adam, metrics=F1Score)

    learn.path = Path('./best_model_fastai')
    model_name = 'best_model_fastai'

    single_image = Path('d:/data/hot_dog-not_hot_dog/seefood/train/hot_dog/1013916.jpg')
    # single_image = Path('d:/data/hot_dog-not_hot_dog/seefood/test/not_hot_dog/82535.jpg')

    learn.fit_one_cycle(n_epoch=1, lr_max=1e-2)
    
    # lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    # print('lrs: ', lrs)

    # # learn.fine_tune(10,
    # #                 lrs.valley,
    # #                 cbs=[EarlyStoppingCallback(monitor='accuracy', min_delta=0.01, patience=5),
    # #                      SaveModelCallback(monitor='accuracy', min_delta=0.01, fname=model_name)])
    
    # learn.fit_one_cycle(n_epoch=10,
    #                     lr_max=lrs.minimum,
    #                     cbs=[EarlyStoppingCallback(monitor='accuracy', min_delta=1e-5, patience=5),
    #                          SaveModelCallback(monitor='accuracy', min_delta=1e-5, fname=model_name)])
    # # # learn.fit_one_cycle(n_epoch=1, lr_max=1e-2)
    # # learn.fit_one_cycle(n_epoch=1, lr_max=lrs.steep)

    # print('learn.predict results: ', learn.predict(single_image))
    # learn.save('myModel_arch', with_opt=False)
    # learn.export('hot_dog_model_shufflenet_224_224.pkl')
    # plt.show()

    learn_inf = load_learner(learn.path/'hot_dog_model_shufflenet_224_224.pkl')
    print('learn_inf.predict results: ', learn_inf.predict(single_image))

    labels = learn_inf.dls.vocab


    # new_net_test = arch()
    new_net = learn_inf.model.eval()
    net_dict = torch.load('./best_model_fastai/models/myModel_arch.pth')
    new_net.load_state_dict(net_dict)
    
    im = Image.open(single_image).resize((224,224))    
    
    tfmd_im = transforms.ToTensor()(im)

    tfmd_im = normalization_layer(tfmd_im)

    tfmd_im = tfmd_im.unsqueeze(0)

    # print('tfmd_im.shape unsqueeze size: ', tfmd_im.shape)

    with torch.no_grad():
        new_net.cuda()
        tfmd_im = tfmd_im.cuda()
        preds = new_net(tfmd_im)
        preds = softmax_layer(preds)
        print('pil image results: ', labels[preds.argmax(dim=-1)], preds.argmax(dim=-1), preds)    
    
    
    img = cv2.imread(str(single_image))
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))    

    # print('opencv img size: ', img.shape)

    img = img.astype("float32") / 255.0
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]    
    img_normalized = img - imagenet_mean
    img_normalized /= imagenet_std
    # img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.transpose(img_normalized, (2, 0, 1)).astype(np.float32)

    # print('opencv transpose img size: ', img.shape)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    # print('opencv unsqueeze(0) transpose img size: ', img.shape)
    
    with torch.no_grad():
        img = img.to(device)
        result = new_net(img)
        result1 = result[0][0]
        result2 = result[0][1]
        pre1 = np.exp(result1.item())/(np.exp(result1.item())+np.exp(result2.item()))
        pre2 = np.exp(result2.item())/(np.exp(result1.item())+np.exp(result2.item()))        
        print('test results(opencv): ', labels[np.argmax((pre1,pre2))], np.argmax((pre1,pre2)), (pre1,pre2))
    
    # Saving Model_ft with Shapes
    torch.save(new_net, 'new_net.pth')

    model_full_path = 'new_net.pth'

    model_ft_full = torch.load(model_full_path)

    # export to onnx
    input_image = torch.zeros((1,3,224,224))
    input_image = input_image.to(device)
    
    input_names = ["x"]
    output_names = ["y"]

    #convert pytorch to onnx
    onnx_model_pathname = 'new_net' + '.onnx'
    torch_out = onnx.export(
        model_ft_full, input_image, onnx_model_pathname, input_names=input_names, output_names=output_names)
    
    # load your predefined ONNX model
    model = onnx_origin.load(onnx_model_pathname)

    # convert model
    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    # Save the ONNX model
    onnx_model_simplified_pathname = onnx_model_pathname[:-5] + '_simplified' + '.onnx'
    onnx_origin.save(model_simp, onnx_model_simplified_pathname)

    # verify pytorch onnx
    #test image
    img_path = str(single_image)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img1 = img.to(device)    

    #pytorch test
    model = torch.load(model_full_path)
    
    output = model.forward(img1)
    val, cls = torch.max(output.data, 1)
    print("[pytorch]--->predicted class:", cls.item())
    print("[pytorch]--->predicted value:", val.item())

    #onnx test
    sess = rt.InferenceSession(onnx_model_pathname)
    
    x = "x"
    y = ["y"]
    output = sess.run(y, {x: img.numpy()})
    cls = np.argmax(output[0][0], axis=0)
    val = output[0][0][cls]
    print("[onnx]--->predicted class:", cls)
    print("[onnx]--->predicted value:", val)
    
    #simplified onnx test
    sess = rt.InferenceSession(onnx_model_simplified_pathname)
    
    x = "x"
    y = ["y"]
    output = sess.run(y, {x: img.numpy()})
    cls = np.argmax(output[0][0], axis=0)
    val = output[0][0][cls]
    print("[onnx_simplified]--->predicted class:", cls)
    print("[onnx_simplified]--->predicted value:", val)
    ###
    # end of addition 2112200810

    """
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
    """
