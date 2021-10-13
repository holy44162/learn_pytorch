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

# added by Holy 2108300810
from torchvision.io import read_image
from pathlib import Path
import PIL
# end of addition 2108300810

import cv2 # added by Holy 2109010810


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
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
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def list_files(directory, extension):
    return (f for f in sorted(os.listdir(directory)) if f.endswith('.' + extension))

if __name__ == "__main__":
    """
    # Load Data
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # data_dir = 'data/hymenoptera_data'
    # data_dir = 'data/z75_data' # added by Holy 2108171500
    data_dir = 'e:/dnn_data/z75_data' # added by Holy 2108171500
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                  shuffle=False, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    # model_ft and model_conv
    model_ft = models.resnet18()
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    # Loading Model_ft Weights
    model_ft.load_state_dict(torch.load('model_ft_weights.pth'))
    model_ft.eval()

    # load model_ft from model file
    model_ft_full = torch.load('model_ft.pth')
    model_ft_full = model_ft_full.to(device)

    # predict with model_ft and model_ft_full
    num_images = 6
    was_training = model_ft.training
    images_so_far = 0
    fig = plt.figure()

    bool_break = False

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            print(inputs.shape)
            print(labels.shape)
            print(inputs[0,0,0,170:175])
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

            print(outputs)
            print(preds)

            outputs_full = model_ft_full(inputs)
            _, preds_full = torch.max(outputs_full, 1)

            print(outputs_full)
            print(preds_full)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model_ft.train(mode=was_training)
                    bool_break = True
                    break
            if bool_break:
                break
        model_ft.train(mode=was_training)
    
    plt.show()

    # added by Holy 2108300810
    image_path = str(Path(r'e:\dnn_data\z75_data\val\mess') / 'img00001.jpg')
    mess1 = read_image(image_path)
    mess1 = mess1.unsqueeze(0)

    # sample_transforms = nn.Sequential(    
    #     # We use single int value inside a list due to torchscript type restrictions
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ConvertImageDtype(torch.float),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # )

    sample_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        mess1 = sample_transforms(mess1)
        print(mess1.shape)
        print(mess1[0,0,0,170:175])
        mess1 = mess1.to(device)
        outputs = model_ft(mess1)
        _, preds = torch.max(outputs, 1)
    
    print(outputs)
    print(preds)

    with torch.no_grad():
        outputs_full = model_ft_full(mess1)
        _, preds_full = torch.max(outputs_full, 1)

    print(outputs_full)
    print(preds_full)

    PIL_image = PIL.Image.open(image_path)    
    # PIL_image = data_transforms['val'](PIL_image)
    # PIL_image = PIL_image.unsqueeze(0)
    # print(PIL_image.shape)
    # print(PIL_image[0,0,0,170:175])

    with torch.no_grad():
        PIL_image = data_transforms['val'](PIL_image)
        PIL_image = PIL_image.unsqueeze(0)
        print(PIL_image.shape)
        print(PIL_image[0,0,0,170:175])
        PIL_image = PIL_image.to(device)
        outputs = model_ft(PIL_image)
        _, preds = torch.max(outputs, 1)
    
    print(outputs)
    print(preds)

    with torch.no_grad():
        outputs_full = model_ft_full(PIL_image)
        _, preds_full = torch.max(outputs_full, 1)

    print(outputs_full)
    print(preds_full)
    # end of addition 2108300810
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model_ft from model file
    model_ft_full = torch.load('model_ft_shufflenet_v2.pth')
    # model_ft_full = model_ft_full.to(device) # hided by Holy 2109010810

    strDatasetPrefix = 'd:/data_seq/gongqiWinding/Z75_DF-4105H-BD/210820/shrinkVideo/bigDatasets/testValidateCV'
    path = str(Path(strDatasetPrefix) / 'imgs')
    print(path)

    strYTest = str(Path(strDatasetPrefix) / 'y_Test.txt')
    print(strYTest)

    with open(strYTest) as fInYTest:
        vecBMessYTest = fInYTest.readlines()
        vecBMessYTest = [bool(int(i)) for i in vecBMessYTest]
        # print(vecBMessYTest)

    vecBMess = []

    # Setting the points for cropped image
    left = 25
    top = 276
    right = 25+681-1
    bottom = 276+201-1

    # added by Holy 2109010810
    num = 0
    # specify ImageNet mean and standard deviation
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    # end of addition 2109010810

    str_imgs_list = list_files(path, "jpg")
    for str_img_name in str_imgs_list:
        str_img_name = str(Path(path) / str_img_name)
        print(str_img_name)

        # hided by Holy 2109010810
        # PIL_image = PIL.Image.open(str_img_name)
        # PIL_image = PIL_image.crop((left, top, right, bottom))
        # with torch.no_grad():
        #     PIL_image = data_transforms['val'](PIL_image)
        #     PIL_image = PIL_image.unsqueeze(0)            
        #     PIL_image = PIL_image.to(device)
        #     outputs = model_ft_full(PIL_image)
        #     _, preds = torch.max(outputs, 1)
        #     preds = 1 - int(preds)
        #     vecBMess.append(bool(preds))
        # end of hide 2109010810

        # added by Holy 2109010810
        img = cv2.imread(str_img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # added by Holy 2109131500
        # img = img[276:276+201, 25:25+681] # hided by Holy 2109071500
        img = cv2.resize(img, (224, 224))

        # tested by Holy 2109060810
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        # break
        # end of test 2109060810

        img = img.astype("float32") / 255.0
        img -= imagenet_mean
        img /= imagenet_std
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img)        
        img = img.unsqueeze(0)
        img = img.to(device) # added by Holy 2109080810
        output = model_ft_full.forward(img)
        # output = model_ft_full(img) # added by Holy 2109060810
        # val, cls = torch.max(output.data, 1)
        val, cls = torch.sigmoid(output), torch.sigmoid(output) >= 0.5
        # val, cls = torch.max(output, 1) # added by Holy 2109060810
        print("[pytorch]--->predicted class:", cls.item())
        print("[pytorch]--->predicted value:", val.item())

        # values, indices = output.data.topk(2, dim=1, largest=True, sorted=True)
        # print("[pytorch]--->predicted topk class:", indices)
        # print("[pytorch]--->predicted topk value:", values)

        preds = 1 - int(cls.item())
        vecBMess.append(bool(preds))

        # if num == 10:
        #     break

        num += 1
        # end of addition2109010810
    
    # print(vecBMess)
    # print(len(vecBMess))

    vecBMessYTest_flip = [ not z for z in vecBMessYTest]
    vecBMess_flip = [ not z for z in vecBMess]

    vecBResult = [ x and y for (x,y) in zip(vecBMessYTest, vecBMess)]    
    tp = sum(vecBResult)
    print(f'tp: {tp}')

    vecBResult = [ x and y for (x,y) in zip(vecBMessYTest_flip, vecBMess)]
    fp = sum(vecBResult)
    print(f'fp: {fp}')

    vecBResult = [ x and y for (x,y) in zip(vecBMessYTest, vecBMess_flip)]
    fn1 = sum(vecBResult)
    print(f'fn: {fn1}')

    prec = float(tp) / float(tp + fp)
    rec = float(tp) / float(tp + fn1)
    dF1 = 2 * prec * rec / (prec + rec)
    print(f'prec: {prec}')
    print(f'rec: {rec}')
    print(f'f1: {dF1}')

    # added by Holy 2109090810
    vecBResult = [ x == y for (x,y) in zip(vecBMessYTest, vecBMess)]    
    acc = float(sum(vecBResult)) / float(len(vecBResult))
    print(f'acc: {acc}')
    print(f'num: {len(vecBResult)}')    
    # end of addition 2109090810
