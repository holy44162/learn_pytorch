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

from tqdm import tqdm # added by Holy 2109081500

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


def train_model(model, criterion, optimizer, scheduler, start_epochs, n_epochs, valid_loss_min_input=None, checkpoint_path=None, best_model_path=None):
    since = time.time()

    # added by Holy 2109041500
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input
    # end of addition 2109041500

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    break_num = 0 # added by Holy 2109100810

    # for epoch in range(num_epochs):
    for epoch in range(start_epochs, n_epochs+1): # added by Holy 2109041500        
        # initialize variables to monitor training and validation loss
        valid_loss = 0.0 # added by Holy 2109041500

        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('Epoch {}/{}'.format(epoch, n_epochs)) # added by Holy 2109041500
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            with tqdm(dataloaders[phase], unit="batch") as tepoch: # added by Holy 2109081500
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                # for inputs, labels in dataloaders[phase]:
                for inputs, labels in tepoch: # added by Holy 2109081500
                    tepoch.set_description(f"Epoch {epoch}") # added by Holy 2109081500

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

                    # added by Holy 2109081500
                    # correct = (preds == labels.data).sum().item()
                    accuracy = torch.mean((preds == labels.data).float()).item()
                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                    # added by Holy 2109081500
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
                
                # added by Holy 2109041500
                if phase == 'val':
                    valid_loss = epoch_loss
                # end of addition 2109041500

        print()

        # added by Holy 2109041500
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

            break_num = 0 # added by Holy 2109100810
        # end of addition 2109041500
        # added by Holy 2109100810
        else:
            break_num += 1
            if break_num > 10:
                break
        # end of addition 2109100810

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


if __name__ == "__main__":
    plt.ion()

    # Load Data
    # Data augmentation and normalization for training
    # Just normalization for validation

    # hided by Holy 2109041002
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }
    # end of hide 2109041002

    # added by Holy 2109080810
    INIT_LR = 1e-3
    # BATCH_SIZE = 2**9 # 1m 37s 0.549111
    # BATCH_SIZE = 2**8 # 1m 9s 0.512161
    # BATCH_SIZE = 2**7 # 1m 6s 0.507951
    BATCH_SIZE = 2**6 # 1m 5s 0.699252 Best val Acc: 0.867166
    # BATCH_SIZE = 2**5 # 1m 5s 0.667446
    # BATCH_SIZE = 2**4 # 1m 8s 0.744153
    # BATCH_SIZE = 2**3 # 1m 34s 0.783910
    # BATCH_SIZE = 2**2 # 2m 19s 0.967727
    # BATCH_SIZE = 2**1 # 3m 54s 0.964920
    # BATCH_SIZE = 2**0 # 6m 56s 0.757717
    EPOCHS = 100
    # end of addition 2109080810

    # added by Holy 2109041002
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([224, 224]),
            # transforms.RandomHorizontalFlip(), # added by Holy 2109100810
            # transforms.ColorJitter(brightness=.5, hue=.3), # added by Holy 2109100810
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # end of addition 2109041002

    # data_dir = 'data/hymenoptera_data'
    # data_dir = 'data/z75_data' # added by Holy 2108171500
    data_dir = 'e:/dnn_data/z75_data' # added by Holy 2108171500
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Visualize a few images
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    # Training the model
    # hided by Holy 2109031500
    # model_ft = models.resnet18(pretrained=True)
    model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True) # added by Holy 2109021500
    # model_ft = torch.hub.load('pytorch/vision:v0.9.1', 'shufflenet_v2_x0_5', pretrained=True) # added by Holy 2109030810
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    # end of hide 2109031500

    # added by Holy 2109031500
    # num_classes = 1000
    # model_width = 0.5
    # model_ft = shufflenet_v2.Network(num_classes, model_width)
    # params = torch.load('shufflenet_v2_x0.5.pth', map_location=lambda storage, loc: storage)
    # model_ft.load_state_dict(params)

    # model_ft.num_classes = 2
    # end of addition 2109031500

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss() # Best val Acc: 0.767495 0.820902
    # criterion = nn.BCEWithLogitsLoss() # Best val Acc: 

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=INIT_LR, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=INIT_LR)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    # model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    #                        num_epochs=25)
    # model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    #                        num_epochs=2) # added by Holy 2109030810
    
    # added by Holy 2109041500
    start_epochs = 1
    end_epochs = start_epochs + EPOCHS
    checkpoint_path = './checkpoint/current_checkpoint.pt'
    best_model_path = './best_model/best_model.pt'
    resume_training = False
    if resume_training:
        # added by Holy 2109060810
        model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=False)
        
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        
        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=7, gamma=0.1)
        # end of addition 2109060810

        # load the saved checkpoint
        model_ft, optimizer_ft, start_epochs, valid_loss_min = load_ckp(checkpoint_path, model_ft, optimizer_ft)
        end_epochs = start_epochs + EPOCHS
        print("model = ", model_ft)
        print("optimizer = ", optimizer_ft)
        print("start_epoch = ", start_epochs)
        print("valid_loss_min = ", valid_loss_min)
        print("valid_loss_min = {:.6f}".format(valid_loss_min))

        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           start_epochs, end_epochs, valid_loss_min, checkpoint_path, best_model_path)
    else:
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           start_epochs, end_epochs, np.Inf, checkpoint_path, best_model_path)
    # end of addition 2109041500
    
    # save model_ft
    torch.save(model_ft.state_dict(), 'model_ft_weights_shufflenet_v2.pth')
    # torch.save(model_ft.state_dict(), 'model_ft_weights_shufflenet_v2_x0_5.pth') # added by Holy 2109030810

    visualize_model(model_ft)

    # added by Holy 2109041500
    model_ft.eval()

    # Saving Model_ft with Shapes
    torch.save(model_ft, 'model_ft_shufflenet_v2.pth')

    # hided by Holy 2109080810
    # test_acc = 0.0
    # for samples, labels in dataloaders['val']:
    #     with torch.no_grad():
    #         samples, labels = samples.to(device), labels.to(device)
    #         output = model_ft(samples)
    #         # calculate accuracy
    #         # pred = torch.argmax(output, dim=1)
    #         _, preds = torch.max(output, 1) # added by Holy 2109060810
    #         # correct = pred.eq(labels)
    #         test_acc += torch.sum(preds == labels.data) # added by Holy 2109060810
    #         # test_acc += torch.mean(correct.float())
    # # print('Accuracy of the network on {} test images: {}%'.format(len(image_datasets['val']), round(test_acc.item()*100.0/len(image_datasets['val']), 2)))
    # print('Accuracy of the network on {} test images: {}%'.format(len(image_datasets['val']),
    #                                                               test_acc.double()*100.0/len(image_datasets['val']))) # added by Holy 2109060810
    # end of hide 2109080810
    # end of addition 2109041500

    """
    # ConvNet as fixed feature extractor
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(
        model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25)
    
    # save model_conv
    torch.save(model_conv.state_dict(), 'model_conv_weights.pth')

    visualize_model(model_conv)
    """
    plt.ioff()
    plt.show()
