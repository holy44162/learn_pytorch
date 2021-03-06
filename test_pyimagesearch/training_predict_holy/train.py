# usage example: python train.py --model output/model.pth --plot output/plot.png
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

# added by Holy 2109070810
from pyimagesearch.shufflenetv2_pytorch import ShuffleNetV2
from torchvision import datasets, models, transforms
import os
import torch.optim as optim
# end of addition 2109070810

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, required=True,
        help="path to output trained model")
    ap.add_argument("-p", "--plot", type=str, required=True,
        help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())

    # define training hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 2**6
    EPOCHS = 5

    # define the train and val splits
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 1 - TRAIN_SPLIT

    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hided by Holy 2109081500
    """
    # load the KMNIST dataset
    print("[INFO] loading the KMNIST dataset...")
    trainData = KMNIST(root="data", train=True, download=True,
        transform=ToTensor())
    testData = KMNIST(root="data", train=False, download=True,
        transform=ToTensor())

    # calculate the train/validation split
    print("[INFO] generating the train/validation split...")
    numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
    numValSamples = int(len(trainData) * VAL_SPLIT)
    (trainData, valData) = random_split(trainData,
        [numTrainSamples, numValSamples],
        generator=torch.Generator().manual_seed(42))

    # initialize the train, validation, and test data loaders
    trainDataLoader = DataLoader(trainData, shuffle=True,
        batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
    testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE
    """
    # end of hide 2109081500

    # added by Holy 2109070810
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'e:/dnn_data/z75_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x])
                        for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                    shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    trainDataLoader = dataloaders['train']
    valDataLoader = dataloaders['val']
    testDataLoader = dataloaders['val']

    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE

    # tested by Holy 2109070810
    print(class_names)
    print(dataset_sizes)
    images, labels = next(iter(dataloaders['train']))
    print(images.shape)
    print(labels.shape)
    # end of test 2109070810

    # end of addition 2109070810

    # initialize the LeNet model
    print("[INFO] initializing the model...")
    # hided by Holy 2109081500
    # model = LeNet(
    #     numChannels=1,
    #     classes=len(trainData.dataset.classes)).to(device)
    # end of hide 2109081500
    
    model = ShuffleNetV2(
        [4, 8, 4], [24, 116, 232, 464, 1024],
        num_classes=2).to(device)  # added by Holy 2109070810

    # initialize our optimizer and loss function
    # opt = Adam(model.parameters(), lr=INIT_LR)
    # lossFn = nn.NLLLoss()

    # added by Holy 2109071500
    opt = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9)
    lossFn = nn.CrossEntropyLoss()
    # end of addition 2109071500

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()

    # loop over our epochs
    for e in range(0, EPOCHS):
        # set the model in training mode
        model.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0

        # loop over the training set
        for (x, y) in trainDataLoader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))

            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFn(pred, y)

            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()
        
        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()

            # loop over the validation set
            for (x, y) in valDataLoader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += lossFn(pred, y)

                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))

    # finish measuring how long training took
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # we can now evaluate the network on the test set
    print("[INFO] evaluating network...")

    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        
        # initialize a list to store our predictions
        preds = []

        # loop over the test set
        for (x, y) in testDataLoader:
            # send the input to the device
            x = x.to(device)

            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

    # generate a classification report
    # print(classification_report(testData.targets.cpu().numpy(),
    #     np.array(preds), target_names=testData.classes)) # hided by Holy 2109071500

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
    plt.savefig(args["plot"])

    # serialize the model to disk
    torch.save(model, args["model"])
