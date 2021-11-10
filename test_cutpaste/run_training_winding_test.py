# head dims:512,512,512,512,512,512,512,512,128
# code is basicly:https://github.com/google-research/deep_representation_one_class
# example: python run_training_winding.py --model_dir models_winding --head_layer 2 --cuda 1 --epochs 2
from pathlib import Path
from tqdm import tqdm
import datetime
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# from dataset import MVTecAT, Repeat
# added by Holy 2111020810
from dataset_winding import winding_data, Repeat, normal_mess_data, CatsVsDogsDataset_holy
import os
# end of addition 2111020810

from cutpaste import CutPasteNormal,CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
from model_winding import ProjectionNet_winding
from eval_winding import eval_model
from utils import str2bool

# added by Holy 2111040810
import albumentations as A
from albumentations.pytorch import ToTensorV2
# end of addition 2111040810

import random
from urllib.request import urlretrieve
from tqdm import tqdm
import shutil
import cv2
import numpy as np

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print("Filepath already exists. Skipping download.")
        return

    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(filepath)) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    shutil.unpack_archive(filepath, extract_dir)

def run_training(data_type="screw",
                 model_dir="models",
                 epochs=256,
                 pretrained=True,
                 test_epochs=10,
                 freeze_resnet=20,
                 learninig_rate=0.03,
                 optim_name="SGD",
                 batch_size=64,
                 head_layer=8,
                 cutpate_type=CutPasteNormal,
                 device = "cuda",
                 workers=8,
                 size = 256):
    torch.multiprocessing.freeze_support()
    # TODO: use script params for hyperparameter
    # Temperature Hyperparameter currently not used
    temperature = 0.2

    weight_decay = 0.00003
    momentum = 0.9
    #TODO: use f strings also for the date LOL
    model_name = f"model-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now() )

    #augmentation:
    min_scale = 1

    # hided by Holy 2111040810
    # # create Training Dataset and Dataloader
    # after_cutpaste_transform = transforms.Compose([])
    # after_cutpaste_transform.transforms.append(transforms.ToTensor())
    # after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                 std=[0.229, 0.224, 0.225]))

    # train_transform = transforms.Compose([])
    # #train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    # train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
    # train_transform.transforms.append(transforms.Resize((size,size)))
    # train_transform.transforms.append(cutpate_type(transform = after_cutpaste_transform))
    # # train_transform.transforms.append(transforms.ToTensor())
    # end of hide 2111040810

    # added by Holy 2111040810
    # create Training Dataset and Dataloader
    after_cutpaste_transform = A.Compose([ToTensorV2(),
                                          A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_transform = A.Compose([A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                 A.Resize(size,size),
                                 cutpate_type(transform = after_cutpaste_transform)])
    
    train_transform_test1 = A.Compose([A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                 A.Resize(size,size)])
    # train_transform_test2 = cutpate_type(transform = after_cutpaste_transform)
    train_transform_test2 = cutpate_type(transform = A.Resize(size,size))
    # end of addition 2111040810

    # hided by Holy 2111021500
    # # train_data = MVTecAT("Data", data_type, transform = train_transform, size=int(size * (1/min_scale)))
    # train_data = winding_data("d:/temp/winding_data", data_type, transform = train_transform, size=int(size * (1/min_scale))) # added by Holy 2111020810
    # dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True,
    #                         shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn,
    #                         persistent_workers=True, pin_memory=True, prefetch_factor=5)
    # end of hide 2111021500
    
    # added by Holy 2111021500
    data_dir = 'e:/dnn_data/ZTC950V763_data'
    # data_dir = 'e:/dnn_data/ZTC950V763_data/train/normal'
    train_data = normal_mess_data(
        data_dir, transform=[train_transform_test1,train_transform_test2], size=int(size * (1/min_scale)))
    print('train_data length:(total number of images) ', len(train_data))
    print('train_data[0] length:(3) ', len(train_data[0]))
    print(type(train_data[0][0]))
    print('train_data[0][0] shape:(3) ', np.array(train_data[0][0]).shape)

    """
    dataset_directory = 'd:/data_seq/cats-vs-dogs'
    root_directory = os.path.join(dataset_directory, "PetImages")

    filepath = os.path.join(dataset_directory, "kagglecatsanddogs_3367a.zip")
    download_url(
        url="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip",
        filepath=filepath,
    )

    if not os.path.isdir(root_directory):
        print('extracting...')
        extract_archive(filepath)
    
    correct_imgs_filepaths_fpname = os.path.join(dataset_directory, 'correct_images_filepaths.txt')

    if os.path.isfile(correct_imgs_filepaths_fpname):
        correct_images_filepaths = []
        with open(correct_imgs_filepaths_fpname, 'r') as filehandle:
            correct_images_filepaths = [current_place.rstrip() for current_place in filehandle.readlines()]
    else:    
        cat_directory = os.path.join(root_directory, "Cat")
        dog_directory = os.path.join(root_directory, "Dog")

        cat_images_filepaths = sorted([os.path.join(cat_directory, f) for f in os.listdir(cat_directory)])
        dog_images_filepaths = sorted([os.path.join(dog_directory, f) for f in os.listdir(dog_directory)])
        images_filepaths = [*cat_images_filepaths, *dog_images_filepaths]
        correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]
        
        with open(correct_imgs_filepaths_fpname, 'wt') as fin:
            for correct_images_filepath in correct_images_filepaths:
                fin.write("%s\n" % correct_images_filepath)

    random.seed(42)
    random.shuffle(correct_images_filepaths)
    train_images_filepaths = correct_images_filepaths[:20000]
    train_transform1 = A.Compose(
        [
            A.SmallestMaxSize(max_size=160),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=128, width=128),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    train_transform2 = [train_transform1, cutpate_type(transform = A.Resize(size,size))]
    train_data = CatsVsDogsDataset_holy(images_filepaths=train_images_filepaths, transform=train_transform2)
    print('train_data length:(total number of images) ', len(train_data))
    print('train_data[0] length:(3) ', len(train_data[0]))
    print(type(train_data[0][0]))
    print('train_data[0][0] shape:(3) ', np.array(train_data[0][0]).shape)
    # print('train_data[0][0] length:(3) ', len(train_data[0][0]))
    # print('train_data[0][0][0] length:(256) ', len(train_data[0][0][0]))
    # print('train_data[0][0][0][0] length:(256) ', len(train_data[0][0][0][0]))
    """

    dataloader = DataLoader(Repeat(train_data, 5000), batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn,
                            persistent_workers=True, pin_memory=True, prefetch_factor=5)
    for i_batch, sample_batched in enumerate(dataloader):
        print('i_batch: ', i_batch)
        print('sample_batched length: (3)', len(sample_batched))
        print('sample_batched[0] size: (64x3x256x256)', sample_batched[0].size())
        print('sample_batched[1] size: (64x3x256x256)', sample_batched[1].size())
        print('sample_batched[2] size: (64x3x256x256)', sample_batched[2].size())
        break
    # end of addition 2111021500

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(Path("logdirs_winding") / model_name)

    # create Model:
    head_layers = [512]*head_layer+[128]
    num_classes = 2 if cutpate_type is not CutPaste3Way else 3
    # model = ProjectionNet(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)
    model = ProjectionNet_winding(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # added by Holy 2111020810
    model.to(device)

    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()

    loss_fn = torch.nn.CrossEntropyLoss()
    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learninig_rate, momentum=momentum,  weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
        #scheduler = None
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learninig_rate, weight_decay=weight_decay)
        scheduler = None
    else:
        print(f"ERROR unkown optimizer: {optim_name}")

    step = 0
    # num_batches = len(dataloader)
    num_batches = len(dataloader) # added by Holy 2111021500

    # tested by Holy 2111020810
    # print('num_batches: ', num_batches)
    # print('type of dataloader: ', type(dataloader))
    # print('type of train_data: ', type(train_data))
    # print('first data of dataloader: ', dataloader)    
    # end of test 2111020810

    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                # print(out) # tested by Holy 2111041500
                yield out
    dataloader_inf =  get_data_inf()
    # From paper: "Note that, unlike conventional definition for an epoch,
    #              we define 256 parameter update steps as one epoch.
    for step in tqdm(range(epochs)):
        epoch = int(step / 1)
        if epoch == freeze_resnet:
            model.unfreeze()
        
        batch_embeds = []
        batch_idx, data = next(dataloader_inf)
        xs = [x.to(device) for x in data]

        # zero the parameter gradients
        optimizer.zero_grad()

        xc = torch.cat(xs, axis=0)
        embeds, logits = model(xc)
        
#         embeds = F.normalize(embeds, p=2, dim=1)
#         embeds1, embeds2 = torch.split(embeds,x1.size(0),dim=0)
#         ip = torch.matmul(embeds1, embeds2.T)
#         ip = ip / temperature

#         y = torch.arange(0,x1.size(0), device=device)
#         loss = loss_fn(ip, torch.arange(0,x1.size(0), device=device))

        # calculate label
        y = torch.arange(len(xs), device=device)
        y = y.repeat_interleave(xs[0].size(0))
        loss = loss_fn(logits, y)
        

        # regulize weights:
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)
        
        writer.add_scalar('loss', loss.item(), step)
        
#         predicted = torch.argmax(ip,axis=0)
        predicted = torch.argmax(logits,axis=1)
#         print(logits)
#         print(predicted)
#         print(y)
        accuracy = torch.true_divide(torch.sum(predicted==y), predicted.size(0))
        writer.add_scalar('acc', accuracy, step)
        if scheduler is not None:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)
        
        # save embed for validation:
        if test_epochs > 0 and epoch % test_epochs == 0:
            batch_embeds.append(embeds.cpu().detach())

        writer.add_scalar('epoch', epoch, step)

        # run tests
        if test_epochs > 0 and epoch % test_epochs == 0:
            # run auc calculation
            #TODO: create dataset only once.
            #TODO: train predictor here or in the model class itself. Should not be in the eval part
            #TODO: we might not want to use the training datat because of droupout etc. but it should give a indecation of the model performance???
            # batch_embeds = torch.cat(batch_embeds)
            # print(batch_embeds.shape)
            model.eval()
            roc_auc= eval_model(model_name, data_type, device=device,
                                save_plots=False,
                                size=size,
                                show_training_data=False,
                                model=model)
                                #train_embed=batch_embeds)
            model.train()
            writer.add_scalar('eval_auc', roc_auc, step)


    torch.save(model.state_dict(), model_dir / f"{model_name}.tch")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--epochs', default=256, type=int,
                        help='number of epochs to train the model , (default: 256)')
    
    parser.add_argument('--model_dir', default="models",
                        help='output folder of the models , (default: models)')
    
    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',
                        help='use pretrained values to initalize ResNet18 , (default: True)')
    
    parser.add_argument('--test_epochs', default=10, type=int,
                        help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')                  

    parser.add_argument('--freeze_resnet', default=20, type=int,
                        help='number of epochs to freeze resnet (default: 20)')
    
    parser.add_argument('--lr', default=0.03, type=float,
                        help='learning rate (default: 0.03)')

    parser.add_argument('--optim', default="sgd",
                        help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')   

    parser.add_argument('--head_layer', default=1, type=int,
                    help='number of layers in the projection head (default: 1)')
    
    parser.add_argument('--variant', default="3way", choices=['normal', 'scar', '3way', 'union'], help='cutpaste variant to use (dafault: "3way")')
    
    parser.add_argument('--cuda', default=False, type=str2bool,
                    help='use cuda for training (default: False)')
    
    parser.add_argument('--workers', default=8, type=int, help="number of workers to use for data loading (default:8)")


    args = parser.parse_args()
    print(args)
    all_types = ['ZTC950V763_data']
    
    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")
    
    variant_map = {'normal':CutPasteNormal, 'scar':CutPasteScar, '3way':CutPaste3Way, 'union':CutPasteUnion}
    variant = variant_map[args.variant]
    
    device = "cuda" if args.cuda else "cpu"
    print(f"using device: {device}")
    
    # create modle dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))

    for data_type in types:
        print(f"training {data_type}")
        run_training(data_type,
                     model_dir=Path(args.model_dir),
                     epochs=args.epochs,
                     pretrained=args.pretrained,
                     test_epochs=args.test_epochs,
                     freeze_resnet=args.freeze_resnet,
                     learninig_rate=args.lr,
                     optim_name=args.optim,
                     batch_size=args.batch_size,
                     head_layer=args.head_layer,
                     device=device,
                     cutpate_type=variant,
                     workers=args.workers)
