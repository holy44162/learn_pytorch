# example: python run_training_A_plus_shufflenet_v2.py --model_dir models_A_plus --head_layer 2 --cuda 1 --epochs 100
# head dims:512,512,512,512,512,512,512,512,128
# code is basicly:https://github.com/google-research/deep_representation_one_class
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
# from dataset import MVTecAT_A, Repeat # added by Holy 2111090810
# from dataset import MVTecAT_B, Repeat # added by Holy 2111090810
# from dataset import MVTecAT_C, Repeat # added by Holy 2111090810
from dataset import normal_mess_data, Repeat # added by Holy 2111090810
from cutpaste_A import CutPasteNormal,CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
from model_winding import ProjectionNet_winding
# from eval import eval_model
from eval_shufflenet_v2 import eval_model # added by Holy 2111091500
from utils import str2bool

# added by Holy 2111090810
import albumentations as A
from albumentations.pytorch import ToTensorV2
# end of addition 2111090810

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
                 size = 224):
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

    # create Training Dataset and Dataloader
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose([])
    #train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
    train_transform.transforms.append(transforms.Resize((size,size)))
    train_transform.transforms.append(cutpate_type(transform = after_cutpaste_transform))
    # train_transform.transforms.append(transforms.ToTensor())

    # added by Holy 2111090810
    # train_transform_A = A.Resize(size,size)
    # train_transform_B = cutpate_type(transform =ToTensorV2())

    after_cutpaste_transform_A = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ToTensorV2()])
    # train_transform_B1 = A.Compose([A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    #                              A.Resize(size,size)])
    train_transform_B1 = A.Compose([
        A.Resize(size,size),
        A.OneOf([
                A.HorizontalFlip(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
                ], p=1)
    ])
    train_transform_B2 = cutpate_type(transform = after_cutpaste_transform_A)
    # end of addition 2111090810

    # train_data = MVTecAT("Data", data_type, transform = train_transform, size=int(size * (1/min_scale)))
    # train_data = MVTecAT("d:/temp/mvtec_anomaly_detection", data_type, transform = train_transform, size=int(size * (1/min_scale))) # added by Holy 2111020810
    # train_data = MVTecAT_A("d:/temp/mvtec_anomaly_detection", data_type, transform = train_transform_A, size=int(size * (1/min_scale))) # added by Holy 2111020810
    # train_data = MVTecAT_B("d:/temp/mvtec_anomaly_detection", data_type, transform = train_transform_B, size=int(size * (1/min_scale))) # added by Holy 2111020810
    # train_data = MVTecAT_C("d:/temp/mvtec_anomaly_detection", data_type, transform = [train_transform_B1, train_transform_B2], size=int(size * (1/min_scale))) # added by Holy 2111091500
    train_data = normal_mess_data("e:/dnn_data/ZTC950V763_data", data_type, transform = [train_transform_B1, train_transform_B2], size=int(size * (1/min_scale))) # added by Holy 2111091500
    print('train_data length:(total number of images) ', len(train_data))
    print('type(train_data):', type(train_data))

    # added by Holy 2111090810
    train_sample = train_data[0]
    repeat_dataset = Repeat(train_data, 3000)
    # end of addition 2111090810

    # print('train_data[0] length:(3) ', len(train_data[0]))
    # print('type(train_data[0]):', type(train_data[0]))
    # print('train_data[0][0] length:(3) ', len(train_data[0][0]))
    # print('type(train_data[0][0]):', type(train_data[0][0]))
    # print('train_data[0][0][0] length:(256) ', len(train_data[0][0][0]))
    # print('type(train_data[0][0][0]):', type(train_data[0][0][0]))
    # print('train_data[0][0][0][0] length:(256) ', len(train_data[0][0][0][0]))
    # print('type(train_data[0][0][0][0]):', type(train_data[0][0][0][0]))
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

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(Path("logdirs_A_plus") / model_name)

    # create Model:
    head_layers = [512]*head_layer+[128]
    num_classes = 2 if cutpate_type is not CutPaste3Way else 3
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
        # optimizer = optim.Adam(model.parameters(), lr=learninig_rate, weight_decay=weight_decay)
        # scheduler = None

        # added by Holy 2111111500
        INIT_LR = 1e-3
        BATCH_SIZE = 2**7
        EPOCHS = 100
        STEP_SIZE = 7
        GAMMA = 0.1
        optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        # end of addition 2111111500
    else:
        print(f"ERROR unkown optimizer: {optim_name}")

    step = 0
    num_batches = len(dataloader)
    print('num_batches: ', num_batches) # tested by Holy 2111020810
    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
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
        
        # if scheduler is not None:
        #     scheduler.step(epoch)
        # added by Holy 2111111500
        if optim_name == "sgd":
            if scheduler is not None:
                scheduler.step(epoch)
        else:
            if scheduler is not None:
                scheduler.step()
        # end of addition 2111111500

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

        # hided by Holy 2111111500
        # # run tests
        # if test_epochs > 0 and epoch % test_epochs == 0:
        #     # run auc calculation
        #     #TODO: create dataset only once.
        #     #TODO: train predictor here or in the model class itself. Should not be in the eval part
        #     #TODO: we might not want to use the training datat because of droupout etc. but it should give a indecation of the model performance???
        #     # batch_embeds = torch.cat(batch_embeds)
        #     # print(batch_embeds.shape)
        #     model.eval()
        #     roc_auc= eval_model(model_name, data_type, device=device,
        #                         save_plots=False,
        #                         size=size,
        #                         show_training_data=False,
        #                         model=model)
        #                         #train_embed=batch_embeds)
        #     model.train()
        #     writer.add_scalar('eval_auc', roc_auc, step)
        # end of hide 2111111500


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
    all_types = ['winding']
    
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
                     workers=args.workers,
                     size = 224)
