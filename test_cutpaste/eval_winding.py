# example: python eval_winding.py --model_dir models_winding --head_layer 2
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from dataset_winding import normal_mess_data
from cutpaste import CutPaste
from model_winding import ProjectionNet_winding
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from cutpaste import CutPaste, cut_paste_collate_fn
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import numpy as np
from collections import defaultdict
from density import GaussianDensitySklearn, GaussianDensityTorch
import pandas as pd
from utils import str2bool

# added by Holy 2111040810
import albumentations as A
from albumentations.pytorch import ToTensorV2
# end of addition 2111040810

test_data_eval = None
test_transform = None
cached_type = None

def get_train_embeds(model, size, defect_type, transform, device):
    # train data / train kde
    # # test_data = MVTecAT("Data", defect_type, size, transform=transform, mode="train")
    # test_data = winding_data("d:/temp/winding_data", defect_type, size, transform=transform, mode="train") # added by Holy 2111020810

    # dataloader_train = DataLoader(test_data, batch_size=64,
    #                         shuffle=False, num_workers=0)
    
    # added by Holy 2111021500
    data_dir = 'e:/dnn_data/ZTC950V763_data'
    test_data = normal_mess_data(
        data_dir, size, transform=transform, mode="train")
    dataloader_train = DataLoader(test_data, batch_size=64,
                                  shuffle=False, num_workers=0)
    # end of addition 2111021500

    train_embed = []
    with torch.no_grad():
        for x in dataloader_train:
            embed, logit = model(x.to(device))

            train_embed.append(embed.cpu())
    train_embed = torch.cat(train_embed)
    return train_embed

def eval_model(modelname, defect_type, device="cpu", save_plots=False, size=256, show_training_data=True, model=None, train_embed=None, head_layer=8, density=GaussianDensityTorch()):
    # create test dataset
    global test_data_eval,test_transform, cached_type
    data_dir = 'e:/dnn_data/ZTC950V763_data' # added by Holy 2111030810

    # TODO: cache is only nice during training. do we need it?
    if test_data_eval is None or cached_type != defect_type:
        cached_type = defect_type

        # hided by Holy 2111041500
        # test_transform = transforms.Compose([])
        # test_transform.transforms.append(transforms.Resize((size,size)))
        # test_transform.transforms.append(transforms.ToTensor())
        # test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                                     std=[0.229, 0.224, 0.225]))
        # end of hide 2111041500

        # added by Holy 2111040810
        # create Training Dataset and Dataloader
        test_transform = A.Compose([A.Resize(size,size),
                                    ToTensorV2(),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # end of addition 2111040810

        # test_data_eval = MVTecAT("Data", defect_type, size, transform = test_transform, mode="test")
        # test_data_eval = winding_data("d:/temp/winding_data", defect_type, size, transform = test_transform, mode="test") # added by Holy 2111020810
        test_data_eval = normal_mess_data(data_dir, size, transform = test_transform, mode="test") # added by Holy 2111030810

    dataloader_test = DataLoader(test_data_eval, batch_size=64,
                                    shuffle=False, num_workers=0)

    # create model
    if model is None:
        print(f"loading model {modelname}")
        head_layers = [512]*head_layer+[128]
        print(head_layers)
        weights = torch.load(modelname)
        classes = weights["out.weight"].shape[0]
        # model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        model = ProjectionNet_winding(pretrained=False, head_layers=head_layers, num_classes=classes)
        model.load_state_dict(weights)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # added by Holy 2111020810
        model.to(device)
        model.eval()

    #get embeddings for test data
    labels = []
    embeds = []
    with torch.no_grad():
        for x, label in dataloader_test:
            embed, logit = model(x.to(device))

            # save 
            embeds.append(embed.cpu())
            labels.append(label.cpu())
    labels = torch.cat(labels)
    embeds = torch.cat(embeds)

    if train_embed is None:
        train_embed = get_train_embeds(model, size, defect_type, test_transform, device)

    # norm embeds
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

    #create eval plot dir
    if save_plots:
        eval_dir = Path("eval_winding") / modelname
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # plot tsne
        # also show some of the training data
        show_training_data = False
        if show_training_data:
            #augmentation setting
            # TODO: do all of this in a separate function that we can call in training and evaluation.
            #       very ugly to just copy the code lol
            min_scale = 0.5

            # hided by Holy 2111041500
            # # create Training Dataset and Dataloader
            # after_cutpaste_transform = transforms.Compose([])
            # after_cutpaste_transform.transforms.append(transforms.ToTensor())
            # after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                                                 std=[0.229, 0.224, 0.225]))

            # train_transform = transforms.Compose([])
            # #train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
            # #train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
            # train_transform.transforms.append(CutPaste(transform=after_cutpaste_transform))
            # # train_transform.transforms.append(transforms.ToTensor())
            # end of hide 2111041500

            # added by Holy 2111040810
            # create Training Dataset and Dataloader
            after_cutpaste_transform = A.Compose([ToTensorV2(),
                                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            train_transform = A.Compose([CutPaste(transform = after_cutpaste_transform)])    
            # end of addition 2111040810

            # train_data = MVTecAT("Data", defect_type, transform=train_transform, size=size)
            # train_data = winding_data("d:/temp/winding_data", defect_type, transform=train_transform, size=size) # added by Holy 2111020810
            train_data = normal_mess_data(data_dir, transform=train_transform, size=size) # added by Holy 2111030810
            dataloader_train = DataLoader(train_data, batch_size=32,
                        shuffle=True, num_workers=8, collate_fn=cut_paste_collate_fn,
                        persistent_workers=True)
            # inference training data
            train_labels = []
            train_embeds = []
            with torch.no_grad():
                for x1, x2 in dataloader_train:
                    x = torch.cat([x1,x2], axis=0)
                    embed, logit = model(x.to(device))

                    # generate labels:
                    y = torch.tensor([0, 1])
                    y = y.repeat_interleave(x1.size(0))

                    # save 
                    train_embeds.append(embed.cpu())
                    train_labels.append(y)
                    # only less data
                    break
            train_labels = torch.cat(train_labels)
            train_embeds = torch.cat(train_embeds)

            # for tsne we encode training data as 2, and augmentet data as 3
            tsne_labels = torch.cat([labels, train_labels + 2])
            tsne_embeds = torch.cat([embeds, train_embeds])
        else:
            tsne_labels = labels
            tsne_embeds = embeds
        plot_tsne(tsne_labels, tsne_embeds, eval_dir / "tsne.png")
    else:
        eval_dir = Path("unused")
    
    print(f"using density estimation {density.__class__.__name__}")
    density.fit(train_embed)
    distances = density.predict(embeds)
    #TODO: set threshold on mahalanobis distances and use "real" probabilities

    roc_auc = plot_roc(labels, distances, eval_dir / "roc_plot.png", modelname=modelname, save_plots=save_plots)
    

    return roc_auc
    

def plot_roc(labels, scores, filename, modelname="", save_plots=False):

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    #plot roc
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc

def plot_tsne(labels, embeds, filename):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    embeds, labels = shuffle(embeds, labels)
    tsne_results = tsne.fit_transform(embeds)
    fig, ax = plt.subplots(1)
    colormap = ["b", "r", "c", "y"]

    ax.scatter(tsne_results[:,0], tsne_results[:,1], color=[colormap[l] for l in labels])
    fig.savefig(filename)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--model_dir', default="models",
                    help=' directory contating models to evaluate (default: models)')
    
    parser.add_argument('--cuda', default=False, type=str2bool,
                    help='use cuda for model predictions (default: False)')

    parser.add_argument('--head_layer', default=8, type=int,
                    help='number of layers in the projection head (default: 8)')

    parser.add_argument('--density', default="torch", choices=["torch", "sklearn"],
                    help='density implementation to use. See `density.py` for both implementations. (default: torch)')

    parser.add_argument('--save_plots', default=True, type=str2bool,
                    help='save TSNE and roc plots')
    

    args = parser.parse_args()

    args = parser.parse_args()
    print(args)
    all_types = ['ZTC950V763_data']
    
    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")
    
    device = "cuda" if args.cuda else "cpu"

    density_mapping = {
        "torch": GaussianDensityTorch,
        "sklearn": GaussianDensitySklearn
    }
    density = density_mapping[args.density]

    # find models
    model_names = [list(Path(args.model_dir).glob(f"model-{data_type}*"))[0] for data_type in types if len(list(Path(args.model_dir).glob(f"model-{data_type}*"))) > 0]
    if len(model_names) < len(all_types):
        print("warning: not all types present in folder")

    obj = defaultdict(list)
    for model_name, data_type in zip(model_names, types):
        print(f"evaluating {data_type}")

        roc_auc = eval_model(model_name, data_type, save_plots=args.save_plots, device=device, head_layer=args.head_layer, density=density())
        print(f"{data_type} AUC: {roc_auc}")
        obj["defect_type"].append(data_type)
        obj["roc_auc"].append(roc_auc)
    
    # save pandas dataframe
    eval_dir = Path("eval_winding") / args.model_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(obj)
    df.to_csv(str(eval_dir) + "_perf.csv")
