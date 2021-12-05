from __future__ import print_function, division

import torch
from shutil import copyfile
from sys import exit
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
import torch.onnx as onnx
import onnx as onnx_origin
from onnxsim import simplify
import onnxruntime as rt
import configparser
import subprocess, sys
import glob

import cv2

# added by Holy 2111170810
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.covariance import LedoitWolf
from sklearn.metrics import precision_recall_curve

class GaussianDensityTorch(object):
    """Gaussian Density estimation similar to the implementation used by Ripple et al.
    The code of Ripple et al. can be found here: https://github.com/ORippler/gaussian-ad-mvtec.
    """
    def fit(self, embeddings):
        self.mean = torch.mean(embeddings, axis=0)
        self.inv_cov = torch.Tensor(LedoitWolf().fit(embeddings.cpu()).precision_,device="cpu")

    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        return distances

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.

        from https://github.com/ORippler/gaussian-ad-mvtec/blob/4e85fb5224eee13e8643b684c8ef15ab7d5d016e/src/gaussian/model.py#L308
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()

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
# end of addition 2111170810

def list_files(directory, extension):
    return (f for f in sorted(os.listdir(directory)) if f.endswith('.' + extension))


def compute_f1(model_full, test_path, img_size, best_threshold, density):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = str(Path(test_path) / 'imgs')    

    strYTest = str(Path(test_path) / 'y_Test.txt')    

    with open(strYTest) as fInYTest:
        vecBMessYTest = fInYTest.readlines()
        vecBMessYTest = [bool(int(i)) for i in vecBMessYTest]        

    vecBMess = []    
    
    # specify ImageNet mean and standard deviation
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    

    str_imgs_list = list_files(path, "jpg")
    for str_img_name in str_imgs_list:
        str_img_name = str(Path(path) / str_img_name)
        
        img = cv2.imread(str_img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, img_size)       

        img = img.astype("float32") / 255.0
        img -= imagenet_mean
        img /= imagenet_std
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img)        
        img = img.unsqueeze(0)
        img = img.to(device) 
        # output = model_full.forward(img)

        # added by Holy 2111110810
        embed_img, logit = model_full.forward(img)
        embed_img = torch.nn.functional.normalize(embed_img.cpu(), p=2, dim=1)
        distances = density.predict(embed_img)
        val, cls = distances.item(), distances.item() >= best_threshold.astype(float)

        print(str_img_name) # added by Holy 2110260810
        print("[pytorch]--->predicted class:", cls)
        print("[pytorch]--->predicted value:", val)

        preds = int(cls)
        # end of addition 2111110810
        
        # # val, cls = torch.max(output.data, 1)
        # val, cls = torch.sigmoid(output), torch.sigmoid(output) >= 0.5
        
        # print(str_img_name) # added by Holy 2110260810
        # print("[pytorch]--->predicted class:", cls.item())
        # print("[pytorch]--->predicted value:", val.item())        

        # preds = 1 - int(cls.item())
        vecBMess.append(bool(preds))
    
    vecBMessYTest_flip = [ not z for z in vecBMessYTest]
    vecBMess_flip = [ not z for z in vecBMess]

    vecBResult = [ x and y for (x,y) in zip(vecBMessYTest, vecBMess)]    
    tp = sum(vecBResult)    

    vecBResult = [ x and y for (x,y) in zip(vecBMessYTest_flip, vecBMess)]
    fp = sum(vecBResult)    

    vecBResult = [ x and y for (x,y) in zip(vecBMessYTest, vecBMess_flip)]
    fn1 = sum(vecBResult)    

    prec = float(tp) / float(tp + fp)
    rec = float(tp) / float(tp + fn1)
    dF1 = 2 * prec * rec / (prec + rec)
    
    vecBResult = [ x == y for (x,y) in zip(vecBMessYTest, vecBMess)]    
    acc = float(sum(vecBResult)) / float(len(vecBResult))

    print(f'total frames: {len(vecBMess)}')
    total_frames_num = len(vecBMess)
    
    return dF1, tp, fp, fn1, prec, rec, acc, total_frames_num


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


def copy_file(src_file, dst_file):
    try:
        copyfile(src_file, dst_file)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)


def update_file(file_path_name, src_str, dst_str):
    #read input file    
    with open(file_path_name, "rt") as fin:
        #read file contents to string
        data = fin.read()
        #replace all occurrences of the required string
        data = data.replace(src_str, dst_str)
    
    #open the input file in write mode    
    with open(file_path_name, "wt") as fin:
        #overrite the input file with the resulting data
        fin.write(data)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # added by Holy 2111170810

    config = configparser.ConfigParser()
    config.read('d:/backup/project/learn_pytorch/test_cutpaste/automation_pytorch_config_oc.ini')
    # first trained model
    model_full_path = 'd:/backup/project/learn_pytorch/test_cutpaste/model_ft_shufflenet_v2.pth'

    if not os.path.exists(model_full_path):
        files_path = os.path.dirname(model_full_path)
        if not files_path:
            files_path = '.'
        
        file_type = '\*' + os.path.splitext(model_full_path)[1][1:]
        pth_files = glob.glob(files_path + file_type)
        model_full_path = max(pth_files, key=os.path.getmtime)

    test_path = config['train']['test_dataset_path']

    img_size1 = config['train']['img_size'].split(',')

    img_size = (int(img_size1[0]), int(img_size1[1]))

    model_full = torch.load(model_full_path)

    # added by Holy 2111170810
    density = GaussianDensityTorch()

    datasets_path = config['train']['datasets_path']

    batch_size = config['train']['batch_size']

    test_transform = A.Compose([A.Resize(img_size[0], img_size[1]),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(
                                    0.229, 0.224, 0.225)),
                                ToTensorV2()])
    
    test_data = normal_mess_data_test(datasets_path, transform=test_transform, mode="train")

    test_data_eval = normal_mess_data_test(datasets_path, transform = test_transform, mode="test")

    dataloader_train = DataLoader(test_data, batch_size=int(eval(batch_size)),
                                  shuffle=False, num_workers=0)

    dataloader_test = DataLoader(test_data_eval, batch_size=int(eval(batch_size)),
                                 shuffle=False, num_workers=0)

    best_threshold_pathname = 'd:/backup/project/learn_pytorch/test_cutpaste/best_threshold.npy'
    best_F1_Score_pathname = 'd:/backup/project/learn_pytorch/test_cutpaste/best_F1_Score.npy'
    mean_pathname = 'd:/backup/project/learn_pytorch/test_cutpaste/mean.pt'
    inv_cov_pathname = 'd:/backup/project/learn_pytorch/test_cutpaste/inv_cov.pt'

    if os.path.exists(best_threshold_pathname):
        print('npy files exist.')
        best_threshold = np.load(best_threshold_pathname)
        best_F1_Score = np.load(best_F1_Score_pathname)        
        density.mean = torch.load(mean_pathname)
        density.inv_cov = torch.load(inv_cov_pathname)

        # tested by Holy 2111181500
        mean_np = density.mean.numpy()
        inv_cov_np = density.inv_cov.numpy()
        np.save('d:/backup/project/learn_pytorch/test_cutpaste/mean.npy', mean_np)
        np.save('d:/backup/project/learn_pytorch/test_cutpaste/inv_cov.npy', inv_cov_np)
        # end of test 2111181500
    else:
        # get embeddings for test data
        labels = []
        embeds = []
        with torch.no_grad():
            for x, label in dataloader_test:
                embed, logit = model_full(x.to(device))

                # save 
                embeds.append(embed.cpu())
                labels.append(label.cpu())
        labels = torch.cat(labels)
        embeds = torch.cat(embeds)

        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)

        # get embeddings for train data
        train_embed = []
        with torch.no_grad():
            for x in dataloader_train:
                embed, logit = model_full(x.to(device))

                train_embed.append(embed.cpu())
        train_embed = torch.cat(train_embed)

        train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

        density.fit(train_embed)
        distances = density.predict(embeds)

        precision, recall, thresholds = precision_recall_curve(labels, distances)
        f1_scores = 2*recall*precision/(recall+precision)
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_F1_Score = np.max(f1_scores)

        torch.save(density.mean, mean_pathname)
        torch.save(density.inv_cov, inv_cov_pathname)

        np.save(best_threshold_pathname, best_threshold)
        np.save(best_F1_Score_pathname, best_F1_Score)
    # end of addition 2111170810

    tag_compute_f1 = True # added by Holy 2111180810
    if tag_compute_f1: # added by Holy 2111180810
        dF1, tp, fp, fn1, prec, rec, acc, total_frames_num = compute_f1(model_full, test_path, img_size, best_threshold, density)
    else: # added by Holy 2111180810
        dF1 = best_F1_Score # added by Holy 2111180810
    model_full_path_first = model_full_path[:-4] + '_' + '{:.2f}'.format(dF1) + '.pth'

    copy_file(model_full_path, model_full_path_first)

    # second trained checkpoint
    # pytorch_vision_version = 'pytorch/vision:' + config['train']['pytorch_vision_ver']
    # shufflenet_version = 'shufflenet_' + config['train']['shufflenet_version']
    # model_ft = torch.hub.load(pytorch_vision_version, shufflenet_version, pretrained=False) 
    
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 1)

    # added by Holy 2111170810
    # num_classes = 3
    num_classes = 4 # added by Holy 2112011315
    head_layer = 2
    head_layers = [512]*head_layer+[128]
    model_ft = ProjectionNet_winding(pretrained=False, head_layers=head_layers, num_classes=num_classes)
    # end of addition 2111170810

    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    INIT_LR = 1e-3

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=INIT_LR)
    
    best_model_path = 'd:/backup/project/learn_pytorch/test_cutpaste/best_model/best_model.pt'

    if not os.path.exists(best_model_path):
        files_path1 = os.path.dirname(best_model_path)
        if not files_path1:
            files_path1 = '.'
        
        file_type1 = '\*' + os.path.splitext(best_model_path)[1][1:]
        pth_files1 = glob.glob(files_path1 + file_type1)
        best_model_path = max(pth_files1, key=os.path.getmtime)
    
    # load the saved checkpoint
    model_ft, optimizer_ft, start_epochs, valid_loss_min = load_ckp(best_model_path, model_ft, optimizer_ft)

    model_ft.eval()

    if tag_compute_f1: # added by Holy 2111180810
        dF1_2, tp_2, fp_2, fn1_2, prec_2, rec_2, acc_2, total_frames_num = compute_f1(model_ft, test_path, img_size, best_threshold, density)
    else: # added by Holy 2111180810
        dF1_2 = best_F1_Score # added by Holy 2111180810
    # print(f'test results 1: f1:{dF1}, tp:{tp}, fp:{fp}, fn:{fn1}, prec:{prec}, rec:{rec}, acc:{acc}, total frames:{total_frames_num}')
    # print(f'test results 2: f1:{dF1_2}, tp:{tp_2}, fp:{fp_2}, fn:{fn1_2}, prec:{prec_2}, rec:{rec_2}, acc:{acc_2}, total frames:{total_frames_num}')
    # print('best_F1_Score: ', best_F1_Score)

    model_full_path_second = model_full_path[:-4] + '_ckp_' + '{:.2f}'.format(dF1_2) + '.pth'
    torch.save(model_ft, model_full_path_second)

    if dF1 > dF1_2:
        choosed_model_path = model_full_path_first
    else:
        choosed_model_path = model_full_path_second
    
    # added by Holy 2110280810
    latest_best_model_path = 'latest_best_shufflenet_model.pth'
    copy_file(choosed_model_path, latest_best_model_path)
    # end of addition 2110280810

    if tag_compute_f1: # added by Holy 2111180810
        print(f'first: f1:{dF1}, tp:{tp}, fp:{fp}, fn:{fn1}, prec:{prec}, rec:{rec}, acc:{acc}, total frames:{total_frames_num}')
        print(f'second: f1:{dF1_2}, tp:{tp_2}, fp:{fp_2}, fn:{fn1_2}, prec:{prec_2}, rec:{rec_2}, acc:{acc_2}, total frames:{total_frames_num}')
        print('best_F1_Score: ', best_F1_Score)

        # added by Holy 2109300810
        first_str = f'first: f1:{dF1}, tp:{tp}, fp:{fp}, fn:{fn1}, prec:{prec}, rec:{rec}, acc:{acc}'
        second_str = f'second: f1:{dF1_2}, tp:{tp_2}, fp:{fp_2}, fn:{fn1_2}, prec:{prec_2}, rec:{rec_2}, acc:{acc_2}'
        # end of addition 2109300810
    else: # added by Holy 2111180810
        first_str = 'first: ' # added by Holy 2111180810
        second_str = 'second: ' # added by Holy 2111180810
    
    # export to simplified onnx
    model_ft_full = torch.load(choosed_model_path)
    
    # export to onnx
    input_image = torch.zeros((1,3,img_size[0],img_size[1]))
    input_image = input_image.to(device)
    
    input_names = ["x"]
    output_names = ["y"]

    #convert pytorch to onnx
    onnx_model_pathname = choosed_model_path[:-4] + '.onnx'
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
    img_path = os.path.join(test_path, 'imgs/img00001.jpg')
    img = cv2.imread(img_path)
    # cv2.imshow('augmented',img)
    # cv2.waitKey(0)
    img = cv2.resize(img, img_size)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img1 = img.to(device)    

    # pytorch test
    model = torch.load(choosed_model_path)
    
    # output = model.forward(img1)
    
    # # val, cls = torch.max(output.data, 1)
    # val, cls = torch.sigmoid(output), torch.sigmoid(output) >= 0.5
    # print("[pytorch]--->predicted class:", cls.item())
    # print("[pytorch]--->predicted value:", val.item())

    # added by Holy 2111180810
    embed_img, logit = model.forward(img1)
    embed_img = torch.nn.functional.normalize(embed_img.cpu(), p=2, dim=1)
    distances = density.predict(embed_img)
    val, cls = distances.item(), distances.item() >= best_threshold.astype(float)
    
    print("[pytorch]--->predicted class:", cls)
    print("[pytorch]--->predicted value:", val)    
    # end of addition 2111180810

    #onnx test
    sess = rt.InferenceSession(onnx_model_pathname)
    
    x = "x"
    y = ["y"]
    output = sess.run(y, {x: img.numpy()})
    # # cls = np.argmax(output[0][0], axis=0)
    # # val = output[0][0][cls]
    # val, cls = 1 / (1 + np.exp(-1*output[0][0])), (1 / (1 + np.exp(-1*output[0][0]))) >= 0.5
    # print("[onnx]--->predicted class:", cls)
    # print("[onnx]--->predicted value:", val)

    # added by Holy 2111180810
    output_expand = np.expand_dims(output[0][0], axis=0)
    embed_img_onnx = torch.nn.functional.normalize(torch.from_numpy(output_expand), p=2, dim=1)
    # norm = np.linalg.norm(output[0][0])
    # embed_img_onnx = output[0][0]/norm
    distances_onnx = density.predict(embed_img_onnx)
    val_onnx, cls_onnx = distances_onnx.item(), distances_onnx.item() >= best_threshold.astype(float)
    print("[onnx]--->predicted class:", cls_onnx)
    print("[onnx]--->predicted value:", val_onnx)
    # end of addition 2111180810
    
    #simplified onnx test
    sess = rt.InferenceSession(onnx_model_simplified_pathname)
    
    x = "x"
    y = ["y"]
    output = sess.run(y, {x: img.numpy()})
    # # cls = np.argmax(output[0][0], axis=0)
    # # val = output[0][0][cls]
    # val, cls = 1 / (1 + np.exp(-1*output[0][0])), (1 / (1 + np.exp(-1*output[0][0]))) >= 0.5
    # print("[onnx_simplified]--->predicted class:", cls)
    # print("[onnx_simplified]--->predicted value:", val)

    # added by Holy 2111180810
    output_expand = np.expand_dims(output[0][0], axis=0)
    embed_img_onnx = torch.nn.functional.normalize(torch.from_numpy(output_expand), p=2, dim=1)
    # norm = np.linalg.norm(output[0][0])
    # embed_img_onnx = output[0][0]/norm
    distances_onnx = density.predict(embed_img_onnx)
    val_onnx, cls_onnx = distances_onnx.item(), distances_onnx.item() >= best_threshold.astype(float)
    print("[onnx_simplified]--->predicted class:", cls_onnx)
    print("[onnx_simplified]--->predicted value:", val_onnx)
    # end of addition 2111180810

    # copy ps1 file
    src_ps1_file = './templates/onnx_to_ncnn_shufflenet_v2_one_class.ps1'
    dst_ps1_file = './onnx_to_ncnn_shufflenet_v2_one_class.ps1'
    copy_file(src_ps1_file, dst_ps1_file)

    # update ps1 file
    src_ps1_str = 'onnx2ncnn.exe'
    dst_ps1_str = os.path.join(config['train']['onnx2ncnn_path'], 'onnx2ncnn.exe').replace('\\', '/')
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    src_ps1_str = 'model_shufflenet_v2_simplified.onnx'
    dst_ps1_str = onnx_model_simplified_pathname.replace('\\', '/')
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    src_ps1_str = 'shufflenet_v2_x1_0.param'
    dst_ps1_str = onnx_model_pathname[:-5] + '.param'
    dst_ps1_str = dst_ps1_str.replace('\\', '/')
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)
    param_filename = os.path.basename(dst_ps1_str)
    param_pathname = dst_ps1_str
    dst_pytorch_param_path_name = os.path.join(config['train']['cpp_project_path'],
                                             'support/holyMess_one_class/'+param_filename).replace('\\', '/')

    src_ps1_str = 'shufflenet_v2_x1_0.bin'
    dst_ps1_str = onnx_model_pathname[:-5] + '.bin'
    dst_ps1_str = dst_ps1_str.replace('\\', '/')
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)
    bin_filename = os.path.basename(dst_ps1_str)
    bin_pathname = dst_ps1_str
    dst_pytorch_bin_path_name = os.path.join(config['train']['cpp_project_path'],
                                             'support/holyMess_one_class/'+bin_filename).replace('\\', '/')

    # run ps1
    p = subprocess.Popen(['pwsh.exe', dst_ps1_file], stdout=sys.stdout)
    p.communicate()
    
    # update cpp project
    str_time = '2110281054'

    src_cnn_param_path_name = os.path.join(config['train']['cpp_project_path'],
                                           'support/holyMess_one_class/holy_one_class_param.ini').replace('\\', '/')
    dst_cnn_param_path_name = os.path.join(config['train']['cpp_project_path'],
                                           'support/holyMess_one_class/holy_one_class_param_'+str_time+'.ini').replace('\\', '/')

    src_param_path_name = os.path.join(config['train']['cpp_project_path'],
                                       'param_'+config['train']['model_tag'] +
                                       '.ini').replace('\\', '/')
    src_param_path_name_origin = os.path.join(config['train']['cpp_project_path'],
                                       'param.ini').replace('\\', '/')
    dst_param_path_name = os.path.join(config['train']['cpp_project_path'],
                                       'param_' + config['train']['model_tag'] +
                                       '_' + str_time + '.ini').replace('\\', '/')
    
    src_param_tmp_path_name = os.path.join(config['train']['cpp_project_path'],
                                       'param_tmp.ini').replace('\\', '/')
    
    with open(src_param_tmp_path_name) as f_cnn_param_tmp:
        cnn_param_tmp_lines = f_cnn_param_tmp.readlines()
        cnn_param_tmp_lines[0] = 'AlgorithmNO:5' + '\n'
        cnn_param_tmp_lines[1] = 'windingType:' + \
            config['train']['model_tag'] + '\n'
    with open(src_param_tmp_path_name, 'w') as f_cnn_param_tmp_w:
        f_cnn_param_tmp_w.writelines(cnn_param_tmp_lines)
    
    if os.path.exists(src_cnn_param_path_name):
        copy_file(src_cnn_param_path_name, dst_cnn_param_path_name)
    
    with open(src_cnn_param_path_name) as f_cnn_param:
        cnn_param_lines = f_cnn_param.readlines()
        cnn_param_lines[1] = 'holy_net = ./support/holyMess_one_class/' + \
            bin_filename + '\n'
        cnn_param_lines[2] = 'holy_param = ./support/holyMess_one_class/' + \
            param_filename + '\n'
        cnn_param_lines[3] = 'netW = ' + img_size1[0] + '\n'
        cnn_param_lines[4] = 'netH =' + img_size1[1] + '\n'
    with open(src_cnn_param_path_name, 'w') as f_cnn_param_w:
        f_cnn_param_w.writelines(cnn_param_lines)
    
    if os.path.exists(param_pathname):
        copy_file(param_pathname, dst_pytorch_param_path_name)

    if os.path.exists(bin_pathname):
        copy_file(bin_pathname, dst_pytorch_bin_path_name)
    
    if os.path.exists(src_param_path_name):
        copy_file(src_param_path_name, dst_param_path_name)
    else:
        copy_file(src_param_path_name_origin, src_param_path_name)
    with open(src_param_path_name) as f_param:
        param_lines = f_param.readlines()        
        param_lines[4] = 'demo_video:' + \
            config['train']['demo_video'] + '\n'
        param_lines[5] = 'test_dataset_path:' + \
            config['train']['test_dataset_path'] + '/\n'        
    with open(src_param_path_name, 'w') as f_param_w:
        f_param_w.writelines(param_lines)
    
    print(f'model_full_path: {model_full_path}')
    print(f'best_model_path: {best_model_path}')

    # added by Holy 2109290810
    dst_pytorch_param_path_name_build = os.path.join(config['train']['cpp_project_path'],
                                             'build/install/bin/support/holyMess_one_class/'+param_filename).replace('\\', '/')

    dst_pytorch_bin_path_name_build = os.path.join(config['train']['cpp_project_path'],
                                             'build/install/bin/support/holyMess_one_class/'+bin_filename).replace('\\', '/')

    dst_cnn_param_path_name_build = os.path.join(config['train']['cpp_project_path'],
                                           'build/install/bin/support/holyMess_one_class/holy_one_class_param.ini').replace('\\', '/')

    dst_param_tag_build = os.path.join(config['train']['cpp_project_path'],
                                       'build/install/bin/param_'+config['train']['model_tag'] +
                                       '.ini').replace('\\', '/')
    
    dst_cnn_param_tmp_path_name_build = os.path.join(config['train']['cpp_project_path'],
                                           'build/install/bin/param_tmp.ini').replace('\\', '/')
    
    if os.path.exists(dst_cnn_param_path_name_build):
        copy_file(src_cnn_param_path_name, dst_cnn_param_path_name_build)
        copy_file(dst_pytorch_param_path_name, dst_pytorch_param_path_name_build)
        copy_file(dst_pytorch_bin_path_name, dst_pytorch_bin_path_name_build)
        copy_file(src_param_path_name, dst_param_tag_build)

        src_param_str = 'bool_debug:false'
        dst_param_str = 'bool_debug:true'
        update_file(dst_param_tag_build, src_param_str, dst_param_str)

        copy_file(src_param_tmp_path_name, dst_cnn_param_tmp_path_name_build)
    # end of addition 2109290810

    # added by Holy 2109300810
    # copy ps1 file
    src_ps1_file = './templates/run_demo_one_class.ps1'
    dst_ps1_file = './run_demo_one_class.ps1'
    copy_file(src_ps1_file, dst_ps1_file)

    # update ps1 file
    src_ps1_str = 'd:/backup/project/windingMessRope_holy/build/install/bin'
    dst_ps1_str = os.path.join(config['train']['cpp_project_path'],
                               'build/install/bin').replace('\\', '/')
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    src_ps1_str = 'first string'
    dst_ps1_str = first_str
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    src_ps1_str = 'second string'
    dst_ps1_str = second_str
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    src_ps1_str = 'total_frames_num'
    dst_ps1_str = f'total_frames_num: {total_frames_num}'
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    # run ps1
    p = subprocess.Popen(['pwsh.exe', dst_ps1_file], stdout=sys.stdout)
    p.communicate()
    # end of addition 2109300810
