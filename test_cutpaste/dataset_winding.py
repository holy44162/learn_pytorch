from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed

# added by Holy 2111021500
from torchvision.datasets.folder import *
import cv2
from torchvision import transforms
# end of addition 2111021500

class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]

class MVTecAT(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size
        
        # find test images
        if self.mode == "train":
            # self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.*")) # tested by Holy 2111020810
            print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            #self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
            print(f"loaded {len(self.imgs)} images")
        else:
            #test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.*"))) # tested by Holy 2111020810
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size,self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "good"

class winding_data(Dataset):
    """winding dataset."""

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the winding dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size
        
        # find test images
        if self.mode == "train":
            # self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.image_names = list((self.root_dir / defect_name / "train" / "normal").glob("*.*")) # tested by Holy 2111020810
            print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            #self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
            print(f"loaded {len(self.imgs)} images")
        else:
            #test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            self.image_names = list((self.root_dir / defect_name / "val").glob(str(Path("*") / "*.*"))) # tested by Holy 2111020810
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size,self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "normal"

class normal_mess_data(Dataset):
    """normal and mess winding dataset."""

    def __init__(self, root_dir, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the winding dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.size = size
        
        # find test images
        if self.mode == "train":
            # self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.image_names = list((self.root_dir / "train" / "normal").glob("*.*"))            
        else:
            #test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            self.image_names = list((self.root_dir / "val").glob(str(Path("*") / "*.*"))) # tested by Holy 2111020810
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            filename = self.image_names[idx]

            # hided by Holy 2111040810
            # img = Image.open(filename)
            # img = img.resize((self.size,self.size)).convert("RGB")
            # end of hide 2111040810

            # added by Holy 2111040810
            img = cv2.imread(filename.__str__())
            print(type(img))
            print('img shape: ', img.shape)
            img = cv2.resize(img, (self.size,self.size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print('img_after shape: ', img.shape)
            # end of addition 2111040810
            
            if self.transform is not None:
                # img = self.transform(img)
                # added by Holy 2111040810
                # print(type(img))
                # img = self.transform(image=img)["image"]
                # img = Image.fromarray(img)
                img = self.transform[0](image=img)["image"]
                print(type(img))
                # image = img.cpu().clone()
                # image = image.squeeze(0)
                # img = transforms.ToPILImage(image)
                img = Image.fromarray(img)
                img = self.transform[1](img)
                # end of addition 2111040810
            else:
                img = Image.fromarray(img)
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]

            # hided by Holy 2111040810
            # img = Image.open(filename)
            # img = img.resize((self.size,self.size)).convert("RGB")
            # end of hide 2111040810

            # added by Holy 2111040810
            img = cv2.imread(filename.__str__())
            img = cv2.resize(img, (self.size,self.size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # end of addition 2111040810

            if self.transform is not None:
                # img = self.transform(img)
                # added by Holy 2111040810
                # img = self.transform(image=img)["image"]
                # img = Image.fromarray(img)
                # end of addition 2111040810
                img = self.transform[0](image=img)["image"]
                print(type(img))
                # image = img.cpu().clone()
                # image = image.squeeze(0)
                # img = transforms.ToPILImage(image)
                img = Image.fromarray(img)
                img = self.transform[1](img)
            else:
                img = Image.fromarray(img)
            return img, label != "normal"

class CatsVsDogsDataset_holy(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_org = image.copy()
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "Cat":
            label = 1.0
        else:
            label = 0.0
        if self.transform is not None:
            image = self.transform[0](image=image_org)["image"]
            print(type(image))
            img = Image.fromarray(image_org)
            image = self.transform[1](img)
        return image
# class normal_mess_data(Dataset):
#     """normal and mess winding dataset."""

#     def __init__(self,root: str,size,transform = None,target_transform = None,loader = default_loader,is_valid_file= None, mode="train"):
#         # super(normal_mess_data, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
#         #                                   transform=transform,
#         #                                   target_transform=target_transform,
#         #                                   is_valid_file=is_valid_file)
#         self.mode = mode
#         self.root_dir = Path(root)
#         self.size = size

#         if self.mode == "train":
#             self.image_names = list((self.root_dir / "train" / "normal").glob("*.*"))
#         else:
#             #test mode
#             self.image_names = list((self.root_dir / "val").glob(str(Path("*") / "*.*")))
    
#     def __len__(self):
#         return len(self.image_names)
    
#     def __getitem__(self, idx: int):
#         """
#         Args:
#             idx (int): Index

#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         if self.mode == "train":
#             filename = self.image_names[idx]
#             sample = cv2.imread(filename.__str__())
#             sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
#             sample = cv2.resize(sample, (self.size,self.size))
#             if self.transform is not None:
#                 sample = self.transform(sample)["image"]
            
#             return sample
#         else:
#             filename = self.image_names[idx]
#             target = filename.parts[-2]
#             sample = cv2.imread(filename.__str__())
#             sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
#             sample = cv2.resize(sample, (self.size,self.size))
#             if self.transform is not None:
#                 sample = self.transform(sample)["image"]
#             if self.target_transform is not None:
#                 target = self.target_transform(target)
#             return sample, target != "normal"        
