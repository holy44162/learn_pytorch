from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed

# added by Holy 2111090810
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy
# end of addition 2111090810

import cv2 # added by Holy 2111120810

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

class MVTecAT_A(Dataset):
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
                # img = self.transform(img)
                # added by Holy 2111090810
                img = numpy.array(img)
                img = self.transform(image=img)['image']
                img = Image.fromarray(img)
                # end of addition 2111090810
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size,self.size)).convert("RGB")
            if self.transform is not None:
                # img = self.transform(img)
                # added by Holy 2111090810
                img = numpy.array(img)
                img = self.transform(image=img)['image']
                img = Image.fromarray(img)
                # end of addition 2111090810
            return img, label != "good"

class MVTecAT_B(Dataset):
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
                # added by Holy 2111090810
                # img = numpy.array(img)
                # img = self.transform(image=img)['image']
                # img = Image.fromarray(img)
                # end of addition 2111090810
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size,self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
                # added by Holy 2111090810
                # img = numpy.array(img)
                # img = self.transform(image=img)['image']
                # img = Image.fromarray(img)
                # end of addition 2111090810
            return img, label != "good"

class MVTecAT_C(Dataset):
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
                # img = self.transform(img)
                # added by Holy 2111090810
                img = numpy.array(img)
                img = self.transform[0](image=img)['image']
                img = Image.fromarray(img)
                img = self.transform[1](img)
                # end of addition 2111090810
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size,self.size)).convert("RGB")
            if self.transform is not None:
                # img = self.transform(img)
                # added by Holy 2111090810
                img = numpy.array(img)
                img = self.transform[0](image=img)['image']
                img = Image.fromarray(img)
                img = self.transform[1](img)
                # end of addition 2111090810
            return img, label != "good"

# added by Holy 2111120810
def read_and_resize(filename, size):
    img = cv2.imread(filename.__str__())
    img = cv2.resize(img, (size,size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
# end of addition 2111120810

class normal_mess_data(Dataset):
    """normal mess dataset."""

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
            self.image_names = list((self.root_dir / "train" / "normal").glob("*.*")) # tested by Holy 2111020810
            print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            #self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            # self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
            self.imgs = Parallel(n_jobs=10)(delayed(read_and_resize)(file,size) for file in self.image_names) # added by Holy 2111120810
            print(f"loaded {len(self.imgs)} images")
        else:
            #test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            self.image_names = list((self.root_dir / "val").glob(str(Path("*") / "*.*"))) # tested by Holy 2111020810
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
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
            img = cv2.resize(img, (self.size,self.size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # end of addition 2111120810

            if self.transform is not None:
                # img = self.transform(img)
                img = self.transform(image=img)['image'] # added by Holy 2111120810
            return img, label != "normal"

class normal_mess_data_test(Dataset):
    """normal mess dataset."""

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
            self.image_names = list((self.root_dir / "train" / "normal").glob("*.*")) # tested by Holy 2111020810
            print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            #self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            # self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
            self.imgs = Parallel(n_jobs=10)(delayed(read_and_resize)(file,size) for file in self.image_names) # added by Holy 2111120810
            print(f"loaded {len(self.imgs)} images")
        else:
            #test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            self.image_names = list((self.root_dir / "val").glob(str(Path("*") / "*.*"))) # tested by Holy 2111020810
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
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
            img = cv2.resize(img, (self.size,self.size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # end of addition 2111120810

            if self.transform is not None:
                # img = self.transform(img)
                img = self.transform(image=img)['image'] # added by Holy 2111120810
            return img, label != "normal"