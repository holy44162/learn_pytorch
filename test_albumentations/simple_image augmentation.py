import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

from torchvision.datasets.folder import *
from albumentations.pytorch import ToTensorV2
import torch

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

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    image_mess = cv2.imread('e:/dnn_data/z75_data/train/mess/img00009.jpg')
    image_normal = cv2.imread('e:/dnn_data/z75_data/train/normal/img00009.jpg')
    image_mess = cv2.cvtColor(image_mess, cv2.COLOR_BGR2RGB)
    image_normal = cv2.cvtColor(image_normal, cv2.COLOR_BGR2RGB)
    # visualize(image)
    augmented_image = []
    augmented_image.append(image_mess)
    augmented_image.append(image_normal)

    # transform = A.HorizontalFlip(p=0.5)
    # transform = A.Blur()
    # transform = A.CLAHE()
    # transform = A.ColorJitter()
    # transform = A.Emboss()
    # transform = A.Equalize()
    # transform = A.FancyPCA()
    # transform = A.GaussNoise()
    # transform = A.GaussianBlur()
    # transform = A.HueSaturationValue()
    # transform = A.ISONoise()
    # transform = A.ImageCompression()
    # transform = A.MedianBlur()
    # transform = A.MotionBlur()
    # transform = A.MultiplicativeNoise()
    # transform = A.Posterize()
    # transform = A.RGBShift()
    # transform = A.RandomBrightnessContrast()
    # transform = A.RandomFog()
    # transform = A.RandomGamma()
    # transform = A.RandomRain()
    # transform = A.RandomShadow()
    # transform = A.RandomSnow()
    # transform = A.RandomSunFlare()
    # transform = A.RandomToneCurve()
    # transform = A.Sharpen()
    # transform = A.CoarseDropout()
    # transform = A.Flip()
    # transform = A.Perspective()
    # transform = A.RandomGridShuffle()
    # transform = A.ShiftScaleRotate()
    transform = A.VerticalFlip()
    # random.seed(7)
    augmented_image.append(transform(image=augmented_image[0])['image'])
    augmented_image.append(transform(image=augmented_image[1])['image'])
    # visualize(augmented_image[1])    

    transform = A.ShiftScaleRotate(p=0.5)
    random.seed(7) 
    augmented_image.append(transform(image=augmented_image[0])['image'])
    augmented_image.append(transform(image=augmented_image[1])['image'])
    # visualize(augmented_image)

    transform = A.Compose([
        A.CLAHE(),
        A.RandomRotate90(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.HueSaturationValue(),
    ])
    random.seed(42) 
    augmented_image.append(transform(image=augmented_image[0])['image'])
    augmented_image.append(transform(image=augmented_image[1])['image'])
    # visualize(augmented_image)

    transform = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.GaussNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),            
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
    random.seed(42) 
    augmented_image.append(transform(image=augmented_image[0])['image'])
    augmented_image.append(transform(image=augmented_image[1])['image'])
    # visualize(augmented_image)       

    images_so_far = 0
    fig = plt.figure()
    # num_images = len(augmented_image)
    num_images = 4
    
    for j in range(num_images):
        images_so_far += 1
        ax = plt.subplot(num_images//2, 2, images_so_far)
        ax.axis('off')
        plt.imshow(augmented_image[j])
    
    manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    manager.window.showMaximized()
    plt.show()

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
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToTensorV2(),
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
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToTensorV2(),        
        ])
    }

    BATCH_SIZE = 2**5
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

    images_so_far = 0
    fig = plt.figure()
    num_images = 8
    flag = False

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            print(inputs.size())
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                plt.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    flag = True
                    break

            if flag == True:
                break
            
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()    