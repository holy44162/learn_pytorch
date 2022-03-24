import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

from torchvision.datasets.folder import *
from albumentations.pytorch import ToTensorV2
import torch


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
    transform = A.VerticalFlip(p=1)
    # random.seed(7)
    augmented_image.append(transform(image=augmented_image[0])['image'])
    augmented_image.append(transform(image=augmented_image[1])['image'])
    # visualize(augmented_image[1])    

    # transform = A.ShiftScaleRotate(p=0.5)
    # random.seed(7) 
    transform = A.Blur(p=1)
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
    num_images = 6
    
    for j in range(num_images):
        images_so_far += 1
        ax = plt.subplot(num_images//2, 2, images_so_far)
        ax.axis('off')
        plt.imshow(augmented_image[j])
    
    manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    manager.window.showMaximized()
    plt.show()  