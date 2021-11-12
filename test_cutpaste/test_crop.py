import cv2
from PIL import Image
import random
import imutils
import numpy

img = cv2.imread('d:/backup/project/automation_pytorch/plot_2110281054.png')
h, w = img.shape[:2]
# cv2.imshow('img',img)
# cv2.waitKey(0)

img_pil = Image.open('d:/backup/project/automation_pytorch/plot_2110281054.png')
# img_pil.show()

from_location_w = 10
cut_w = 100
from_location_h = 20
cut_h = 200

box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
patch_pil = img_pil.crop(box)
patch = img[from_location_h:from_location_h + cut_h, from_location_w:from_location_w + cut_w] # roi = image[startY:endY, startX:endX]
patch1 = patch.copy()

print('patch_pil size: ', patch_pil.size)
print('patch size: ', (patch.shape[1],patch.shape[0]))

# cv2.imshow('patch',patch)
# cv2.waitKey(0)

# patch_pil.show()

to_location_h = int(random.uniform(0, h - cut_h))
to_location_w = int(random.uniform(0, w - cut_w))

insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
augmented_pil = img_pil.copy()
augmented_pil.paste(patch_pil, insert_box)

augmented = img.copy()
augmented[to_location_h:to_location_h + cut_h,to_location_w:to_location_w + cut_w] = patch

# cv2.imshow('augmented',augmented)
# cv2.waitKey(0)

# augmented_pil.show()

rot_deg = random.uniform(-45,45)
patch_pil_rotated = patch_pil.convert("RGBA").rotate(rot_deg,expand=True)

patch = cv2.cvtColor(patch, cv2.COLOR_RGB2RGBA)
patch_rotated = imutils.rotate_bound(patch, -1*rot_deg)

print('patch_pil_rotated size: ', patch_pil_rotated.size)
print('patch_rotated size: ', (patch_rotated.shape[1],patch_rotated.shape[0]))

# cv2.imshow('patch_rotated',patch_rotated)
# cv2.waitKey(0)

# patch_pil_rotated.show()

mask = patch_pil_rotated.split()[-1]
print('mask size: ', mask.size)
sequence_of_pixels = mask.getdata()
list_of_pixels = list(sequence_of_pixels)
print('min value of mask: ', min(list_of_pixels))
print('max value of mask: ', max(list_of_pixels))
mask.show()

mask_cv = patch_rotated[:,:,-1]
print('min value of mask_cv: ', mask_cv.min())
print('max value of mask_cv: ', mask_cv.max())
cv2.imshow('mask_cv',mask_cv)
cv2.waitKey(0)

patch_rotated_to_pil = Image.fromarray(patch_rotated)
patch_rotated_to_pil.show()

mask_to_pil = patch_rotated_to_pil.split()[-1]
print('mask_to_pil size: ', mask_to_pil.size)
mask_to_pil.show()

patch = Image.fromarray(patch1)
patch = patch.convert("RGBA").rotate(rot_deg,expand=True)

to_location_h = int(random.uniform(0, h - patch.size[0]))
to_location_w = int(random.uniform(0, w - patch.size[1]))

mask = patch.split()[-1]
patch = patch.convert("RGB")

augmented = img.copy()
augmented = Image.fromarray(augmented)
augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
augmented = numpy.array(augmented)

cv2.imshow('augmented',augmented)
cv2.waitKey(0)

augmented_cv = img.copy()
mask_cv_h = mask_cv.shape[0]
mask_cv_w = mask_cv.shape[1]
orininal_h = augmented_cv.shape[0]
orininal_w = augmented_cv.shape[1]
print('mask_cv height: ', mask_cv_h)
print('mask_cv width:', mask_cv_w)
print('augmented_cv height: ', orininal_h)
print('augmented_cv width:', orininal_w)

for j in range(mask_cv_h):
    for k in range(mask_cv_w):
        if mask_cv[j,k] == 255:
            augmented_cv[j+to_location_h,k+to_location_w,0] = patch_rotated[j,k,0]
            augmented_cv[j+to_location_h,k+to_location_w,1] = patch_rotated[j,k,1]
            augmented_cv[j+to_location_h,k+to_location_w,2] = patch_rotated[j,k,2]

cv2.imshow('augmented_cv',augmented_cv)
cv2.waitKey(0)