import torch
import torchvision
import onnxruntime as rt
import numpy as np
import cv2

#test image
img_path = 'd:/data_seq/gongqiWinding/Z75_DF-4105H-BD/210820/shrinkVideo/smallDatasets/test/imgs/img00115.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = np.transpose(img, (2, 0, 1)).astype(np.float32)
img = torch.from_numpy(img)
img = img.unsqueeze(0)

# added by Holy 2109080810
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img1 = img.to(device)
# end of addition 2109080810

#pytorch test
model = torch.load('model_ft_shufflenet_v2.pth')
# model = torch.load('model_ft_shufflenet_v2_x0_5.pth') # added by Holy 2109030810
output = model.forward(img1)
val, cls = torch.max(output.data, 1)
print("[pytorch]--->predicted class:", cls.item())
print("[pytorch]--->predicted value:", val.item())

#onnx test
sess = rt.InferenceSession("model_shufflenet_v2.onnx")
# sess = rt.InferenceSession("model_shufflenet_v2_x0_5.onnx") # added by Holy 2109030810
x = "x"
y = ["y"]
output = sess.run(y, {x: img.numpy()})
cls = np.argmax(output[0][0], axis=0)
val = output[0][0][cls]
print("[onnx]--->predicted class:", cls)
print("[onnx]--->predicted value:", val)
