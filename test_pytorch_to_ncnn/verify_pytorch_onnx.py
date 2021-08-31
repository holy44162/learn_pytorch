import torch
import torchvision
import onnxruntime as rt
import numpy as np
import cv2

#test image
img_path = 'd:/data_seq/gongqiWinding/Z75_DF-4105H-BD/210820/shrinkVideo/smallDatasets/test/imgs/img00001.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = np.transpose(img, (2, 0, 1)).astype(np.float32)
img = torch.from_numpy(img)
img = img.unsqueeze(0)

#pytorch test
model = torchvision.models.resnet18(pretrained=True)
model.eval()
output = model.forward(img)
val, cls = torch.max(output.data, 1)
print("[pytorch]--->predicted class:", cls.item())
print("[pytorch]--->predicted value:", val.item())

#onnx test
sess = rt.InferenceSession("resnet18.onnx")
x = "x"
y = ["y"]
output = sess.run(y, {x: img.numpy()})
cls = np.argmax(output[0][0], axis=0)
val = output[0][0][cls]
print("[onnx]--->predicted class:", cls)
print("[onnx]--->predicted value:", val)
