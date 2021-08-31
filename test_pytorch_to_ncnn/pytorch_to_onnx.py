import torch
import torchvision

#define resnet18 model
model = torchvision.models.resnet18(pretrained=True)
#define input shape
x = torch.rand(1, 3, 224, 224)
#define input and output nodes, can be customized
input_names = ["x"]
output_names = ["y"]
#convert pytorch to onnx
torch_out = torch.onnx.export(
    model, x, "resnet18.onnx", input_names=input_names, output_names=output_names)
