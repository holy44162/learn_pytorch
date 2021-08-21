import torch
import torch.onnx as onnx
import torchvision.models as models

if __name__ == "__main__":
    # load model_ft from model file
    model_ft_full = torch.load('model_ft.pth')

    # export to onnx
    input_image = torch.zeros((1,3,224,224))
    onnx.export(model_ft_full, input_image, 'model.onnx')