import torch
import torch.onnx as onnx
import torchvision.models as models

if __name__ == "__main__":
    # load model_ft from model file
    model_ft_full = torch.load('model_ft_shufflenet_v2.pth')
    # model_ft_full = torch.load('model_ft_shufflenet_v2_x0_5.pth') # added by Holy 2109030810

    # export to onnx
    input_image = torch.zeros((1,3,224,224))
    # onnx.export(model_ft_full, input_image, 'model_shufflenet_v2.onnx')

    # added by Holy 2109021500
    input_names = ["x"]
    output_names = ["y"]
    #convert pytorch to onnx
    torch_out = onnx.export(
        model_ft_full, input_image, 'model_shufflenet_v2.onnx', input_names=input_names, output_names=output_names)
    # torch_out = onnx.export(
    #     model_ft_full, input_image, 'model_shufflenet_v2_x0_5.onnx',
    #     input_names=input_names, output_names=output_names)  # added by Holy 2109030810
    # end of addition 2109021500
