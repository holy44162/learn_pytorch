import torch
import torch.onnx as onnx
import torchvision.models as models

# added by Holy 2109080810
import onnx as onnx_origin
from onnxsim import simplify
# end of addition 2109080810

if __name__ == "__main__":
    # load model_ft from model file
    model_ft_full = torch.load('model_ft_shufflenet_v2.pth')
    # model_ft_full = torch.load('model_ft_shufflenet_v2_x0_5.pth') # added by Holy 2109030810

    # export to onnx
    input_image = torch.zeros((1,3,224,224))
    # onnx.export(model_ft_full, input_image, 'model_shufflenet_v2.onnx')

    # added by Holy 2109080810
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_image = input_image.to(device)
    # end of addition 2109080810

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

    # added by Holy 2109080810
    # load your predefined ONNX model
    model = onnx_origin.load('model_shufflenet_v2.onnx')

    # convert model
    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    # Save the ONNX model
    onnx_origin.save(model_simp, 'model_shufflenet_v2_simplified.onnx')
    # end of addition 2109080810
