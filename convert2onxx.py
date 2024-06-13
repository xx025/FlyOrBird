import torch.onnx

from model import FlyOrBird


def Convert_ONNX(model, dummy_input=None, output_path=None):
    model.eval()
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        f=output_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['modelInput'],  # the model's input names
        output_names=['modelOutput'],  # the model's output names
        dynamic_axes={
            'modelInput': {
                0: 'batch_size',
                2: 'height',
                3: 'width'
            },  # 可变长度轴
            'modelOutput': {
                0: 'batch_size',
            }
        }  # 可变长度轴)
    )
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":

    torch_model = FlyOrBird()
    model_path = "data/model_best.pth"  # 加载模型权重
    output_path = "data/model_best.pth.onnx"

    model_data = torch.load(model_path)
    torch_model.load_state_dict(model_data)

    dummy_input = torch.randn([1, 3, 32, 32], requires_grad=False)
    Convert_ONNX(model=torch_model, dummy_input=dummy_input, output_path=output_path)
