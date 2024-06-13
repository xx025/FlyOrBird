import gradio as gr

import numpy as np
import onnxruntime
import torch
from torchvision import transforms

# Load the ONNX model
ort_session = onnxruntime.InferenceSession(
    path_or_bytes=r"data\model_best.pth.onnx",
    providers=[
        'CUDAExecutionProvider',  # 需要CuDNN
        'CPUExecutionProvider'
    ]
)


def preprocess(img):
    preprocess_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)),
    ])
    return preprocess_transform(img)


# 创建一个推理函数
def inference(image):
    img = image

    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    # Run the model on the input image
    ort_inputs = {ort_session.get_inputs()[0].name: batch_t.numpy()}

    ort_outs = ort_session.run(None, ort_inputs)

    # Postprocess the output
    softmax_outs = torch.nn.functional.softmax(torch.tensor(ort_outs[0]), dim=1).numpy()
    probabilities = softmax_outs[0]
    class_name = ['飞机', '鸟']
    return  {class_name[i]: probabilities[i] for i in range(len(class_name))}


# 创建 Gradio 接口
demo = gr.Interface(
    title="FlyOrBird",
    fn=inference,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
    ],
    outputs=[
        gr.Label(label="置信度"),
    ],
    examples=[
        ["data/example/1.jpg"],
        ["data/example/2.jpg"],
        ["data/example/3.jpg"],
        ["data/example/4.jpg"],
    ],
)

# 启动 Gradio 接口
demo.launch()
