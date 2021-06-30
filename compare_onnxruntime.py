#!/usr/bin/python3

import numpy as np
import torch
from unet import UNet
import onnx
import onnxruntime


def testOnnxFile():
    onnx_model = onnx.load("unet.onnx")
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy(
    ) if tensor.requires_grad else tensor.cpu().numpy()


testOnnxFile()

# Instantiate PyTorch model
net = UNet(n_channels=3, n_classes=1)
net.load_state_dict(
    torch.load('unet_carvana_scale1_epoch5.pth',
               map_location=torch.device('cpu')))
print(net.eval())

#Define input and run PyTorch model.
batch_size, channels, height, width = 1, 3, 1080, 1920
inputs = torch.randn((batch_size, channels, height, width))
pt_output = net(inputs)
print(pt_output)

# Compare ONNX runtime to PyTorch
ort_session = onnxruntime.InferenceSession("unet.onnx")
print(ort_session.get_inputs())

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0])

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(pt_output),
                           ort_outs[0],
                           rtol=1e-03,
                           atol=1e-05)
