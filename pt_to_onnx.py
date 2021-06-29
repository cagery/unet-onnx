#!/usr/bin/python3

import argparse
import logging
import os
from typing import NewType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
#from utils.data_vis import plot_img_and_mask
#from utils.dataset import BasicDataset

def parseToOnnx():

    net = UNet(n_channels=3, n_classes=1)
    net.load_state_dict(torch.load('unet_carvana_scale1_epoch5.pth', map_location=torch.device('cpu')))

    print(net.eval())

    #torch.Size([1, 3, 1080, 1920])
    batch_size, channels, height, width = 1, 3, 1080, 1920
    # batch_size = 1

    inputs = torch.randn((batch_size, channels, height, width))
    outputs = net(inputs)
    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


    torch.onnx.export(net,               # model being run
                      inputs,                         # model input (or a tuple for multiple inputs)
                      "unet.onnx",   # where to save the model (can be a file or file-like   object)
                      export_params=True,        # store the trained parameter weights inside the model     file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names = ['inputs'],   # the model's input names
                      output_names = ['outputs'], # the model's output names
                      dynamic_axes={'inputs' : {0 : 'batch_size'},    # variable lenght axes
                                    'outputs' : {0 : 'batch_size'}})

    print("ONNX model conversion is complete.")
    return inputs, outputs

def testOnnxFile():
    onnx_model = onnx.load("dm_nfnet_f0.onnx")
    onnx.checker.check_model(onnx_model)

parseToOnnx()


# testOnnxFile()

# # Compare ONNX runtime to PyTorch
# ort_session = onnxruntime.InferenceSession("dm_nfnet_f0.onnx")

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
# ort_outs = ort_session.run(None, ort_inputs)

# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(outputs), ort_outs[0], rtol=1e-03, atol=1e-05)

# print("Exported model has been tested with ONNXRuntime, and the result looks good!")