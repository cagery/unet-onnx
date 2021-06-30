#!/usr/bin/python3

import torch
from unet import UNet

def parseToOnnx():

    net = UNet(n_channels=3, n_classes=1)
    net.load_state_dict(torch.load('unet_carvana_scale1_epoch5.pth', map_location=torch.device('cpu')))

    print(net.eval())

    batch_size, channels, height, width = 1, 3, 1080, 1920
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
    return

parseToOnnx()