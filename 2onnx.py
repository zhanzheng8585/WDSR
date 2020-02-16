import os, time, shutil, argparse
from functools import partial
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from collections import OrderedDict
import torch.utils.data
import torch.utils.data.distributed
import torch.onnx as torch_onnx
import onnx

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage import io
import onnxruntime

# import prune_util
# from prune_util import GradualWarmupScheduler
# from prune_util import CrossEntropyLossMaybeSmooth
# from prune_util import mixup_data, mixup_criterion

# from utils import save_checkpoint, AverageMeter, visualize_image, GrayscaleImageFolder
# from model import ColorNet
from wdsr_b import *
from args import *

def main():

    use_gpu = torch.cuda.is_available()
    # Create model  
    # models.resnet18(num_classes=365)
    # model = ColorNet()
    args = get_args()
    model = MODEL(args)
    # state_dict = torch.load("./checkpoint/checkpoint6/model_epoch133_step1.pth")
    # new_state_dict = OrderedDict()

    # for k, v in state_dict.items():
    #     k = k.replace('module.', '')
    #     new_state_dict[k] = v

    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(new_state_dict)
    checkpoint = torch.load("./checkpoint/checkpoint6/model_epoch133_step1.pth")
    model.load_state_dict(checkpoint["model"].state_dict())

    # print(model)
    torch.nn.utils.remove_weight_norm(model.head[0])
    torch.nn.utils.remove_weight_norm(model.body[0].body[0])
    torch.nn.utils.remove_weight_norm(model.body[0].body[2])
    torch.nn.utils.remove_weight_norm(model.body[0].body[3])
    torch.nn.utils.remove_weight_norm(model.body[1].body[0])
    torch.nn.utils.remove_weight_norm(model.body[1].body[2])
    torch.nn.utils.remove_weight_norm(model.body[1].body[3])
    torch.nn.utils.remove_weight_norm(model.body[2].body[0])
    torch.nn.utils.remove_weight_norm(model.body[2].body[2])
    torch.nn.utils.remove_weight_norm(model.body[2].body[3])
    torch.nn.utils.remove_weight_norm(model.body[3].body[0])
    torch.nn.utils.remove_weight_norm(model.body[3].body[2])
    torch.nn.utils.remove_weight_norm(model.body[3].body[3])
    torch.nn.utils.remove_weight_norm(model.body[4].body[0])
    torch.nn.utils.remove_weight_norm(model.body[4].body[2])
    torch.nn.utils.remove_weight_norm(model.body[4].body[3])
    torch.nn.utils.remove_weight_norm(model.body[5].body[0])
    torch.nn.utils.remove_weight_norm(model.body[5].body[2])
    torch.nn.utils.remove_weight_norm(model.body[5].body[3])
    torch.nn.utils.remove_weight_norm(model.body[6].body[0])
    torch.nn.utils.remove_weight_norm(model.body[6].body[2])
    torch.nn.utils.remove_weight_norm(model.body[6].body[3])
    torch.nn.utils.remove_weight_norm(model.body[7].body[0])
    torch.nn.utils.remove_weight_norm(model.body[7].body[2])
    torch.nn.utils.remove_weight_norm(model.body[7].body[3])
    torch.nn.utils.remove_weight_norm(model.tail[0])
    torch.nn.utils.remove_weight_norm(model.skip[0])
    # print(model)
    input_shape = (3, 256, 256)
    model_onnx_path = "./wdsr_b.onnx"
    model.train(False)
    model.eval()

    # Export the model to an ONNX file
    dummy_input = Variable(torch.randn(1, *input_shape))
    output = torch_onnx.export(model, 
                              dummy_input, 
                              model_onnx_path, 
                              verbose=False)
    print("Export of torch_model.onnx complete!")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def check():

    args = get_args()
    model = MODEL(args)
    checkpoint = torch.load("./checkpoint/checkpoint6/model_epoch133_step1.pth")
    model.load_state_dict(checkpoint["model"].state_dict())

    ort_session = onnxruntime.InferenceSession("wdsr_b.onnx")

    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    torch_out = model(x)
    # # Load the ONNX model
    # model = onnx.load("wdsr_b.onnx")

    # # Check that the IR is well formed
    # onnx.checker.check_model(model)

    # # Print a human readable representation of the graph
    # onnx.helper.printable_graph(model.graph)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == '__main__':
    # main()
    check()
