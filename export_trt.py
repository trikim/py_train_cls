
"""author 
   baiyu
"""

import argparse
import glob
import os

#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
import numpy as np

#from PIL import Image
#import transforms 
#from torchvision import transforms
#from conf import settings
from utils import *
import time
import cv2
import tensorrt as trt
TRT_LOGGER = trt.Logger()#(trt.Logger.VERBOSE)

from collections import defaultdict, OrderedDict
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpus', type=int, default=0, help='gpu device')
    parser.add_argument('-num_classes', type=int, default=2, help='num_classes')
    parser.add_argument('-model_file_path', type=str, required=True, help='weights path')
    ##onnx
    parser.add_argument('-input_h', type=int, default=224, help='deploy image height')
    parser.add_argument('-input_w', type=int, default=224, help='deploy image width')
    #parser.add_argument('-onnx_name', type=str, default='cls', help='onnx file name')
    parser.add_argument('-input_names', type=str, default='input_names', help='input name')
    parser.add_argument('-output_names', type=str, default='output_names', help='output name')
    parser.add_argument('-export_onnx',action='store_true',default=True, help='onnx.')

    ##trt 
    parser.add_argument('-max_batch_size', type=int, default=32, help='max batch size')
    parser.add_argument('-trt_name', type=str, default='cls', help='trt file name')
    parser.add_argument('-fp16', action='store_true',default=False, help='half precision float point')
    parser.add_argument('-int8', action='store_true',default=False, help='int8')
    parser.add_argument('-trt_version', type=str, default='601', help='model version')
    parser.add_argument('-model_version', type=str, default='1001', help='model version')

    args = parser.parse_args()

    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')
    
    model_precision = 'fp32'
    name = args.trt_name
    platform = "tensorrt_plan"
    version = "1001"
    cc = torch.cuda.get_device_capability(args.device)
    cc_arch = str(cc[0])+str(cc[1])
    if args.fp16:
        model_precision = 'fp16'
    if args.int8:
        model_precision = 'int8'
        print("int8 model is not supported now")
        exit(0)
    if not os.path.exists("./onnx"):
        os.mkdir("./onnx")
    if not os.path.exists("./trt"):
        os.mkdir("./trt")

    config_dict = {
      'name': args.trt_name,
      'platform': "tensorrt_plan",
      'max_batch_size': args.max_batch_size,
        
      'input': 
      {
        'name': args.input_names,
        'data_type': "TYPE_FP32",
        'color_type': "BGR",
        'mean':"[123.829891747,127.351147446,110.256170154]",
        'std':"[0.016895854,0.017222115,0.014714524]",
        'format': "FORMAT_NCHW",
        'dims': "[3,"+ str(args.input_h)+","+str(args.input_w)+"]",
      },
      'output': 
      {
        'name': args.output_names,
        'data_type': "TYPE_FP32",
        'dims': "["+str(args.num_classes)+",1,1]",
      }
    }
    json_str = json.dumps(config_dict,indent=1)
    with open('./trt/config.pbtxt', 'w') as json_file:
        json_file.write(json_str)

    onnx_file_name = args.trt_name + ".onnx." + args.trt_version + "." + cc_arch + ".00.00."+version
    onnx_file_path = "./onnx/"+onnx_file_name
    if args.export_onnx:
        net = get_network(args)

        print('Loading torch file from path {}...'.format(args.model_file_path))
        #checkpoint = torch.load(args.model_file_path)
	
        net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.model_file_path)['model'].items()})
        print("net:",net)
        net = net.to(args.device)
        #net = net.cuda()
        net.eval()
        #dummy_input = torch.tensor([1, 3, args.input_h, args.input_w], dtype=torch.float32)
        dummy_input=torch.tensor(torch.randn(1, 3, args.input_h, args.input_w), dtype=torch.float32).to(args.device)
    
        torch.onnx.export(net, dummy_input, onnx_file_path, export_params=True, verbose=True, input_names=[args.input_names], output_names=[args.output_names])

        print('The onnx model is saved into {}.'.format(onnx_file_path))
 
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = args.max_batch_size
        if args.fp16:
            if not builder.platform_has_fast_fp16:
                print("fp16 model is not supported for this device")
                esit(0)
            print('fp16 mode...')
            builder.fp16_mode = True
        if args.int8:
            if not builder.platform_has_fast_int8:#To Do 
                print("fp16 model is not supported for this device")
                esit(0)
            print('int8 mode...')
            builder.int8_mode = True
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please first to generate it.'.format(onnx_file_path))
            exit(0)

        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        #print("num_layers:",network.num_layers)
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")

        trt_file_name = args.trt_name + ".trt." + args.trt_version + "." + cc_arch + "." + model_precision + "." + str(args.max_batch_size) +"." + version+".mod"
        trt_file_path = "./trt/"+trt_file_name
        with open(trt_file_path, "wb") as f:
            f.write(engine.serialize())
            print('The tensorrt engine model is saved into {}.'.format(trt_file_path))




    


    

