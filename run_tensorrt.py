import argparse
import numpy as np
import tensorrt as trt
import time

from PIL import Image
import cv2
import pycuda.driver as cuda
import pycuda.autoinit

LOGGER = trt.Logger(trt.Logger.WARNING)
DTYPE = trt.float32

# Model
MODEL_FILE = './trt/mask.trt.601.75.fp16.32.1001.mod'
INPUT_NAME = 'input_names'
INPUT_SHAPE = (3, 128, 128)
OUTPUT_NAME = 'output_names'


def allocate_buffers(engine):
    print('allocate buffers')
    
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    return h_input, d_input, h_output, d_output


def build_engine(model_file):
    print('build engine...')
    with open(MODEL_FILE, "rb") as f, trt.Runtime(LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def load_input(img_path, host_buffer):
    print('load input')
    c, h, w = INPUT_SHAPE
    img = cv2.imread(img_path)
    mean = [123.829891747,127.351147446,110.256170154]
    stdv = [0.016895854,0.017222115,0.014714524]    


    img = cv2.resize(img,(h,w))

    img = np.swapaxes(img,0,2)
    img = np.swapaxes(img,1,2)
    img = np.array(img, dtype=float)

    mean = np.array(mean, dtype=float)
    stdv = np.array(stdv, dtype=float)
    img[0,:,:] -= mean[0]
    img[1,:,:] -= mean[1]
    img[2,:,:] -= mean[2]
    img[0,:,:] *= stdv[0]
    img[1,:,:] *= stdv[1] 
    img[2,:,:] *= stdv[2]
    dtype = trt.nptype(DTYPE)
    img_array = np.asarray(img).astype(dtype).ravel() 
    np.copyto(host_buffer, img_array)


def do_inference(n, context, h_input, d_input, h_output, d_output):
    # Transfer input data to the GPU.
    cuda.memcpy_htod(d_input, h_input)

    # Run inference.
    st = time.time()
    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
    print('Inference time {}: {} [msec]'.format(n, (time.time() - st)*1000))

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh(h_output, d_output)
    
    return h_output


def parse_args():
    parser = argparse.ArgumentParser(description='TensorRT execution smaple')
    parser.add_argument('img', help='input image')
    
    return parser.parse_args()


def main():
    args = parse_args()

    with build_engine(MODEL_FILE) as engine:
        h_input, d_input, h_output, d_output = allocate_buffers(engine)
        load_input(args.img, h_input)
        
        with engine.create_execution_context() as context:
            output = do_inference(0, context, h_input, d_input, h_output, d_output)

    pred_idx = np.argsort(output)[::-1]
    pred_prob = np.sort(output)[::-1]

    print('\nClassification Result:',pred_idx,pred_prob)

                
if __name__ == '__main__':
    main()
