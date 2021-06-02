import os
import cv2
import argparse
import numpy as np
import os.path as osp
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from PIL import Image, ImageOps

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
MODEL_INPUT_RES = [513, 513]
MODEL_OUTPUT_RES = [513, 513]

parser = argparse.ArgumentParser("TensorRT Test")
parser.add_argument('--image', type=str, help="Input image path", required=True)
parser.add_argument('--engine', type=str, help="TensorRT engine path", required=True)
parser.add_argument('--output_dir', type=str, help="Output dir path", required=True)
args = parser.parse_args()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def get_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def main():

    if not osp.exists(args.image):
        print("Input image does not exist.")
        exit(-1)

    if not osp.exists(args.output_dir):
        print("Output dir does not exist. Creating one.")
        try:
            os.mkdir(args.output_dir)
        except:
            try:
                os.makedirs(args.output_dir)
            except:
                print("Cannot create output dir.")
                exit(-2)

    img = np.array(Image.open(args.image))
    resized_img = cv2.resize(img, tuple(MODEL_INPUT_RES), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    output_shapes = [(1, 1, 513, 513)]

    trt_outputs = []
    with get_engine(args.engine) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        inputs[0].host = resized_img
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]


if __name__ == "__main__":
    main()
