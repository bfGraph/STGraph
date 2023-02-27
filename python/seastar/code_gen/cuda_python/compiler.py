from cuda import cuda, nvrtc
import numpy as np

from .cuda_driver import *
from .device_info import DeviceInfo, get_global_gpu_device
from .cuda_error import ASSERT_DRV

CU_PATH='./egl_kernel.cu'
PTX_PATH='./egl_kernel.ptx'

def compile_with_cuda_python(cuda_code):

    # breakpoint()

    # writing the CUDA Kernel source code
    # to egl_kernel.cu
    with open(CU_PATH, "w+") as f:
        f.write("\n")
        f.write(cuda_code)

    gpu_device = get_global_gpu_device()

    err, prog = nvrtc.nvrtcCreateProgram(str.encode(cuda_code), b"egl_kernel.cu", 0, [], [])
    ASSERT_DRV(err)

    compute_capability = str(10*gpu_device.cc_major + gpu_device.cc_minor)
    opts = [b"--fmad=false", b"--gpu-architecture=compute_" + compute_capability.encode("utf-8")]
    err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)
    ASSERT_DRV(err)

    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ASSERT_DRV(err)

    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    ASSERT_DRV(err)

    # writing the PTX code generated to
    # egl_kernel.ptx
    with open(PTX_PATH, "w+") as f:
        f.write(ptx.decode())

    return np.char.array(ptx)

def compile_cuda(cuda_code):

    # print(cuda_code)
    ptx = compile_with_cuda_python(cuda_code)

    err, module = cuModuleLoadData(ptx.ctypes.data)
    ASSERT_DRV(err)

    return module

    print(ptx)
    print(quit)
    quit()