from .cuda_driver import *
from pynvrtc.compiler import Program, ProgramException
from .device_info import deviceinfo
import subprocess
import ctypes

PTX_PATH='./egl_kernel.ptx'
CU_PATH='./egl_kernel.cu'

def compile_with_nvcc(cuda_text):
    with open(CU_PATH, 'w+') as f:
        f.write(cuda_text)
    device = deviceinfo()
    nvcc_path = device.nvcc_path
    cp = str(device.cc_major.value * 10 + device.cc_minor.value)
    extra_flags = ' -lineinfo'
    cmd = nvcc_path + ' ' + CU_PATH + ' -arch=compute_' + cp + ' -ptx ' + extra_flags
    print('cmd', cmd)
    ret = subprocess.check_output(cmd, shell=True)
    print('Output of nvcc call', ret)

def compile_with_nvrtc(cuda_text):
    c = Program(cuda_text)
    device = deviceinfo()
    cp = str(device.cc_major.value * 10 + device.cc_minor.value)
    ptx = c.compile(['-arch=compute_' + cp])
    with open(PTX_PATH, 'w+') as f:
        f.write(ptx)

def compile_cuda(cuda_text):
    try:
        compile_with_nvcc(cuda_text)
        cu_module = c_void_p()
        char_p = c_char_p((PTX_PATH).encode())
        ret = cuModuleLoad(byref(cu_module), char_p)
        if ret:
            raise Exception('cuModuleLoad fails with ret ' + str(ret))
        return cu_module
    except Exception as e:
        raise e
