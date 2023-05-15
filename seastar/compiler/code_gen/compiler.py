from .cuda_driver import *
from pynvrtc.compiler import Program, ProgramException
from .device_info import DeviceInfo
import subprocess
import ctypes
import snoop
from ctypes import c_void_p, c_char_p, byref
from .cuda_error import ASSERT_DRV

PTX_PATH='./egl_kernel.ptx'
CU_PATH='./egl_kernel.cu'


def compile_with_nvcc(cuda_text):
    with open(CU_PATH, 'w+') as f:
        f.write(cuda_text)
    device = DeviceInfo()
    nvcc_path = device.nvcc_path
    cp = str(device.cc_major * 10 + device.cc_minor)
    extra_flags = ' -lineinfo'
    cmd = nvcc_path + ' ' + CU_PATH + ' -arch=compute_' + cp + ' -ptx ' + extra_flags

    # Trying to set max register count
    # cmd = nvcc_path + ' ' + CU_PATH + ' -arch=compute_' + cp + ' -ptx ' + ' -maxrregcount=32 ' + extra_flags

    print('cmd', cmd)
    ret = subprocess.check_output(cmd, shell=True)
    print('Output of nvcc call', ret)

def compile_with_nvrtc(cuda_text):
    c = Program(cuda_text)
    device = DeviceInfo()
    cp = str(device.cc_major * 10 + device.cc_minor)
    ptx = c.compile(['-arch=compute_' + cp])
    with open(PTX_PATH, 'w+') as f:
        f.write(ptx)

def compile_cuda(cuda_text):
    try:
        compile_with_nvcc(cuda_text)
        char_p = (PTX_PATH).encode()
        ret, cu_module = cuModuleLoad(char_p)
        ASSERT_DRV(ret)
        return cu_module
    except Exception as e:
        raise e
