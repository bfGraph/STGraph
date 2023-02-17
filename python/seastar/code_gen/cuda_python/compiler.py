from .cuda_driver import *

CU_PATH='./egl_kernel.cu'

def compile_with_cuda_python(cuda_code):

    # writing the CUDA Kernel source code
    # to egl_kernel.cu
    with open(CU_PATH, "w+") as f:
        f.write(cuda_code)

    # TODO: Continue from here again

def compile_cuda(cuda_code):
    print(cuda_code)
    compile_with_cuda_python(cuda_code)
    print(quit)
    quit()