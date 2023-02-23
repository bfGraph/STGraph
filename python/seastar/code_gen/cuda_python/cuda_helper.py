from cuda import cuda, nvrtc
import numpy as np

from .cuda_error import ASSERT_DRV
from .cuda_result import *

def createNpArray(array, element_type):
    """
        array:          A list of numbers
        element_type:   np.dtype

        returns:        np.Array of the list and the buffer
                        size of the array
    """
    numpy_array = np.array(array, dtype=element_type)
    buffer_size = numpy_array.size * numpy_array.dtype.itemsize
    return numpy_array, buffer_size

def createNpArrayZeros(count, element_type):
    """
        count:          Number of zeros in the array
        element_type:   np.dtype

        returns:        np.Array containing zeros and the
                        buffer size of the array
    """
    numpy_array = np.zeros(count).astype(dtype=element_type)
    buffer_size = numpy_array.size * numpy_array.dtype.itemsize
    return numpy_array, buffer_size

def getArraySize(array):
    return array.size * array.dtype.itemsize

def allocAndCopyToDevice(array, array_size, stream):
    """
        array:          numpy array which is allocated in host
        array_size:     size of the numpy array
        stream:         CUDA stream created

        returns:        Array that is stored in device
                        and pointer that is allocated in GPU

        This function initially allocates neccessary space
        in the GPU with the cuMemAlloc() API with the array size
        passed as an argument.

        Then it makes sure to copy the array from host to device,
        i.e from CPU to GPU.

        To work with it, the array is then converted into a numpy
        array of dtype uint64, which is required according to the 
        CUDA Python library.
    """
    err, arrayClass = cuda.cuMemAlloc(array_size)
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyHtoDAsync(arrayClass, array.ctypes.data, array_size, stream)
    ASSERT_DRV(err)

    d_array = np.array([int(arrayClass)], dtype=np.uint64)

    return d_array, arrayClass

def get_kernel_args(arg_list):
    """
        arg_list:   List of arguments in the order passed
                    to the kernel.

        returns     numpy array of arguments that is valid to 
                    be passed to the cuda.cuLaunchKernel()
    """
    return np.array([arg.ctypes.data for arg in arg_list], dtype=np.uint64)

def copy_arguments_to_gpu(kernel_name, argument_list, stream):

    # breakpoint()

    ret_tensor_indices = get_kernel_ret_tensor_indices(kernel_name)
    # breakpoint()

    graph_vector_arguments = argument_list[:-8]
    graph_csr_arguments = argument_list[-8:-5]
    scalar_arguments = argument_list[-5:]

    kernel_arguments = []
    result_tensor_info = []

    # copying graph vector arguments to GPU Device
    for argument in graph_vector_arguments:
        # breakpoint()
        host_argument, host_argument_size = createNpArray(argument.flatten().tolist(), np.float32)
        device_argument, argument_class = allocAndCopyToDevice(host_argument, host_argument_size, stream)

        # storing return tensor info
        if len(kernel_arguments) in ret_tensor_indices:
            arg_index = len(kernel_arguments)
            ret_tensor_info = (
                get_kernel_arg_name_from_index(kernel_name, arg_index),
                host_argument,
                argument_class,
                host_argument_size
            )
            result_tensor_info.append(ret_tensor_info)

        kernel_arguments.append(device_argument)

    # copying graph csr arguments to GPU Device
    for argument in graph_csr_arguments:
        # breakpoint()
        host_argument, host_argument_size = createNpArray(argument.flatten().tolist(), np.int32)
        device_argument, argument_class = allocAndCopyToDevice(host_argument, host_argument_size, stream)
        kernel_arguments.append(device_argument)

    # copying scalar arguments to GPU Device
    for argument in scalar_arguments:
        host_argument, host_argument_size = createNpArray([argument], np.int32)
        kernel_arguments.append(host_argument)

    # converting it into a form that can be passed 
    # to the kernel function call
    kernel_arguments = get_kernel_args(kernel_arguments)

    return kernel_arguments, result_tensor_info