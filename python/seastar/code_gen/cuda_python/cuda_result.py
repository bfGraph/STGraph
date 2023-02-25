"""This module is made to find a workaround to returning
the return tensors as produced by the CUDA Python back
to Seastar after forward and backward propagation

As of now this is not finalised. Just made for testing it
out if this process would work.
"""

import numpy as np
import torch

from ...program import Var

# Dictionary with key-value pair: (tensor_name, tensor_value)
# stores information about return tensor for all kernels
# tensor_name is the name of the return tensor
# tensor_value is a tensor
# {
#   kernel_name: {
#       tensor_name: tensor_value,
#       ...
#   },
#   ...
# }
cuda_result_tensors = {}

# A dictionary with the following structure:
# {
#   "kernel_name" : {
#       "arg1" : arg1_index,
#       "arg2" : arg2_index,
#               .
#               .
#               .                
#   },
#   . . .
# }
cuda_kernel_args_index = {}

# A dictionary that contains tensor info about
# the return argument of each kernel. Info such as
# tensor.shape, requires_grad and device is stored
# {
#   "kernel_name" : {
#       "ret_arg1" : (device, shape, requires_grad)
#       ...
#   }
#   ...
# }
cuda_result_tensor_info = {}

def init_result_tensors(kernel_name: str, return_vars: list[Var]):
    """ Initialises the cuda_result_tensor dictionary
    
        Initialises the cuda_result_tensor dictionary for the current
        kernel by setting the value to None.

        If a certain kernel K0 has V2 as its return tensor, then 
        cuda_result_tensor would look like {'V2': None}

        Arguments:
            kernel_name: str
                name of the kernel

            return_vars (list[Program.Var]):
                A python list containing Var objects, in this
                case it is the return arguments of a kernel

        Returns:
            None
    """
    kernel_result_tensors = {}
    return_var: Var
    for return_var in return_vars:
        var_id = return_var.id
        kernel_result_tensors[var_id] = None

    cuda_result_tensors[kernel_name] = kernel_result_tensors

def free_result_tensors():
    """ De-initialises the cuda_result_tensor dictionary
    
        To free up space after a single kernel execution and
        the resulting tensor results are stored back in
        tensor_map
    """
    cuda_result_tensors.clear()

def display_result_tensors():
    print(cuda_result_tensors)

def update_cuda_result_tensors(kernel_name: str, ret_arg_name: str, ret_tensor):
    """ Sets the value of the return argument tensor
    
        After launching the kernel and storing the result tensor 
        into the host device, we have to update cuda_result_tensor
        with the newly calculated return value.

        Arguments:
            kernel_name (str):
                Name of the kernel

            ret_arg_name (str):
                Name of the return argument

            ret_tensor (Tensor):
                Tensor that holds the return value of the kernel

        Returns:
            None
    """

    ret_tensor_info = cuda_result_tensor_info[kernel_name][ret_arg_name]

    ret_tensor = torch.tensor(
        np.resize(ret_tensor, ret_tensor_info[1]), 
        device = ret_tensor_info[0],
        requires_grad = ret_tensor_info[2]
    )

    cuda_result_tensors[kernel_name][ret_arg_name] = ret_tensor

def get_kernel_return_tensor(kernel_name: str, ret_arg_name: str):
    return cuda_result_tensors[kernel_name][ret_arg_name]

def create_kernel_args_indices(kernel_name, arg_list):
    """ Gets argument indices for given kernel
    
        Forms a dictionary with named tensor_indices with
        key-value pair (tensor_name, tensor_indices), which is
        the argument name and it's index in the argument list
        of the kernel.

        tensor_indices is then added to cuda_kernel_args_index
        which contains all the argument indices information of 
        all the kernels.

        Arguments:
            kernel_name (str):
                Name of the kernel
            
            arg_list (list[ArgInfo]):
                Python list containing argument information
                of the kernel

        Returns:
            None
    """

    if kernel_name in cuda_kernel_args_index.keys():
        return None

    tensor_indices = {}

    for index, arg in enumerate(arg_list):
        tensor_name = arg.name
        tensor_indices[tensor_name] = index

    cuda_kernel_args_index[kernel_name] = tensor_indices

def get_kernel_ret_indices(kernel_name):
    """ Get index of return arguments in a kernel
    
        In a given kernel, it returns a list of indices which
        corresponds to the index of each return argument in a 
        kernel function in the same order it is mentioned in 
        the kernel argument list.

        Eg. Say for the given kernel K1, this is the kernel signature
        K1(arg1, arg2, ret_arg1, ret_arg2)

        ret_arg1 and ret_arg2 are the return arguments. The return
        argument indices we would get is [2,3]

        get_kernel_ret_indices('K1') = list([2,3])
    """
    arg_indices = cuda_kernel_args_index[kernel_name]
    ret_arg_names = list(cuda_result_tensors[kernel_name].keys())
    ret_arg_indices = []

    for tensor_name, tensor_index in arg_indices.items():
        if tensor_name in ret_arg_names:
            ret_arg_indices.append(tensor_index)

    return ret_arg_indices

def get_kernel_arg_name_from_index(kernel_name, index):
    kernel_args = cuda_kernel_args_index[kernel_name]

    for arg_name, arg_index in kernel_args.items():
        if arg_index == index:
            return arg_name

    return None

def add_cuda_result_tensor_info(kernel_name: str, ret_arg_name: str, tensor_device, tensor_shape, tensor_grad: bool):
    if kernel_name not in cuda_result_tensor_info.keys():
        cuda_result_tensor_info[kernel_name] = {}

    cuda_result_tensor_info[kernel_name][ret_arg_name] = (tensor_device, tuple(tensor_shape), tensor_grad)

def create_result_tensor_from_array(array_name, result_array):
    original_result_tensor = cuda_result_tensors[array_name]
    size_result_tensor = tuple(original_result_tensor.size())

    tensor_device = original_result_tensor.device
    tensor_requires_grad = original_result_tensor.requires_grad

    new_result_tensor = torch.tensor(
        np.resize(result_array, size_result_tensor), 
        device = tensor_device,
        requires_grad = tensor_requires_grad
    )

    update_result_tensors(
        {array_name: new_result_tensor}
    )