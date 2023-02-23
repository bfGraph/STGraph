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
cuda_result_tensors = {}


cuda_kernel_tensor_args_index = {}

def init_result_tensors(return_vars: list[Var]):
    """ Initialises the cuda_result_tensor dictionary
    
        Initialises the cuda_result_tensor dictionary for the current
        kernel by setting the value to None.

        If a certain kernel K0 has V2 as its return tensor, then 
        cuda_result_tensor would look like {'V2': None}
    """
    return_var: Var
    for return_var in return_vars:
        var_id = return_var.id
        cuda_result_tensors[var_id] = None

    breakpoint()

def free_result_tensors():
    """ De-initialises the cuda_result_tensor dictionary
    
        To free up space after a single kernel execution and
        the resulting tensor results are stored back in
        tensor_map
    """
    cuda_result_tensors.clear()

def display_result_tensors():
    print(cuda_result_tensors)

def update_result_tensors(return_tensors: dict):
    """
        TODO: Add docs
    """
    for tensor_name, tensor in return_tensors.items():
        # ()
        cuda_result_tensors[tensor_name] = tensor

    # print("🎲 cuda_result_tensor has been updated: ")
    # display_result_tensors()

def get_kernel_tensor_args_indices(kernel_name, arg_list):
    """
        TODO: Add documentation
    """

    tensor_indices = {}

    for index, arg in enumerate(arg_list):
        tensor_name = arg.name
        tensor_indices[tensor_name] = index

    cuda_kernel_tensor_args_index[kernel_name] = tensor_indices

    # print("🎲 Tensor indices created: ")
    # print(cuda_kernel_tensor_args_index)

def get_kernel_ret_tensor_indices(kernel_name):
    tensor_arg_indices = cuda_kernel_tensor_args_index[kernel_name]
    ret_tensor_names = list(cuda_result_tensors.keys())
    ret_tensor_indices = []

    for tensor_name, tensor_index in tensor_arg_indices.items():
        if tensor_name in ret_tensor_names:
            ret_tensor_indices.append(tensor_index)

    return ret_tensor_indices

def get_kernel_arg_name_from_index(kernel_name, index):
    kernel_args = cuda_kernel_tensor_args_index[kernel_name]

    for arg_name, arg_index in kernel_args.items():
        if arg_index == index:
            return arg_name

    return None

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