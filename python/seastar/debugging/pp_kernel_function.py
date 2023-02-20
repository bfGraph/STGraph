"""Pretty print kernel function information

Used for debugging during development and for learning purposes.
Pretty prints out kernel information before making a call to the
cuda.cuLaunchKernel() function.
"""

import textwrap

from prettytable import PrettyTable, HEADER, NONE, SINGLE_BORDER
from termcolor import colored

from .pp_table_generator import genBorderlessTable

kernel_argument_names = {}
kernel_common_argument_names = [
    "row_offsets", "eids", "column_indices", "num_nodes", "max_dimx", "max_dimy",
]

# TODO:
def pp_kernel_function(kernel, argument_list: list):
    """Pretty print a table containing information about the Kernel function
    
    Prints a table containing the following kernel information:
    1. Kernel Name
    2. gridDimX, gridDimY, gridDimZ
    3. blockDimx, blockDimY, blockDimZ

    The second table displays the following information of the arguments 
    passed to the kernel:
    1. Argument Name
    2. Dimensions/Length
    3. Value of the argument

    Args:
        kernel (class Kernel):
            A class Kernel instance
        
        argument_list (list):
            A list containing the arguments passed to the kernel

    Returns:
        None

    """

    kernel_info_table = genBorderlessTable()
    kernel_info_table.field_names = ["Property", "Value"]
    kernel_info_table.align["Property"] = "r"
    kernel_info_table.align["Value"] = "l"

    kernel_info_table.add_row([colored("Kernel Name", "dark_grey"), kernel.kernel_name])
    kernel_info_table.add_row(["",""])
    kernel_info_table.add_row([colored("gridDimX", "dark_grey"), kernel.launch_config[0]])
    kernel_info_table.add_row([colored("gridDimY", "dark_grey"), kernel.launch_config[1]])
    kernel_info_table.add_row([colored("gridDimZ", "dark_grey"), kernel.launch_config[2]])
    kernel_info_table.add_row(["",""])
    kernel_info_table.add_row([colored("blockDimX", "dark_grey"), kernel.launch_config[3]])
    kernel_info_table.add_row([colored("blockDimY", "dark_grey"), kernel.launch_config[4]])
    kernel_info_table.add_row([colored("blockDimZ", "dark_grey"), kernel.launch_config[5]])

    print("\n")
    print(kernel_info_table)
    print("\n")

    kernel_args_table = genBorderlessTable()
    kernel_args_table.field_names = ["SNo.", "Argument", "Dimensions", "Value"]
    kernel_args_table.align["SNo."] = "l"
    kernel_args_table.align["Argument"] = "r"
    kernel_args_table.align["Dimensions"] = "l"
    kernel_args_table.align["Value"] = "l"

    wrapper = textwrap.TextWrapper(width=80)

    for index, argument in enumerate(argument_list):
        arg_row = []
        kernel_vector_argument_names = kernel_argument_names[kernel.kernel_name]
        count_vector_arguments = len(kernel_vector_argument_names)

        if type(argument) == int:
            argument = [argument]
        else:
            argument = argument.flatten().tolist()
        
        arg_row.append(colored(str(index+1), "dark_grey"))

        if index < count_vector_arguments:
            arg_row.append(colored(kernel_argument_names[kernel.kernel_name][index]))
        else:
            arg_row.append(colored(kernel_common_argument_names[index-count_vector_arguments]))

        arg_row.append(colored(len(argument), "dark_grey"))
        arg_row.append(colored(wrapper.fill(text=str(argument)), "dark_grey"))

        kernel_args_table.add_row(arg_row)
        kernel_args_table.add_row(["", "", "", ""])

    print(kernel_args_table)
    print("\n")

def store_kernel_argument_names(kernel_name, argument_list, template_name):
    """Stores the names of the kernel arguments in a global dictionary
    
    Get the argument names of the kernel in the right order its
    defined. The argument names are stored in the global variable
    dictionary kernel_argument_names as the key-value pair (kernel_name, [argument_names])

    There are 6 common kernel arguments shared by all kernels. The last two
    varies depending on the template used (FA or V2). If it's a FA template, then
    thrs_per_group and nodes_per_block will be added to the common kernel arguments,
    else if it is V2, then tile_sizex and tile_sizey will be added

    Args:
        kernel_name (str):
            Name of the kernel

        argument_list (list):
            List of ArgInfo() class objects

        template_name (str):
            Name of the template used to generate the
            CUDA Kernel

    Returns:
        None
    """
    
    kernel_argument_names[kernel_name] = []
    
    # adding the vector argument names
    for argument in argument_list:
        kernel_argument_names[kernel_name].append(argument.name)

    # adding the common argument names depending on the template used
    if template_name == "fa":
        kernel_common_argument_names.append("thrs_per_group")
        kernel_common_argument_names.append("nodes_per_block")
    else:
        kernel_common_argument_names.append("tile_sizex")
        kernel_common_argument_names.append("tile_sizey")