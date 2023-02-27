import os

from cuda import cuda
from prettytable import PrettyTable, HEADER, NONE, SINGLE_BORDER
from termcolor import colored

from .cuda_driver import *

global_gpu_device = None

class DeviceInfo:
    def __init__(self, print_log=False):
        # breakpoint()
        self.nGpus = 0
        self.name = ""
        self.cc_major = 0
        self.cc_minor = 0
        self.cores = 0
        self.thread_per_core = 0
        self.gpu_clockrate = 0
        self.memory_clockrate = 0
        self.freeMem = 0
        self.totalMem = 0

        # Retrieving the path of nvcc
        self.nvcc_path = os.popen('which nvcc').read()[:-1]
        if not len(self.nvcc_path):
            self.nvcc_path = '/usr/local/cuda/bin/nvcc'

        # Make sure to call cuInit(), otherwise we won't be
        # able to make any CUDA Driver API calls
        cuInit(0)

        # Getting the number of compatible GPU devices
        err, self.nGpus = cuDeviceGetCount()    

        # Going to get certain parameters for only the first GPU
        # i.e ordinal = 0 for cuDeviceGet
        ordinal = 0
        err, self.device = cuDeviceGet(ordinal)
        err, self.context = cuCtxCreate(0, self.device)

        err, device_name = cuDeviceGetName(25, self.device)
        self.name = device_name.decode("utf-8")
        
        err, self.cc_major = cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR, self.device)
        err, self.cc_minor = cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR, self.device)
        err, self.cores = cuDeviceGetAttribute(MULTIPROCESSOR_COUNT, self.device)
        err, self.thread_per_core = cuDeviceGetAttribute(MAX_THREAD_PER_MULTIPROCESSOR, self.device)
        err, self.gpu_clockrate = cuDeviceGetAttribute(CLOCK_RATE, self.device)
        err, self.memory_clockrate = cuDeviceGetAttribute(MEMORY_CLOCK_RATE, self.device)
        err, self.freeMem, self.totalMem = cuMemGetInfo()

        if print_log:
            self.log()

    def log(self):

        log_table = PrettyTable()
        log_table.field_names = ["Device Property", "Value"]
        log_table.align["Device Property"] = "r"
        log_table.align["Value"] = "l"
        log_table.set_style(SINGLE_BORDER)

        log_table.border = True
        log_table.hrules = HEADER
        log_table.vrules = NONE

        log_table.add_row(["Number of Devices", self.nGpus])
        log_table.add_row(["Name", self.name])
        log_table.add_row(["Compute Capability", str(self.cc_major) + "." + str(self.cc_minor)])
        log_table.add_row(["Multiprocessor Count", self.cores])
        log_table.add_row(["Concurrent Threads", self.cores*self.thread_per_core])
        log_table.add_row(["GPU Clock", str(self.gpu_clockrate/1000) + colored(" MHz", "dark_grey")])
        log_table.add_row(["Memory Clock", str(self.memory_clockrate/1000) + colored(" MHz", "dark_grey")])
        log_table.add_row(["Total Memory", str(self.totalMem/1024**2) + colored(" MiB", "dark_grey")])
        log_table.add_row(["Free Memory", str(self.freeMem/1024**2) + colored(" MiB", "dark_grey")])

        print("\n")
        print(log_table)
        print(colored("\nNote: In case either Total Memory or Free Memory is showing 0\n      it is because no context has been loaded into device", "dark_grey"))
        print("\n")

def get_global_gpu_device():
    if global_gpu_device == None:
        return DeviceInfo(print_log=False)
    return global_gpu_device

if __name__ == "__main__":
    device = DeviceInfo(print_log=True)
