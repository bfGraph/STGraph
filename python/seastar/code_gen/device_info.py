import sys
import ctypes
import os

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36




class deviceinfo():

    def __init__(self):
        libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
        for libname in libnames:
            try:
                cuda = ctypes.CDLL(libname)
            except OSError:
                continue
            else:
                break
        else:
            raise OSError("could not load any of: " + ' '.join(libnames))

        self.nGpus = ctypes.c_int()
        self.name = b' ' * 100
        self.cc_major = ctypes.c_int()
        self.cc_minor = ctypes.c_int()
        self.cores = ctypes.c_int()
        self.threads_per_core = ctypes.c_int()
        self.clockrate = ctypes.c_int()
        self.freeMem = ctypes.c_size_t()
        self.totalMem = ctypes.c_size_t()
        print(0)

        #nvcc path
        self.nvcc_path = os.popen('which nvcc').read()[:-1]
        if not len(self.nvcc_path):
            self.nvcc_path = '/usr/local/cuda/bin/nvcc'


        device = ctypes.c_int()

        cuda.cuInit(0)
        print(1)

        cuda.cuDeviceGetCount(ctypes.byref(self.nGpus))
        print("Found %d device(s)." % self.nGpus.value)
        #only count first GPU
        for i in range(1):
            cuda.cuDeviceGet(ctypes.byref(device), i)

            cuda.cuDeviceGetName(ctypes.c_char_p(self.name), len(self.name), device)
            cuda.cuDeviceComputeCapability(ctypes.byref(self.cc_major), ctypes.byref(self.cc_minor), device)
            cuda.cuDeviceGetAttribute(ctypes.byref(self.cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)
            cuda.cuDeviceGetAttribute(ctypes.byref(self.threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device)
            cuda.cuDeviceGetAttribute(ctypes.byref(self.clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device)
            cuda.cuDeviceGetAttribute(ctypes.byref(self.clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device)
            cuda.cuMemGetInfo(ctypes.byref(self.freeMem), ctypes.byref(self.totalMem))
        print(2)

    def log(self):
        print("Number of device : %d" % (self.nGpus.value))
        print("Name: %s" % (self.name.split(b'\0', 1)[0].decode()))
        print("Compute Capability %d.%d" % (self.cc_major.value, self.cc_minor.value))
        print("Multiprocessor : %d" %(self.cores.value)) 
        print("Concurrent threads: %d" % (self.cores.value * self.threads_per_core.value))
        print("GPU clock: %g MHz" % (self.clockrate.value / 1000.))
        print("Memory clock: %g MHz" % (self.clockrate.value / 1000.))
        print("Total Memory: %ld MiB" % (self.totalMem.value / 1024**2))
        print("Free Memory: %ld MiB" % (self.freeMem.value / 1024**2))



if __name__=="__main__":
    device = deviceinfo()
    device.log()
