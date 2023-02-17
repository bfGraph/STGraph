from cuda import cuda

cuInit = cuda.cuInit
cuDeviceGetCount = cuda.cuDeviceGetCount
cuCtxGetCurrent = cuda.cuCtxGetCurrent
cuDeviceGet = cuda.cuDeviceGet
cuDeviceGetName = cuda.cuDeviceGetName
cuDeviceGetAttribute = cuda.cuDeviceGetAttribute
cuMemGetInfo = cuda.cuMemGetInfo
cuCtxCreate = cuda.cuCtxCreate
cuModuleLoad = cuda.cuModuleLoad
cuCtxSynchronize = cuda.cuCtxSynchronize
cuModuleLoadData = cuda.cuModuleLoadData
cuModuleGetFunction = cuda.cuModuleGetFunction
cuStreamCreate = cuda.cuStreamCreate
cuMemAlloc = cuda.cuMemAlloc
cuMemcpyHtoD = cuda.cuMemcpyHtoD
cuMemcpyDtoH = cuda.cuMemcpyDtoH
cuMemcpyHtoDAsync = cuda.cuMemcpyHtoDAsync
cuMemcpyDtoHAsync = cuda.cuMemcpyDtoHAsync
cuStreamSynchronize = cuda.cuStreamSynchronize
cuMemFree = cuda.cuMemFree
cuLaunchKernel = cuda.cuLaunchKernel
cuCtxDestroy = cuda.cuCtxDestroy

# Macros
COMPUTE_CAPABILITY_MAJOR = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
COMPUTE_CAPABILITY_MINOR = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
MULTIPROCESSOR_COUNT = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
MAX_THREAD_PER_MULTIPROCESSOR = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
CLOCK_RATE = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE
MEMORY_CLOCK_RATE = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE