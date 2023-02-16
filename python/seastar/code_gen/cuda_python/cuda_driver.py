from cuda import cuda, nvrtc

cuInit = cuda.cuInit
cuDeviceGetCount = cuda.cuDeviceGetCount
cuCtxGetCurrent = cuda.cuCtxGetCurrent
cuDeviceGet = cuda.cuDeviceGet
cuCtxCreate = cuda.cuCtxCreate
cuModuleLoad = cuda.cuModuleLoad
cuCtxSynchronize = cuda.cuCtxSynchronize
cuModuleGetFunction = cuda.cuModuleGetFunction
cuMemAlloc = cuda.cuMemAlloc
cuMemcpyHtoD = cuda.cuMemcpyHtoD
cuMemcpyDtoH = cuda.cuMemcpyDtoH
cuMemFree = cuda.cuMemFree
cuLaunchKernel = cuda.cuLaunchKernel
cuCtxDestroy = cuda.cuCtxDestroy