# GPMA 

Seastar provides GPMA abstraction to represent dynamic graphs in GPU.

## Building GPMA CUDA/C++ extension

We use PyBind11 to provide Python bindings to the GPMA API written fully in C++/CUDA. You can run the following command to build the `gpma.so` file which can be imported into your python program.

```
/usr/local/cuda-11.7/bin/nvcc $(python3 -m pybind11 --includes) -w -shared -rdc=true --compiler-options '-fPIC' -o gpma.so gpma.cu
```