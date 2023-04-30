# PCSR Python implementation with C++

## Building the shared object

Run the following command while in this same directory to build the `pcsr.so` file.

```
/usr/local/cuda-11.7/bin/nvcc $(python3 -m pybind11 --includes) -w -shared -rdc=true --compiler-options '-fPIC' -o pcsr.so pcsr.cu
```