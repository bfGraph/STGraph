# PCSR Python implementation with C++

## Building the shared object

Run the following command while in this same directory to build the `pcsr.so` file.

```
c++ -O3 -w -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) pcsr.cpp -o pcsr.so
```