# PCSR Python Module

The ```pcsr``` python module is written using C++. The API is connected using PyBind11.

The C++ code used for implementing PCSR is used from ![Packed CSR](https://github.com/wheatman/Packed-Compressed-Sparse-Row).

## Making changes to the API

After making any changes to the API in the ```pcsr.cpp``` file, run the following command to create the ```pcsr.cpython-310-x86_64-linux-gnu.so``` file.

```
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) pcsr.cpp -g -o pcsr$(python3-config --extension-suffix)
```

Make sure to PyBind11 installed in whichever environment you are working on.

## Using the pcsr module

Like any other module, you can import ```pcsr``` as follows. Run the following code to get started.

```
import pcsr

graph = pcsr.PCSR()
graph.init_graph("./datasets/test_graph.txt")
graph.print_graph()
```

Output:

```
   0   1   2
0  •       •
1          •
2  •   •   •
```

## Methods provided by pcsr

TODO: Add documentation for each method