#include "stdio.h"
#include <vector>
#include <tuple>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

class CSR
{
public:
    thrust::device_vector<int> row_offset;
    thrust::device_vector<int> column_indices;
    thrust::device_vector<int> eids;

    CSR(std::vector<std::tuple<int, int>> edge_list, int num_nodes, bool is_edge_reverse);
    int *get_csr_ptrs();
    void print_csr_arrays();
};

CSR::CSR(std::vector<std::tuple<int, int>> edge_list, int num_nodes, bool is_edge_reverse = false)
{
    printf("Hello world! This is CSR\n");

    // initialising row_offset values all to -1
    row_offset.resize(num_nodes + 1);
    thrust::fill(row_offset.begin(), row_offset.end(), -1);
    row_offset[0] = 0;

    int current_src;
    int beg = 0;
    int end = 0;

    // iterating through the edge_list
    for (auto &edge : edge_list)
    {
        int src = is_edge_reverse ? std::get<1>(edge) : std::get<0>(edge);
        int dst = is_edge_reverse ? std::get<0>(edge) : std::get<1>(edge);

        // first edge
        if (beg == 0 && end == 0)
            current_src = src;

        // source node we are dealing with changes
        if (current_src != src)
        {

            // update row_offset
            row_offset[current_src] = beg;
            row_offset[current_src + 1] = end;

            current_src = src;
            beg = end;
        }

        // adding the dst node to the column indices
        // and incrementing the end range by 1
        column_indices.push_back(dst);
        end += 1;
    }

    row_offset[current_src + 1] = end;

    // removing the -1
    int curr_val = row_offset[0];
    for (int i = 1; i < row_offset.size(); ++i)
    {
        if (row_offset[i] != curr_val && row_offset[i] != -1)
            curr_val = row_offset[i];

        if (row_offset[i] == -1)
            row_offset[i] = curr_val;
    }
}

// TODO:
int *CSR::get_csr_ptrs()
{
    return NULL;
}

void CSR::print_csr_arrays()
{
    thrust::host_vector<int> h_row = row_offset;
    thrust::host_vector<int> h_col = column_indices;

    cudaDeviceSynchronize();

    printf("\nRow offsets:\n");
    for (int i = 0; i < h_row.size(); ++i)
        printf("%d ", h_row[i]);

    printf("\n");

    printf("\nColumn Indices:\n");
    for (int i = 0; i < h_col.size(); ++i)
        printf("%d ", h_col[i]);

    printf("\n");
}

PYBIND11_MODULE(csr, m)
{
    m.doc() = "CPython module for CSR"; // optional module docstring

    py::class_<CSR>(m, "CSR")
        .def(py::init<std::vector<std::tuple<int, int>>, int, bool>(), py::arg("edge_list"), py::arg("num_nodes"), py::arg("is_edge_reverse") = false)
        .def("get_csr_ptrs", &CSR::get_csr_ptrs)
        .def("print_csr_arrays", &CSR::print_csr_arrays)
        .def_readwrite("row_offset", &CSR::row_offset)
        .def_readwrite("column_indices", &CSR::column_indices)
        .def_readwrite("eids", &CSR::eids);
}