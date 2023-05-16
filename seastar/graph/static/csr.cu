#include "stdio.h"
#include <vector>
#include <tuple>
#include <chrono>
#include <cstdint>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

typedef thrust::device_vector<int> DEV_VEC;
#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())

class CSR
{
public:
    thrust::host_vector<int> row_offset;
    thrust::host_vector<int> column_indices;
    thrust::host_vector<int> eids;

    DEV_VEC row_offset_device;
    DEV_VEC column_indices_device;
    DEV_VEC eids_device;

    std::vector<int> in_degrees;
    std::vector<int> out_degrees;

    std::uintptr_t row_offset_ptr;
    std::uintptr_t column_indices_ptr;
    std::uintptr_t eids_ptr;

    CSR(std::vector<std::tuple<int, int, int>> edge_list, int num_nodes, bool is_edge_reverse);
    // std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t> get_csr_ptrs();
    void get_csr_ptrs();
    // void label_edges();
    // int find_edge_id(int src, int dst);
    // void copy_label_edges(CSR new_csr);
    void print_row_offset();
    void print_csr_arrays();
    void print_graph();
};

bool sort_by_sec(const std::tuple<int, int> &a,
                 const std::tuple<int, int> &b)
{
    return std::tie(std::get<1>(a), std::get<0>(a)) <
           std::tie(std::get<1>(b), std::get<0>(b));
}

CSR::CSR(std::vector<std::tuple<int, int, int>> edge_list, int num_nodes, bool is_edge_reverse = false)
{

    // initialising row_offset values all to -1
    row_offset.resize(num_nodes + 1);
    column_indices.resize(edge_list.size());
    eids.resize(edge_list.size());
    thrust::fill(row_offset.begin(), row_offset.end(), -1);
    row_offset[0] = 0;

    in_degrees.resize(num_nodes, 0);
    out_degrees.resize(num_nodes, 0);

    int current_src;
    int beg = 0;
    int end = 0;

    // iterating through the edge_list
    for (int i = 0; i < edge_list.size(); ++i)
    {
        int src = is_edge_reverse ? std::get<1>(edge_list[i]) : std::get<0>(edge_list[i]);
        int dst = is_edge_reverse ? std::get<0>(edge_list[i]) : std::get<1>(edge_list[i]);
        int eid = std::get<2>(edge_list[i]);

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
        column_indices[i] = dst;
        eids[i] = eid;
        end += 1;

        // updating the degree arrays
        out_degrees[src] += 1;
        in_degrees[dst] += 1;
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

    get_csr_ptrs();
}

void CSR::get_csr_ptrs()
{
    row_offset_device = row_offset;
    column_indices_device = column_indices;
    eids_device = eids;

    row_offset_ptr = reinterpret_cast<std::uintptr_t>(RAW_PTR(row_offset_device));
    column_indices_ptr = reinterpret_cast<std::uintptr_t>(RAW_PTR(column_indices_device));
    eids_ptr = reinterpret_cast<std::uintptr_t>(RAW_PTR(eids_device));
}

void CSR::print_row_offset()
{
    int *dev_ptr = reinterpret_cast<int *>(row_offset_ptr);
    int *host_ptr = (int *)malloc(row_offset.size() * sizeof(int));
    cudaMemcpy(host_ptr, dev_ptr, row_offset.size() * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "PRINTING DEV AFTER RECAST AS SEP FUNC\n";
    for (int i = 0; i < row_offset.size(); ++i)
    {
        std::cout << host_ptr[i] << " ";
    }

    std::cout << "\n=============================\n";
}

void CSR::print_csr_arrays()
{
    thrust::host_vector<int> h_row = row_offset;
    thrust::host_vector<int> h_col = column_indices;
    thrust::host_vector<int> h_eids = eids;

    cudaDeviceSynchronize();

    printf("\nRow offsets:\n");
    for (int i = 0; i < h_row.size(); ++i)
        printf("%d ", h_row[i]);

    printf("\n");

    printf("\nOut-degrees:\n");
    for (int i = 0; i < out_degrees.size(); ++i)
        printf("%d ", out_degrees[i]);

    printf("\nIn-degrees:\n");
    for (int i = 0; i < in_degrees.size(); ++i)
        printf("%d ", in_degrees[i]);

    printf("\nColumn Indices:\n");
    for (int i = 0; i < h_col.size(); ++i)
        printf("%d ", h_col[i]);

    printf("\nEids:\n");
    for (int i = 0; i < h_eids.size(); ++i)
        printf("%d ", h_eids[i]);

    printf("\n");
}

void CSR::print_graph()
{
    thrust::host_vector<int> h_row_offset = row_offset;
    thrust::host_vector<int> h_column_indices = column_indices;
    thrust::host_vector<int> h_eids = eids;

    int num_vertices = h_row_offset.size() - 1;

    // printing the graph matrix column indices
    for (int dst = 0; dst < num_vertices; ++dst)
        std::cout << "   " << dst << std::flush;

    printf("\n");

    for (int src = 0; src < num_vertices; ++src)
    {
        std::cout << src << " " << std::flush;
        int matrix_index = 0;

        int beg = h_row_offset[src];
        int end = h_row_offset[src + 1];

        for (int dst_idx = beg; dst_idx < end; ++dst_idx)
        {
            while (matrix_index < h_column_indices[dst_idx])
            {
                std::cout << "    " << std::flush;
                matrix_index++;
            }
            std::cout << " " << h_eids[dst_idx] << "  " << std::flush;
            matrix_index++;
        }

        for (int j = matrix_index; j < num_vertices; ++j)
            std::cout << "    " << std::flush;

        printf("\n");
    }
}

void print_dev_array(std::uintptr_t ptr, int size)
{
    std::cout << "RECEIVED ROW OFFSET PTR: " << ptr << "\n";
    int *dev_ptr = reinterpret_cast<int *>(ptr);
    int *host_ptr = (int *)malloc(size * sizeof(int));
    // thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(raw_ptr);
    cudaMemcpy(host_ptr, dev_ptr, size * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "\nPRINTING ARRAY\n";
    for (int i = 0; i < size; ++i)
    {
        std::cout << host_ptr[i] << " ";
    }
    std::cout << "\n";
}

PYBIND11_MODULE(csr, m)
{
    m.doc() = "CPython module for CSR"; // optional module docstring
    m.def("print_dev_array", &print_dev_array, "Print arrays on the device");

    py::class_<CSR>(m, "CSR")
        // .def(py::init<std::vector<std::tuple<int, int>>, int, bool>(), py::arg("edge_list"), py::arg("num_nodes"), py::arg("is_edge_reverse") = false)
        .def(py::init<std::vector<std::tuple<int, int, int>>, int, bool>(), py::arg("edge_list"), py::arg("num_nodes"), py::arg("is_edge_reverse") = false)
        // .def("label_edges", &CSR::label_edges)
        // .def("copy_label_edges", &CSR::copy_label_edges)
        .def("get_csr_ptrs", &CSR::get_csr_ptrs)
        .def("print_csr_arrays", &CSR::print_csr_arrays)
        .def("print_graph", &CSR::print_graph)
        .def("print_row_offset", &CSR::print_row_offset)
        .def_readwrite("row_offset", &CSR::row_offset)
        .def_readwrite("column_indices", &CSR::column_indices)
        .def_readwrite("eids", &CSR::eids)
        .def_readwrite("row_offset_ptr", &CSR::row_offset_ptr)
        .def_readwrite("column_indices_ptr", &CSR::column_indices_ptr)
        .def_readwrite("eids_ptr", &CSR::eids_ptr)
        .def_readwrite("out_degrees", &CSR::out_degrees)
        .def_readwrite("in_degrees", &CSR::in_degrees)
        .def("__copy__", [](const CSR &self)
             { return CSR(self); })
        .def(
            "__deepcopy__", [](const CSR &self, py::dict)
            { return CSR(self); },
            "memo"_a);
}