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

#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())

class CSR
{
public:
    thrust::device_vector<int> row_offset;
    thrust::device_vector<int> column_indices;
    thrust::device_vector<int> eids;
    std::vector<int> num_neighbours;

    CSR(std::vector<std::tuple<int, int>> edge_list, int num_nodes, bool is_edge_reverse);
    std::tuple<std::size_t, std::size_t, std::size_t> get_csr_ptrs();
    void label_edges();
    int find_edge_id(int src, int dst);
    void copy_label_edges(CSR new_csr);
    void print_csr_arrays();
};

bool sort_by_sec(const std::tuple<int, int> &a,
                 const std::tuple<int, int> &b)
{
    return (std::get<1>(a) < std::get<1>(b));
}

CSR::CSR(std::vector<std::tuple<int, int>> edge_list, int num_nodes, bool is_edge_reverse = false)
{
    if (is_edge_reverse)
        sort(edge_list.begin(), edge_list.end(), sort_by_sec);
    else
        sort(edge_list.begin(), edge_list.end());

    thrust::host_vector<int> h_row_offset;
    thrust::host_vector<int> h_column_indices;

    // initialising row_offset values all to -1
    h_row_offset.resize(num_nodes + 1);
    thrust::fill(h_row_offset.begin(), h_row_offset.end(), -1);
    h_row_offset[0] = 0;

    num_neighbours.resize(num_nodes);
    thrust::fill(num_neighbours.begin(), num_neighbours.end(), 0);

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
            h_row_offset[current_src] = beg;
            h_row_offset[current_src + 1] = end;

            current_src = src;
            beg = end;
        }

        // TODO: Not sure this is right
        // the outdegree of this graph
        num_neighbours[dst] += 1;

        // adding the dst node to the column indices
        // and incrementing the end range by 1
        h_column_indices.push_back(dst);
        end += 1;
    }

    h_row_offset[current_src + 1] = end;

    // removing the -1
    int curr_val = h_row_offset[0];
    for (int i = 1; i < h_row_offset.size(); ++i)
    {
        if (h_row_offset[i] != curr_val && h_row_offset[i] != -1)
            curr_val = h_row_offset[i];

        if (h_row_offset[i] == -1)
            h_row_offset[i] = curr_val;
    }

    row_offset = h_row_offset;
    column_indices = h_column_indices;
}

std::tuple<std::size_t, std::size_t, std::size_t> CSR::get_csr_ptrs()
{
    std::tuple<std::size_t, std::size_t, std::size_t> t;
    std::get<0>(t) = (std::size_t)RAW_PTR(row_offset);
    std::get<1>(t) = (std::size_t)RAW_PTR(column_indices);
    std::get<2>(t) = (std::size_t)RAW_PTR(eids);
    return t;
}

// TODO: Do binary search
int CSR::find_edge_id(int src, int dst)
{
    thrust::host_vector<int> h_row_offset = row_offset;
    thrust::host_vector<int> h_column_indices = column_indices;
    thrust::host_vector<int> h_eids = eids;

    int beg = h_row_offset[src];
    int end = h_row_offset[src + 1];

    for (int col_idx = beg; col_idx < end; ++col_idx)
    {
        if (h_column_indices[col_idx] == dst)
            return h_eids[col_idx];
    }

    return -1;
}

void CSR::label_edges()
{
    eids.resize(column_indices.size());
    thrust::sequence(eids.begin(), eids.end());
}

void CSR::copy_label_edges(CSR ref_csr)
{

    thrust::host_vector<int> h_row_offset = row_offset;
    thrust::host_vector<int> h_column_indices = column_indices;
    thrust::host_vector<int> h_eids(h_column_indices.size());

    int num_nodes = h_row_offset.size() - 1;
    for (int src = 0; src < num_nodes; ++src)
    {
        int beg = h_row_offset[src];
        int end = h_row_offset[src + 1];

        for (int dst_id = beg; dst_id < end; ++dst_id)
        {
            h_eids[dst_id] = ref_csr.find_edge_id(h_column_indices[dst_id], src);
        }
    }

    eids = h_eids;
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

    printf("\nColumn Indices:\n");
    for (int i = 0; i < h_col.size(); ++i)
        printf("%d ", h_col[i]);

    printf("\nEids:\n");
    for (int i = 0; i < h_eids.size(); ++i)
        printf("%d ", h_eids[i]);

    printf("\n");
}

PYBIND11_MODULE(csr, m)
{
    m.doc() = "CPython module for CSR"; // optional module docstring

    py::class_<CSR>(m, "CSR")
        .def(py::init<std::vector<std::tuple<int, int>>, int, bool>(), py::arg("edge_list"), py::arg("num_nodes"), py::arg("is_edge_reverse") = false)
        .def("label_edges", &CSR::label_edges)
        .def("copy_label_edges", &CSR::copy_label_edges)
        .def("get_csr_ptrs", &CSR::get_csr_ptrs)
        .def("print_csr_arrays", &CSR::print_csr_arrays)
        .def_readwrite("row_offset", &CSR::row_offset)
        .def_readwrite("column_indices", &CSR::column_indices)
        .def_readwrite("eids", &CSR::eids)
        .def_readwrite("num_neighbours", &CSR::num_neighbours);
}