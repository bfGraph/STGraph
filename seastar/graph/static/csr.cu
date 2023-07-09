#include "stdio.h"
#include <chrono>
#include <cstdint>
#include <tuple>
#include <vector>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

typedef thrust::device_vector<int> DEV_VEC;
#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())

#define cErr(errcode)                                                          \
  { gpuAssert((errcode), __FILE__, __LINE__); }
__inline__ __host__ __device__ void gpuAssert(cudaError_t code,
                                              const char *file, int line) {
  if (code != cudaSuccess) {
    printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  }
}

class CSR {
public:
  std::vector<int> row_offset;
  std::vector<int> column_indices;
  std::vector<int> eids;
  std::vector<int> node_ids;

  int *row_offset_device;
  int *column_indices_device;
  int *eids_device;
  int *node_ids_device;

  std::vector<int> in_degrees;
  std::vector<int> out_degrees;
  std::vector<float> weighted_out_degrees;

  std::uintptr_t row_offset_ptr;
  std::uintptr_t column_indices_ptr;
  std::uintptr_t eids_ptr;
  std::uintptr_t node_ids_ptr;

  CSR(std::vector<std::tuple<int, int, int>> edge_list,
      std::vector<float> edge_weight, int num_nodes, bool is_edge_reverse);
  void get_csr_ptrs();
};

bool sort_by_sec(const std::tuple<int, int> &a, const std::tuple<int, int> &b) {
  return std::tie(std::get<1>(a), std::get<0>(a)) <
         std::tie(std::get<1>(b), std::get<0>(b));
}

CSR::CSR(std::vector<std::tuple<int, int, int>> edge_list,
         std::vector<float> edge_weight, int num_nodes,
         bool is_edge_reverse = false) {
  // initialising row_offset values all to -1
  row_offset.resize(num_nodes + 1);
  column_indices.resize(edge_list.size());
  eids.resize(edge_list.size());
  node_ids.resize(num_nodes);

  // allocating memory on the gpu
  cudaMalloc(&row_offset_device, row_offset.size() * sizeof(int));
  cudaMalloc(&column_indices_device, column_indices.size() * sizeof(int));
  cudaMalloc(&eids_device, eids.size() * sizeof(int));
  cudaMalloc(&node_ids_device, node_ids.size() * sizeof(int));

  // node degree array initializations
  in_degrees.resize(num_nodes, 0);
  out_degrees.resize(num_nodes, 0);
  weighted_out_degrees.resize(num_nodes, 0);

  // row offset initial values assigned
  std::fill(row_offset.begin(), row_offset.end(), -1);
  row_offset[0] = 0;

  int current_src;
  int beg = 0;
  int end = 0;

  // iterating through the edge_list
  for (int i = 0; i < edge_list.size(); ++i) {
    int src =
        is_edge_reverse ? std::get<1>(edge_list[i]) : std::get<0>(edge_list[i]);
    int dst =
        is_edge_reverse ? std::get<0>(edge_list[i]) : std::get<1>(edge_list[i]);
    int eid = std::get<2>(edge_list[i]);

    // first edge
    if (beg == 0 && end == 0)
      current_src = src;

    // source node we are dealing with changes
    if (current_src != src) {
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
    weighted_out_degrees[src] += edge_weight[eid];
  }

  row_offset[current_src + 1] = end;

  // removing the -1
  int curr_val = row_offset[0];
  for (int i = 1; i < row_offset.size(); ++i) {
    if (row_offset[i] != curr_val && row_offset[i] != -1)
      curr_val = row_offset[i];

    if (row_offset[i] == -1)
      row_offset[i] = curr_val;
  }

  // Obtaining the sorted order of node ids (in descending order)
  std::vector<std::pair<int, int>> degree_id_pairs =
      std::vector<std::pair<int, int>>();
  for (int i = 0; i < out_degrees.size(); ++i) {
    degree_id_pairs.push_back(std::make_pair(out_degrees[i], i));
  }

  std::sort(degree_id_pairs.begin(), degree_id_pairs.end(),
            [](const std::pair<int, int> &lhs, const std::pair<int, int> &rhs) {
              return lhs.first > rhs.first;
            });

  for (int i = 0; i < degree_id_pairs.size(); ++i) {
    node_ids[i] = degree_id_pairs[i].second;
  }

  get_csr_ptrs();
}

void CSR::get_csr_ptrs() {
  cErr(cudaMemcpy(row_offset_device, row_offset.data(),
                  row_offset.size() * sizeof(int), cudaMemcpyHostToDevice));
  cErr(cudaMemcpy(column_indices_device, column_indices.data(),
                  column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
  cErr(cudaMemcpy(eids_device, eids.data(), eids.size() * sizeof(int),
                  cudaMemcpyHostToDevice));
  cErr(cudaMemcpy(node_ids_device, node_ids.data(),
                  node_ids.size() * sizeof(int), cudaMemcpyHostToDevice));

  row_offset_ptr = reinterpret_cast<std::uintptr_t>(row_offset_device);
  column_indices_ptr = reinterpret_cast<std::uintptr_t>(column_indices_device);
  eids_ptr = reinterpret_cast<std::uintptr_t>(eids_device);
  node_ids_ptr = reinterpret_cast<std::uintptr_t>(node_ids_device);
}

std::vector<int> get_array(std::uintptr_t ptr, int size) {
  int *dev_ptr = reinterpret_cast<int *>(ptr);
  int *host_ptr = (int *)malloc(size * sizeof(int));
  std::vector<int> vec;
  
  vec.resize(size);
  cudaMemcpy(vec.data(), dev_ptr, size * sizeof(int), cudaMemcpyDeviceToHost);

  /* guaranteed named value return optimization */
  return vec;
}

PYBIND11_MODULE(csr, m) {
  m.doc() = "CPython module for CSR"; // optional module docstring
  m.def("get_array", &get_array, "Get Array");

  py::class_<CSR>(m, "CSR")
      .def(py::init<std::vector<std::tuple<int, int, int>>, std::vector<float>,
                    int, bool>(),
           py::arg("edge_list"), py::arg("edge_weight"), py::arg("num_nodes"),
           py::arg("is_edge_reverse") = false)
      
      /* public members: cuda pointers */
      
      .def_readwrite("row_offset_ptr", &CSR::row_offset_ptr)
      .def_readwrite("column_indices_ptr", &CSR::column_indices_ptr)
      .def_readwrite("eids_ptr", &CSR::eids_ptr)
      .def_readwrite("node_ids_ptr", &CSR::node_ids_ptr)
      
      .def_readwrite("weighted_out_degrees", &CSR::weighted_out_degrees)
      .def_readwrite("out_degrees", &CSR::out_degrees)
      .def_readwrite("in_degrees", &CSR::in_degrees)
      .def("__copy__", [](const CSR &self) { return CSR(self); })
      .def(
          "__deepcopy__", [](const CSR &self, py::dict) { return CSR(self); },
          "memo"_a);
}