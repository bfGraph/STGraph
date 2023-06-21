#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <queue>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std;

using namespace std::chrono;

#define cErr(errcode)                             \
    {                                             \
        gpuAssert((errcode), __FILE__, __LINE__); \
    }
__inline__ __host__ __device__ void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

////////////////////////////////////////////////////////////////////////////

typedef struct _node
{
    // beginning and end of the associated region in the edge list
    uint32_t beginning;     // deleted = max int
    uint32_t end;           // end pointer is exclusive
    uint32_t num_neighbors; // number of edges with this node as source
    uint32_t in_degree;     // in-degree of a node -  number of edges going into the node

    _node(int beg = 0, int _end = 0, int num_neigh = 0, int in_deg = 0)
    {
        beginning = beg;
        end = _end;
        num_neighbors = num_neigh;
        in_degree = in_deg;
    }
} node_t;

typedef struct _edge
{
    uint32_t dest;  // destination of this edge in the graph, MAX_INT if this is a sentinel
    uint32_t value; // edge value of zero means it a null since we don't store 0 edges

    _edge(int _dest = 0, int _value = 0)
    {
        dest = _dest;
        value = _value;
    }
} edge_t;

typedef struct edge_list
{
    int N;
    int H;
    int logN;
    vector<edge_t> items;

    edge_list()
    {
        N = 0;
        H = 0;
        logN = 0;

        vector<edge_t> temp(0, 0);
        items = temp;
    }
} edge_list_t;

typedef struct _pair_int
{
    int x; // length in array
    int y; // depth

    _pair_int(int _x = 0, int _y = 0)
    {
        x = _x;
        y = _y;
    }
} pair_int;

typedef struct _pair_double
{
    double x;
    double y;

    _pair_double(double _x = 0, double _y = 0)
    {
        x = _x;
        y = _y;
    }
} pair_double;

////////////////////////////////////////////////////////////////////////////

static inline int bsf_word(int word)
{
    int result;
    __asm__ volatile("bsf %1, %0"
                     : "=r"(result)
                     : "r"(word));
    return result;
}

static inline int bsr_word(int word)
{
    int result;
    __asm__ volatile("bsr %1, %0"
                     : "=r"(result)
                     : "r"(word));
    return result;
}

// given index, return the starting index of the leaf it is in
int find_leaf(edge_list_t *list, int index)
{
    return (index / list->logN) * list->logN;
}

bool is_null(edge_t e) { return e.value == 0; }

bool is_sentinel(edge_t e)
{
    return e.dest == UINT32_MAX || e.value == UINT32_MAX;
}

// Possibly make this faster
uint32_t binary_search(edge_list_t *list, edge_t *elem, uint32_t start,
                       uint32_t end)
{
    while (start + 1 < end)
    {
        uint32_t mid = (start + end) / 2;

        edge_t item = list->items[mid];
        uint32_t change = 1;
        uint32_t check = mid;

        bool flag = true;
        while (is_null(item) && flag)
        {
            flag = false;
            check = mid + change;
            if (check < end)
            {
                flag = true;
                if (check <= end)
                {
                    item = list->items[check];
                    if (!is_null(item))
                    {
                        break;
                    }
                    else if (check == end)
                    {
                        break;
                    }
                }
            }
            check = mid - change;
            if (check >= start)
            {
                flag = true;
                item = list->items[check];
            }
            change++;
        }

        if (is_null(item) || start == check || end == check)
        {
            if (!is_null(item) && start == check && elem->dest <= item.dest)
            {
                return check;
            }
            return mid;
        }

        // if we found it, return
        if (elem->dest == item.dest)
        {
            return check;
        }
        else if (elem->dest < item.dest)
        {
            end =
                check; // if the searched for item is less than current item, set end
        }
        else
        {
            start = check;
            // otherwise, searched for item is more than current and we set start
        }
    }
    if (end < start)
    {
        start = end;
    }
    // handling the case where there is one element left
    // if you are leq, return start (index where elt is)
    // otherwise, return end (no element greater than you in the range)
    // printf("start = %d, end = %d, n = %d\n", start,end, list->N);
    if (elem->dest <= list->items[start].dest && !is_null(list->items[start]))
    {
        return start;
    }
    return end;
}

// get density of a node
double get_density(edge_list_t *list, int index, int len)
{
    int full = 0;
    for (int i = index; i < index + len; i++)
    {
        full += (!is_null(list->items[i]));
    }
    double full_d = (double)full;
    return full_d / len;
}

int find_node(int index, int len) { return (index / len) * len; }

pair_double density_bound(edge_list_t *list, int depth)
{
    pair_double pair;

    // between 1/4 and 1/2
    // pair.x = 1.0/2.0 - (( .25*depth)/list->H);
    // between 1/8 and 1/4
    pair.x = 1.0 / 4.0 - ((.125 * depth) / list->H);
    pair.y = 3.0 / 4.0 + ((.25 * depth) / list->H);
    return pair;
}

bool edge_equals(edge_t e1, edge_t e2)
{
    return e1.dest == e2.dest && e1.value == e2.value;
}

uint32_t find_elem_pointer(edge_list_t *list, uint32_t index, edge_t elem)
{
    edge_t item = list->items[index];
    while (!edge_equals(item, elem))
    {
        item = list->items[++index];
    }
    return index;
}

////////////////////////////////////////////////////////////////////////////

class PCSR
{
public:
    // data members
    std::vector<node_t> nodes;
    std::vector<uint32_t> in_degrees;
    std::vector<uint32_t> out_degrees;

    edge_list_t edges;
    uint32_t edge_count;

    // replacing device vectors
    uint32_t *row_offset_pinned;
    uint32_t *column_indices_pinned;
    uint32_t *eids_pinned;

    // replacing device vectors
    uint32_t *row_offset_device;
    uint32_t *column_indices_device;
    uint32_t *eids_device;

    // member functions
    PCSR(uint32_t init_n, uint32_t max_edge_count);
    // ~PCSR();
    
    // PCSR specific internal functions
    uint64_t get_n();
    void add_node();
    uint32_t insert(uint32_t index, edge_t elem, uint32_t src);
    void double_list();
    int slide_right(int index);
    void slide_left(int index);
    void fix_sentinel(int32_t node_index, int in);
    void redistribute(int index, int len);
    void add_edge(uint32_t src, uint32_t dest, uint32_t value);
    void add_edge_update(uint32_t src, uint32_t dest, uint32_t value);
    void delete_edge(uint32_t src, uint32_t dest);

    // exposed APIs
    void edge_update_list(std::vector<std::tuple<uint32_t, uint32_t>> edge_list, bool is_delete, bool is_reverse_edge);
    void label_edges();
    void build_csr();
    void build_reverse_csr();
    std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t> get_csr_ptrs();
    vector<tuple<uint32_t, uint32_t, uint32_t>> get_edges();
    void move_pinned_to_gpu();
};

////////////////////////////////////////////////////////////////////////////

PCSR::PCSR(uint32_t init_n, uint32_t max_num_edges)
{
    if (init_n != 0)
    {
        edges.N = 2 << bsr_word(init_n);
        edges.logN = (1 << bsr_word(bsr_word(edges.N) + 1));
        edges.H = bsr_word(edges.N / edges.logN);

        edges.items.resize(edges.N);
        edge_count = 0;

        for (int i = 0; i < edges.N; i++)
        {
            edge_t new_edge(0, 0);
            edges.items[i] = new_edge;
        }

        for (int i = 0; i < init_n; i++)
        {
            add_node();
        }

        in_degrees.resize(init_n);
        out_degrees.resize(init_n);

        cErr(cudaMallocHost(&row_offset_pinned, sizeof(uint32_t) * (init_n + 1)));
        cErr(cudaMallocHost(&column_indices_pinned, sizeof(uint32_t) * max_num_edges));
        cErr(cudaMallocHost(&eids_pinned, sizeof(uint32_t) * max_num_edges));

        cErr(cudaMalloc(&row_offset_device, sizeof(uint32_t) * (init_n + 1)));
        cErr(cudaMalloc(&column_indices_device, sizeof(uint32_t) * max_num_edges));
        cErr(cudaMalloc(&eids_device, sizeof(uint32_t) * max_num_edges));
    }
}

// PCSR::~PCSR()
// {
//     cErr(cudaFree(row_offset_pinned));
//     cErr(cudaFree(column_indices_pinned));
//     cErr(cudaFree(eids_pinned));

//     cErr(cudaFree(row_offset_device));
//     cErr(cudaFree(column_indices_device));
//     cErr(cudaFree(eids_device));

//     nodes.clear();
//     in_degrees.clear();
//     out_degrees.clear();
// }

// add a node to the graph
void PCSR::add_node()
{
    node_t node;
    int len = nodes.size();

    edge_t sentinel;
    sentinel.dest = UINT32_MAX; // placeholder
    sentinel.value = len;       // back pointer

    if (len > 0)
    {
        node.beginning = nodes[len - 1].end;
        node.end = node.beginning + 1;
    }
    else
    {
        node.beginning = 0;
        node.end = 1;
        sentinel.value = UINT32_MAX;
    }
    node.num_neighbors = 0;

    nodes.push_back(node);
    insert(node.beginning, sentinel, nodes.size() - 1);
}

uint32_t PCSR::insert(uint32_t index, edge_t elem, uint32_t src)
{
    int node_index = find_leaf(&edges, index);
    int level = edges.H;
    int len = edges.logN;

    // always deposit on the left
    if (is_null(edges.items[index]))
    {
        edges.items[index].value = elem.value;
        edges.items[index].dest = elem.dest;
    }
    else
    {
        // if the edge already exists in the graph, update its value
        // do not make another edge
        // return index of the edge that already exists
        if (!is_sentinel(elem) && edges.items[index].dest == elem.dest)
        {
            edges.items[index].value = elem.value;
            return index;
        }
        if (index == edges.N - 1)
        {
            // when adding to the end double then add edge
            double_list();
            node_t node = nodes[src];
            uint32_t loc_to_add =
                binary_search(&edges, &elem, node.beginning + 1, node.end);
            return insert(loc_to_add, elem, src);
        }
        else
        {
            if (slide_right(index) == -1)
            {
                index -= 1;
                slide_left(index);
            }
        }
        edges.items[index].value = elem.value;
        edges.items[index].dest = elem.dest;
    }

    double density = get_density(&edges, node_index, len);

    // spill over into next level up, node is completely full.
    if (density == 1)
    {
        node_index = find_node(node_index, len * 2);
        redistribute(node_index, len * 2);
    }
    else
    {
        // makes the last slot in a section empty so you can always slide right
        redistribute(node_index, len);
    }

    // get density of the leaf you are in
    pair_double density_b = density_bound(&edges, level);
    density = get_density(&edges, node_index, len);

    // while density too high, go up the implicit tree
    // go up to the biggest node above the density bound
    while (density >= density_b.y)
    {
        len *= 2;
        if (len <= edges.N)
        {
            level--;
            node_index = find_node(node_index, len);
            density_b = density_bound(&edges, level);
            density = get_density(&edges, node_index, len);
        }
        else
        {
            // if you reach the root, double the list
            double_list();

            // search from the beginning because list was doubled
            return find_elem_pointer(&edges, 0, elem);
        }
    }
    redistribute(node_index, len);

    return find_elem_pointer(&edges, node_index, elem);
}

void PCSR::double_list()
{
    edges.N *= 2;
    edges.logN = (1 << bsr_word(bsr_word(edges.N) + 1));
    edges.H = bsr_word(edges.N / edges.logN);

    edges.items.resize(edges.N);
    for (int i = edges.N / 2; i < edges.N; i++)
    {
        edge_t new_edge(0, 0);
        edges.items[i] = new_edge;
    }

    redistribute(0, edges.N);
}

int PCSR::slide_right(int index)
{
    int rval = 0;
    edge_t el = edges.items[index];
    edges.items[index].dest = 0;
    edges.items[index].value = 0;
    index++;
    while (index < edges.N && !is_null(edges.items[index]))
    {
        edge_t temp = edges.items[index];
        edges.items[index] = el;
        if (!is_null(el) && is_sentinel(el))
        {
            // fixing pointer of node that goes to this sentinel
            uint32_t node_index = el.value;
            if (node_index == UINT32_MAX)
            {
                node_index = 0;
            }
            fix_sentinel(node_index, index);
        }
        el = temp;
        index++;
    }
    if (!is_null(el) && is_sentinel(el))
    {
        // fixing pointer of node that goes to this sentinel
        uint32_t node_index = el.value;
        if (node_index == UINT32_MAX)
        {
            node_index = 0;
        }
        fix_sentinel(node_index, index);
    }
    // TODO There might be an issue with this going of the end sometimes
    if (index == edges.N)
    {
        index--;
        slide_left(index);
        rval = -1;
        printf("slide off the end on the right, should be rare\n");
    }
    edges.items[index] = el;
    return rval;
}

void PCSR::slide_left(int index)
{
    edge_t el = edges.items[index];
    edges.items[index].dest = 0;
    edges.items[index].value = 0;

    index--;
    while (index >= 0 && !is_null(edges.items[index]))
    {
        edge_t temp = edges.items[index];
        edges.items[index] = el;
        if (!is_null(el) && is_sentinel(el))
        {
            // fixing pointer of node that goes to this sentinel
            uint32_t node_index = el.value;
            if (node_index == UINT32_MAX)
            {
                node_index = 0;
            }

            fix_sentinel(node_index, index);
        }
        el = temp;
        index--;
    }

    if (index == -1)
    {
        double_list();

        slide_right(0);
        index = 0;
    }
    if (!is_null(el) && is_sentinel(el))
    {
        // fixing pointer of node that goes to this sentinel
        uint32_t node_index = el.value;
        if (node_index == UINT32_MAX)
        {
            node_index = 0;
        }
        fix_sentinel(node_index, index);
    }

    edges.items[index] = el;
}

void PCSR::fix_sentinel(int32_t node_index, int in)
{
    nodes[node_index].beginning = in;
    if (node_index > 0)
    {
        nodes[node_index - 1].end = in;
    }
    if (node_index == nodes.size() - 1)
    {
        nodes[node_index].end = edges.N - 1;
    }
}

void PCSR::redistribute(int index, int len)
{
    // printf("REDISTRIBUTE: \n");
    // print_array();
    // std::vector<edge_t> space(len); //
    edge_t new_edge;
    vector<edge_t> space(len, new_edge);

    int j = 0;

    // move all items in ofm in the range into
    // a temp array
    for (int i = index; i < index + len; i++)
    {
        space[j] = edges.items[i];
        // counting non-null edges
        j += (!is_null(edges.items[i]));
        // setting section to null
        edges.items[i].value = 0;
        edges.items[i].dest = 0;
    }

    // evenly redistribute for a uniform density
    double index_d = index;
    double step = ((double)len) / j;
    for (int i = 0; i < j; i++)
    {
        int in = index_d;

        edges.items[in] = space[i];
        if (is_sentinel(space[i]))
        {
            // fixing pointer of node that goes to this sentinel
            uint32_t node_index = space[i].value;
            if (node_index == UINT32_MAX)
            {
                node_index = 0;
            }
            fix_sentinel(node_index, in);
        }
        index_d += step;
    }
    //   free(space);
}

void PCSR::add_edge(uint32_t src, uint32_t dest, uint32_t value)
{
    // cout << "Adding edge (" << src << "," << dest << ")\n";
    if (value != 0)
    {
        node_t node = nodes[src];
        nodes[src].num_neighbors++;
        nodes[dest].in_degree++;

        edge_t e;
        e.dest = dest;
        e.value = value;

        uint32_t loc_to_add =
            binary_search(&edges, &e, node.beginning + 1, node.end);
        insert(loc_to_add, e, src);
        ++edge_count;
    }
}

void PCSR::add_edge_update(uint32_t src, uint32_t dest, uint32_t value)
{
    if (value != 0)
    {
        node_t node = nodes[src];

        edge_t e;
        e.dest = dest;
        e.value = value;

        uint32_t loc_to_add =
            binary_search(&edges, &e, node.beginning + 1, node.end);
        if (edges.items[loc_to_add].dest == dest)
        {
            edges.items[loc_to_add].value = value;
            return;
        }
        nodes[src].num_neighbors++;
        nodes[dest].in_degree++;
        insert(loc_to_add, e, src);
        ++edge_count;
    }
}

void PCSR::delete_edge(uint32_t src, uint32_t dest)
{
    // cout << "Deleting edge (" << src << "," << dest << ")\n";
    edge_t e;
    e.value = 0;
    e.dest = dest;
    uint32_t loc = binary_search(&edges, &e, nodes[src].beginning + 1, nodes[src].end);

    if (!is_null(edges.items[loc]) && edges.items[loc].dest == dest)
    {
        edges.items[loc].value = 0;
        nodes[src].num_neighbors -= 1;
        nodes[dest].in_degree -= 1;
        --edge_count;
    }
}

uint64_t PCSR::get_n()
{
    return nodes.size();
}

vector<tuple<uint32_t, uint32_t, uint32_t>> PCSR::get_edges()
{
    uint64_t n = get_n();
    vector<tuple<uint32_t, uint32_t, uint32_t>> output;

    output.resize(edge_count);
    int iter = 0;
    for (int i = 0; i < n; i++)
    {
        uint32_t start = nodes[i].beginning;
        uint32_t end = nodes[i].end;
        for (int j = start + 1; j < end; j++)
        {
            if (!is_null(edges.items[j]))
            {
                output[iter] =
                    make_tuple(i, edges.items[j].dest, edges.items[j].value);
                iter += 1;
            }
        }
    }
    return output;
}

void PCSR::label_edges()
{
    uint32_t column_indices_size = edges.items.size();
    uint32_t counter = 1;
    for (int i = 0; i < column_indices_size; ++i)
    {
        if (!is_sentinel(edges.items[i]) && !is_null(edges.items[i]))
        {
            edges.items[i].value = counter;
            ++counter;
        }
    }
}

void PCSR::edge_update_list(std::vector<std::tuple<uint32_t, uint32_t>> edge_list, bool is_delete = false, bool is_reverse_edge = false)
{
    bool is_reverse_edge_local = is_reverse_edge;
    bool is_delete_local = is_delete;

    for (auto &edge : edge_list)
    {
        uint32_t src = (is_reverse_edge_local == true) ? std::get<1>(edge) : std::get<0>(edge);
        uint32_t dst = (is_reverse_edge_local == true) ? std::get<0>(edge) : std::get<1>(edge);

        if (is_delete_local){
            in_degrees[dst] -= 1;
            out_degrees[src] -= 1;
            delete_edge(src, dst);
        }else{
            in_degrees[dst] += 1;
            out_degrees[src] += 1;
            add_edge(src, dst, 1);
        }
    }
}

void PCSR::build_reverse_csr()
{
    uint64_t n = get_n();
    // computing the bwd row offsets
    row_offset_pinned[0] = in_degrees[0];
    for(int i=1; i<in_degrees.size(); ++i){
        row_offset_pinned[i] = row_offset_pinned[i-1] + in_degrees[i];
    }
    row_offset_pinned[in_degrees.size()] = edge_count;
    
    for (int i = 0; i < n; i++)
    {
        uint32_t start = nodes[i].beginning;
        uint32_t end = nodes[i].end;
        for (int j = start + 1; j < end; j++)
        {
            if (!is_sentinel(edges.items[j]) && !is_null(edges.items[j]))
            {
                row_offset_pinned[edges.items[j].dest] -= 1;

                int col_index = row_offset_pinned[edges.items[j].dest];
                column_indices_pinned[col_index] = i;
                eids_pinned[col_index] = edges.items[j].value;
            }
        }
    }

    move_pinned_to_gpu();
}

void PCSR::build_csr()
{
    // computing the bwd row offsets
    uint64_t n = get_n();
    row_offset_pinned[0] = out_degrees[0];
    for(int i=1; i<out_degrees.size(); ++i){
        row_offset_pinned[i] = row_offset_pinned[i-1] + out_degrees[i];
    }
    row_offset_pinned[out_degrees.size()] = edge_count;
    
    for (int i = 0; i < n; i++)
    {
        uint32_t start = nodes[i].beginning;
        uint32_t end = nodes[i].end;
        for (int j = start + 1; j < end; j++)
        {
            if (!is_sentinel(edges.items[j]) && !is_null(edges.items[j]))
            {
                row_offset_pinned[i] -= 1;
                column_indices_pinned[row_offset_pinned[i]] = edges.items[j].dest;
                eids_pinned[row_offset_pinned[i]] = edges.items[j].value;
            }
        }
    }

    move_pinned_to_gpu();
}

void PCSR::move_pinned_to_gpu(){
    cErr(cudaMemcpy(row_offset_device, row_offset_pinned, sizeof(uint32_t) * (get_n() + 1), cudaMemcpyHostToDevice));
    cErr(cudaMemcpy(column_indices_device, column_indices_pinned, sizeof(uint32_t) * edge_count, cudaMemcpyHostToDevice));
    cErr(cudaMemcpy(eids_device, eids_pinned, sizeof(uint32_t) * edge_count, cudaMemcpyHostToDevice));
}

std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t> PCSR::get_csr_ptrs(){
    std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t> t;
    std::get<0>(t) = (std::uintptr_t)row_offset_device;
    std::get<1>(t) = (std::uintptr_t)column_indices_device;
    std::get<2>(t) = (std::uintptr_t)eids_device;
    return t;
}

std::vector<std::vector<uint32_t>> read_gpu_csr(PCSR &pcsr){
    std::vector<uint32_t> row_offset(pcsr.get_n() + 1);
    std::vector<uint32_t> column_indices(pcsr.edge_count);
    std::vector<uint32_t> eids(pcsr.edge_count);

    cErr(cudaMemcpy(row_offset.data(), pcsr.row_offset_device, sizeof(uint32_t) * (pcsr.get_n() + 1), cudaMemcpyDeviceToHost));
    cErr(cudaMemcpy(column_indices.data(), pcsr.column_indices_device, sizeof(uint32_t) * pcsr.edge_count, cudaMemcpyDeviceToHost));
    cErr(cudaMemcpy(eids.data(), pcsr.eids_device, sizeof(uint32_t) * pcsr.edge_count, cudaMemcpyDeviceToHost));

    std::vector<std::vector<uint32_t>> res;
    res.push_back(row_offset);
    res.push_back(column_indices);
    res.push_back(eids);
    return res;
}

////////////////////////////////////////////////////////////////////////////

// PCSR Python Module

PYBIND11_MODULE(pcsr, m)
{
    m.doc() = "PCSR Dynamic Graph Representation";
    m.def("read_gpu_csr", &read_gpu_csr, "Read CSR arrays from the GPU");

    py::class_<PCSR>(m, "PCSR")
        .def(py::init<int, int>(), py::arg("init_n"), py::arg("max_edge_count"))
        .def_readwrite("in_degrees", &PCSR::in_degrees)
        .def_readwrite("out_degrees", &PCSR::out_degrees)
        .def_readwrite("edge_count", &PCSR::edge_count)
        .def("get_n", &PCSR::get_n)
        .def("edge_update_list", &PCSR::edge_update_list, py::arg("edge_list"), py::arg("is_delete") = false, py::arg("is_reverse_edge") = false)
        .def("label_edges", &PCSR::label_edges, "Creates edge labels for the current GPMA")
        .def("get_edges", &PCSR::get_edges)
        .def("build_csr", &PCSR::build_csr)
        .def("build_reverse_csr", &PCSR::build_reverse_csr)
        .def("get_csr_ptrs", &PCSR::get_csr_ptrs)
        .def("__copy__", [](const PCSR &self)
             { return PCSR(self); })
        .def(
            "__deepcopy__", [](const PCSR &self, py::dict)
            { return PCSR(self); },
            "memo"_a);
}

// /usr/local/cuda-11.7/bin/nvcc $(python3 -m pybind11 --includes) -shared -rdc=true --compiler-options '-fPIC'  -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -o pcsr.so pcsr.cu
