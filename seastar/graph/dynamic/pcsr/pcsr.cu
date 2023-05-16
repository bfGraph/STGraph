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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std;

typedef thrust::device_vector<int> DEV_VEC;
#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())

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
    edge_list_t edges;
    uint32_t edge_count;

    DEV_VEC row_offset_device;
    DEV_VEC column_indices_device;
    DEV_VEC eids_device;

    // member functions
    PCSR(uint32_t init_n);
    void init_graph(std::string graph_file_path);

    void add_node();
    void add_edge(uint32_t src, uint32_t dest, uint32_t value);
    void add_edge_update(uint32_t src, uint32_t dest, uint32_t value);
    void edge_update_list(std::vector<std::tuple<uint32_t, uint32_t>> edge_list, bool is_delete, bool is_reverse_edge);
    void delete_edge(uint32_t src, uint32_t dest);
    uint64_t get_n();
    // vector<tuple<uint32_t, uint32_t, uint32_t>> get_edges();
    void print_graph();
    void print_array();
    std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t> get_csr_ptrs(std::vector<int> eids);
    // void label_edges();
    // uint32_t find_edge_id(uint32_t src, uint32_t dest);
    // std::tuple<uint32_t, uint32_t> get_graph_attr();

    uint32_t insert(uint32_t index, edge_t elem, uint32_t src);
    void double_list();
    int slide_right(int index);
    void slide_left(int index);
    void fix_sentinel(int32_t node_index, int in);
    void redistribute(int index, int len);
};

////////////////////////////////////////////////////////////////////////////

// removed default argument of 0
PCSR::PCSR(uint32_t init_n)
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
    }
}

void PCSR::init_graph(std::string graph_file_path)
{

    FILE *fp;
    fp = fopen(graph_file_path.c_str(), "r");
    if (!fp)
    {
        printf("Open file failed.\n");
        exit(-1);
    }

    // loading the number of nodes and edges present in
    // the graph into the variables node_size and edge_size respectively
    int node_size, edge_size;
    fscanf(fp, "%d %d", &node_size, &edge_size);

    edges.N = 2 << bsr_word(node_size);
    edges.logN = (1 << bsr_word(bsr_word(edges.N) + 1));
    edges.H = bsr_word(edges.N / edges.logN);
    edge_count = 0;

    edges.items.resize(edges.N);
    for (int i = 0; i < edges.N; i++)
    {
        edge_t new_edge(0, 0);
        edges.items[i] = new_edge;
    }

    for (int i = 0; i < node_size; i++)
    {
        add_node();
    }

    for (int i = 0; i < edge_size; i++)
    {
        int source, dest, edge_value;
        fscanf(fp, "%d %d %d", &source, &dest, &edge_value);
        add_edge(source, dest, edge_value);
    }
}

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

// vector<tuple<uint32_t, uint32_t, uint32_t>> PCSR::get_edges()
// {
//     uint64_t n = get_n();
//     vector<tuple<uint32_t, uint32_t, uint32_t>> output;

//     output.resize(edge_count);
//     int iter = 0;
//     for (int i = 0; i < n; i++)
//     {
//         uint32_t start = nodes[i].beginning;
//         uint32_t end = nodes[i].end;
//         for (int j = start + 1; j < end; j++)
//         {
//             if (!is_null(edges.items[j]))
//             {
//                 output[iter] =
//                     make_tuple(i, edges.items[j].dest, edges.items[j].value);
//                 iter += 1;
//             }
//         }
//     }
//     return output;
// }

void PCSR::print_graph()
{
    int num_vertices = nodes.size();

    // printing the graph matrix column indices
    for (int i = 0; i < num_vertices; ++i)
        printf("   %d", i);

    printf("\n");

    for (int i = 0; i < num_vertices; i++)
    {
        // +1 to avoid sentinel

        // printing the graph matrix row indices
        printf("%d ", i);
        int matrix_index = 0;

        for (uint32_t j = nodes[i].beginning + 1; j < nodes[i].end; j++)
        {
            if (!is_null(edges.items[j]))
            {
                while (matrix_index < edges.items[j].dest)
                {
                    printf("    ");
                    matrix_index++;
                }
                // printf("%03d ", edges.items[j].value);
                printf(" %d  ", edges.items[j].value);
                matrix_index++;
            }
        }
        for (uint32_t j = matrix_index; j < num_vertices; j++)
        {
            printf("    ");
        }
        printf("\n");
    }
}

void PCSR::print_array()
{
    for (int i = 0; i < edges.N; i++)
    {
        if (is_null(edges.items[i]))
        {
            printf("%d-x ", i);
        }
        else if (is_sentinel(edges.items[i]))
        {
            uint32_t value = edges.items[i].value;
            if (value == UINT32_MAX)
            {
                value = 0;
            }
            printf("\n%d-s(%u):(%d, %d) ", i, value, nodes[value].beginning,
                   nodes[value].end);
        }
        else
        {
            printf("%d-(%d, %u) ", i, edges.items[i].dest, edges.items[i].value);
        }
    }
    printf("\n\n");
}

std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t> PCSR::get_csr_ptrs(std::vector<int> eids)
{

    // if (edge_count != eids.size())
    // {
    //     cout << "🍎🍎🍎 The eids(" << eids.size() << ") and edge_count(" << edge_count << ") [col_indices.size() = " << edges.items.size() << "] DONT match\n"
    //          << flush;
    // }

    // we are building a compressed CSR arrays without
    // the -1 values which indicates an empty edge
    thrust::host_vector<int> row_offset;
    thrust::host_vector<int> column_indices;

    int row_offset_size = nodes.size() + 1;
    int column_indices_size = edges.items.size();

    // first element of row offset is always zero
    row_offset.resize(row_offset_size);
    column_indices.resize(edge_count);
    row_offset[0] = 0;

    for (int i = 0; i < row_offset_size - 1; ++i)
    {
        row_offset[i + 1] = nodes[i].num_neighbors + row_offset[i];
    }

    int iter = 0;
    for (int i = 0; i < column_indices_size; ++i)
    {
        if (!is_sentinel(edges.items[i]) && !is_null(edges.items[i]))
        {
            column_indices[iter] = edges.items[i].dest;
            iter += 1;
        }
    }

    row_offset_device = row_offset;
    column_indices_device = column_indices;
    eids_device.resize(edge_count);

    if (eids.size() == 0)
        thrust::sequence(eids_device.begin(), eids_device.end());
    else
        thrust::copy(eids.begin(), eids.end(), eids_device.begin());

    std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t> t;
    std::get<0>(t) = (std::uintptr_t)RAW_PTR(row_offset_device);
    std::get<1>(t) = (std::uintptr_t)RAW_PTR(column_indices_device);
    std::get<2>(t) = (std::uintptr_t)RAW_PTR(eids_device);
    return t;
}

// void PCSR::label_edges()
// {

//     uint32_t column_indices_size = edges.items.size();
//     uint32_t counter = 1;
//     for (int i = 0; i < column_indices_size; ++i)
//     {
//         if (!is_sentinel(edges.items[i]) && !is_null(edges.items[i]))
//         {
//             edges.items[i].value = counter;
//             ++counter;
//         }
//     }
// }

// uint32_t PCSR::find_edge_id(uint32_t src, uint32_t dest)
// {
//     edge_t e;
//     e.dest = dest;
//     // setting a random number
//     e.value = 0;

//     uint32_t loc = binary_search(&edges, &e, nodes[src].beginning + 1, nodes[src].end);
//     return edges.items[loc].value;
// }

// std::tuple<uint32_t, uint32_t> PCSR::get_graph_attr()
// {
//     std::tuple<uint32_t, uint32_t> t;
//     std::get<0>(t) = nodes.size();
//     std::get<1>(t) = edge_count;
//     return t;
// }

// void copy_label_edges(PCSR &pcsr, PCSR &ref_pcsr)
// {
//     uint64_t n = pcsr.get_n();

//     for (int i = 0; i < n; i++)
//     {
//         uint32_t start = pcsr.nodes[i].beginning;
//         uint32_t end = pcsr.nodes[i].end;
//         for (int j = start + 1; j < end; j++)
//         {
//             if (!is_null(pcsr.edges.items[j]))
//             {
//                 // Searching for the edge_id of reverse edge from reference_pcsr
//                 pcsr.edges.items[j].value = ref_pcsr.find_edge_id(pcsr.edges.items[j].dest, i);
//             }
//         }
//     }
// }

void PCSR::edge_update_list(std::vector<std::tuple<uint32_t, uint32_t>> edge_list, bool is_delete = false, bool is_reverse_edge = false)
{
    // cout << "📦📦📦 Edge Update list" << endl
    //      << flush;
    for (auto &edge : edge_list)
    {
        uint32_t src = (is_reverse_edge == true) ? std::get<1>(edge) : std::get<0>(edge);
        uint32_t dst = (is_reverse_edge == true) ? std::get<0>(edge) : std::get<1>(edge);

        if (is_delete)
            delete_edge(src, dst);
        else
            add_edge(src, dst, 1);
    }
}

void build_reverse_pcsr(PCSR &pcsr, PCSR &ref_pcsr)
{
    // cout << "💄💄💄 Building Reverse PCSR" << endl
    //      << flush;
    uint64_t n = ref_pcsr.get_n();
    for (int i = 0; i < n; i++)
    {
        uint32_t start = ref_pcsr.nodes[i].beginning;
        uint32_t end = ref_pcsr.nodes[i].end;
        for (int j = start + 1; j < end; j++)
        {
            if (!is_sentinel(ref_pcsr.edges.items[j]) && !is_null(ref_pcsr.edges.items[j]))
            {
                pcsr.add_edge(ref_pcsr.edges.items[j].dest, i, ref_pcsr.edges.items[j].value);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////

// PCSR Python Module

PYBIND11_MODULE(pcsr, m)
{
    m.doc() = "PCSR Dynamic Graph Representation";
    // m.def("copy_label_edges", &copy_label_edges, "Label edges of a PCSR based on another PCSR");
    m.def("build_reverse_pcsr", &build_reverse_pcsr, "Builds the reverse PCSR based on another PCSR");

    py::class_<node_t>(m, "Node")
        .def(py::init<int, int, int>(), py::arg("beg") = 0, py::arg("_end") = 0, py::arg("num_neigh") = 0)
        .def_readwrite("beginning", &_node::beginning)
        .def_readwrite("end", &_node::end)
        .def_readwrite("num_neighbors", &_node::num_neighbors)
        .def_readwrite("in_degree", &_node::in_degree);

    py::class_<edge_t>(m, "Edge")
        .def(py::init<int, int>(), py::arg("_dest") = 0, py::arg("_value") = 0)
        .def_readwrite("dest", &edge_t::dest)
        .def_readwrite("value", &edge_t::value);

    py::class_<edge_list_t>(m, "EdgeList")
        .def(py::init<>())
        .def_readwrite("N", &edge_list_t::N)
        .def_readwrite("H", &edge_list_t::H)
        .def_readwrite("logN", &edge_list_t::logN)
        .def_readwrite("items", &edge_list_t::items);

    py::class_<pair_int>(m, "PairInt")
        .def(py::init<int, int>())
        .def_readwrite("x", &pair_int::x)
        .def_readwrite("y", &pair_int::y);

    py::class_<pair_double>(m, "PairDouble")
        .def(py::init<double, double>())
        .def_readwrite("x", &pair_double::x)
        .def_readwrite("y", &pair_double::y);

    py::class_<PCSR>(m, "PCSR")
        .def(py::init<int>(), py::arg("init_n") = 0)
        .def("init_graph", &PCSR::init_graph)
        .def("add_node", &PCSR::add_node)
        .def("insert", &PCSR::insert)
        .def("double_list", &PCSR::double_list)
        .def("slide_right", &PCSR::slide_right)
        .def("slide_left", &PCSR::slide_left)
        .def("fix_sentinel", &PCSR::fix_sentinel)
        .def("redistribute", &PCSR::redistribute)
        .def("print_graph", &PCSR::print_graph)
        .def("add_edge", &PCSR::add_edge)
        .def("add_edge_update", &PCSR::add_edge_update)
        .def("edge_update_list", &PCSR::edge_update_list, py::arg("edge_list"), py::arg("is_delete") = false, py::arg("is_reverse_edge") = false)
        // .def("label_edges", &PCSR::label_edges, "Creates edge labels for the current GPMA")
        .def("delete_edge", &PCSR::delete_edge)
        .def("get_n", &PCSR::get_n)
        // .def("get_edges", &PCSR::get_edges)
        .def("print_array", &PCSR::print_array)
        .def("get_csr_ptrs", &PCSR::get_csr_ptrs, py::arg("eids"))
        // .def("get_graph_attr", &PCSR::get_graph_attr)
        // .def("find_edge_id", &PCSR::find_edge_id)
        .def_readwrite("nodes", &PCSR::nodes)
        .def_readwrite("edges", &PCSR::edges)
        .def_readwrite("edge_count", &PCSR::edge_count)
        .def("__copy__", [](const PCSR &self)
             { return PCSR(self); })
        .def(
            "__deepcopy__", [](const PCSR &self, py::dict)
            { return PCSR(self); },
            "memo"_a);
}

// /usr/local/cuda-11.7/bin/nvcc $(python3 -m pybind11 --includes) -shared -rdc=true --compiler-options '-fPIC'  -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -o pcsr.so pcsr.cu