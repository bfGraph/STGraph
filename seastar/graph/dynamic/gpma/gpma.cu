#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>
// #include <memory>
#include "stdio.h"

namespace py = pybind11;
using namespace pybind11::literals;

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

typedef unsigned long long KEY_TYPE;
typedef double VALUE_TYPE;
typedef unsigned int SIZE_TYPE;

typedef thrust::device_vector<KEY_TYPE> DEV_VEC_KEY;
typedef thrust::device_vector<VALUE_TYPE> DEV_VEC_VALUE;
typedef thrust::device_vector<SIZE_TYPE> DEV_VEC_SIZE;

typedef KEY_TYPE *KEY_PTR;
typedef VALUE_TYPE *VALUE_PTR;

#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())

const KEY_TYPE KEY_NONE = 0xFFFFFFFFFFFFFFFF;
const KEY_TYPE KEY_MAX = 0xFFFFFFFFFFFFFFFE;
const SIZE_TYPE SIZE_NONE = 0xFFFFFFFF;
const VALUE_TYPE VALUE_NONE = 0;
const KEY_TYPE COL_IDX_NONE = 0xFFFFFFFF;

const SIZE_TYPE MAX_BLOCKS_NUM = 96 * 8;
#define CALC_BLOCKS_NUM(ITEMS_PER_BLOCK, CALC_SIZE) min(MAX_BLOCKS_NUM, (CALC_SIZE - 1) / ITEMS_PER_BLOCK + 1)

class GPMA
{
public:
    // NOTE: Assuming that keys holds the column_indices
    // and values is the column indices (subject to change)
    DEV_VEC_KEY keys;
    DEV_VEC_VALUE values;

    SIZE_TYPE segment_length;
    SIZE_TYPE tree_height;

    double density_lower_thres_leaf = 0.08;
    double density_lower_thres_root = 0.42;
    double density_upper_thres_root = 0.84;
    double density_upper_thres_leaf = 0.92;

    thrust::host_vector<SIZE_TYPE> lower_element;
    thrust::host_vector<SIZE_TYPE> upper_element;

    // addition for csr
    SIZE_TYPE row_num; // number of nodes
    DEV_VEC_SIZE row_offset;
    SIZE_TYPE *row_offset_ptr;

    int edge_count = 0;
    std::vector<unsigned int> in_degree;
    std::vector<unsigned int> out_degree;

    GPMA();

    inline int get_size()
    {
        return keys.size();
    }
};

GPMA::GPMA()
{
}

__forceinline__ __host__ __device__
    SIZE_TYPE
    fls(SIZE_TYPE x)
{
    SIZE_TYPE r = 32;
    if (!x)
        return 0;
    if (!(x & 0xffff0000u))
        x <<= 16, r -= 16;
    if (!(x & 0xff000000u))
        x <<= 8, r -= 8;
    if (!(x & 0xf0000000u))
        x <<= 4, r -= 4;
    if (!(x & 0xc0000000u))
        x <<= 2, r -= 2;
    if (!(x & 0x80000000u))
        x <<= 1, r -= 1;
    return r;
}

template <typename T>
__global__ void memcpy_kernel(T *dest, const T *src, SIZE_TYPE size)
{
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset)
    {
        dest[i] = src[i];
    }
}

template <typename T>
__global__ void memset_kernel(T *data, T value, SIZE_TYPE size)
{
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset)
    {
        data[i] = value;
    }
}

__host__ void recalculate_density(GPMA &gpma)
{
    gpma.lower_element.resize(gpma.tree_height + 1);
    gpma.upper_element.resize(gpma.tree_height + 1);
    cErr(cudaDeviceSynchronize());

    SIZE_TYPE level_length = gpma.segment_length;
    for (SIZE_TYPE i = 0; i <= gpma.tree_height; i++)
    {
        double density_lower = gpma.density_lower_thres_root + (gpma.density_lower_thres_leaf - gpma.density_lower_thres_root) * (gpma.tree_height - i) / gpma.tree_height;
        double density_upper = gpma.density_upper_thres_root + (gpma.density_upper_thres_leaf - gpma.density_upper_thres_root) * (gpma.tree_height - i) / gpma.tree_height;

        gpma.lower_element[i] = (SIZE_TYPE)ceil(density_lower * level_length);
        gpma.upper_element[i] = (SIZE_TYPE)floor(density_upper * level_length);

        // special trim for wrong threshold introduced by float-integer conversion
        if (0 < i)
        {
            gpma.lower_element[i] = max(gpma.lower_element[i], 2 * gpma.lower_element[i - 1]);
            gpma.upper_element[i] = min(gpma.upper_element[i], 2 * gpma.upper_element[i - 1]);
        }
        level_length <<= 1;
    }
}

__device__ void cub_sort_key_value(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size, KEY_TYPE *tmp_keys,
                                   VALUE_TYPE *tmp_values)
{

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cErr(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, tmp_keys, values, tmp_values, size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, tmp_keys, values, tmp_values, size));
    cErr(cudaDeviceSynchronize());

    SIZE_TYPE THREADS_NUM = 128;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
    memcpy_kernel<KEY_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(keys, tmp_keys, size);
    memcpy_kernel<VALUE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(values, tmp_values, size);
    cErr(cudaDeviceSynchronize());

    cErr(cudaFree(d_temp_storage));
}

__device__ SIZE_TYPE handle_del_mod(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE seg_length, KEY_TYPE key,
                                    VALUE_TYPE value, SIZE_TYPE leaf)
{

    if (VALUE_NONE == value)
        leaf = SIZE_NONE;
    for (SIZE_TYPE i = 0; i < seg_length; i++)
    {
        if (keys[i] == key)
        {
            values[i] = value;
            leaf = SIZE_NONE;
            break;
        }
    }
    return leaf;
}

__global__ void locate_leaf_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE tree_size, SIZE_TYPE seg_length,
                                   SIZE_TYPE tree_height, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size,
                                   SIZE_TYPE *leaf, bool return_leaf_loc = true)
{

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < update_size; i += block_offset)
    {
        KEY_TYPE key = update_keys[i];
        VALUE_TYPE value = update_values[i];

        SIZE_TYPE prefix = 0;
        SIZE_TYPE current_bit = seg_length << tree_height >> 1;

        while (seg_length <= current_bit)
        {
            if (keys[prefix | current_bit] <= key)
                prefix |= current_bit;
            current_bit >>= 1;
        }

        prefix = handle_del_mod(keys + prefix, values + prefix, seg_length, key, value, prefix);

        if (return_leaf_loc)
            leaf[i] = prefix;
    }
}

__host__ void locate_leaf_batch(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE tree_size, SIZE_TYPE seg_length,
                                SIZE_TYPE tree_height, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size,
                                SIZE_TYPE *leaf, bool return_leaf_loc = true)
{

    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size);

    locate_leaf_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(keys, values, tree_size, seg_length, tree_height, update_keys,
                                                    update_values, update_size, leaf, return_leaf_loc);
    cErr(cudaDeviceSynchronize());
}

template <SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__device__ void block_compact_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE &compacted_size)
{
    typedef cub::BlockScan<SIZE_TYPE, THREAD_PER_BLOCK> BlockScan;
    SIZE_TYPE thread_id = threadIdx.x;

    KEY_TYPE *block_keys = keys;
    VALUE_TYPE *block_values = values;

    KEY_TYPE thread_keys[ITEM_PER_THREAD];
    VALUE_TYPE thread_values[ITEM_PER_THREAD];

    SIZE_TYPE thread_offset = thread_id * ITEM_PER_THREAD;
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++)
    {
        thread_keys[i] = block_keys[thread_offset + i];
        thread_values[i] = block_values[thread_offset + i];
        block_keys[thread_offset + i] = KEY_NONE;
    }

    __shared__ typename BlockScan::TempStorage temp_storage;
    SIZE_TYPE thread_data[ITEM_PER_THREAD];
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++)
    {
        thread_data[i] = (thread_keys[i] == KEY_NONE || thread_values[i] == VALUE_NONE) ? 0 : 1;
    }
    __syncthreads();

    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
    __syncthreads();

    __shared__ SIZE_TYPE exscan[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++)
    {
        exscan[i + thread_offset] = thread_data[i];
    }
    __syncthreads();

    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++)
    {
        if (thread_id == THREAD_PER_BLOCK - 1 && i == ITEM_PER_THREAD - 1)
            continue;
        if (exscan[thread_offset + i] != exscan[thread_offset + i + 1])
        {
            SIZE_TYPE loc = exscan[thread_offset + i];
            block_keys[loc] = thread_keys[i];
            block_values[loc] = thread_values[i];
        }
    }

    // special logic for the last element
    if (thread_id == THREAD_PER_BLOCK - 1)
    {
        SIZE_TYPE loc = exscan[THREAD_PER_BLOCK * ITEM_PER_THREAD - 1];
        if (thread_keys[ITEM_PER_THREAD - 1] == KEY_NONE || thread_values[ITEM_PER_THREAD - 1] == VALUE_NONE)
        {
            compacted_size = loc;
        }
        else
        {
            compacted_size = loc + 1;
            block_keys[loc] = thread_keys[ITEM_PER_THREAD - 1];
            block_values[loc] = thread_values[ITEM_PER_THREAD - 1];
        }
    }
}

template <typename FIRST_TYPE, typename SECOND_TYPE>
__device__ void block_pair_copy_kernel(FIRST_TYPE *dest_first, SECOND_TYPE *dest_second, FIRST_TYPE *src_first,
                                       SECOND_TYPE *src_second, SIZE_TYPE size)
{
    for (SIZE_TYPE i = threadIdx.x; i < size; i += blockDim.x)
    {
        dest_first[i] = src_first[i];
        dest_second[i] = src_second[i];
    }
}

template <SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__device__ void block_redispatch_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE rebalance_width, SIZE_TYPE seg_length,
                                        SIZE_TYPE merge_size, SIZE_TYPE *row_offset, SIZE_TYPE update_node)
{

    // step1: load KV in shared memory
    __shared__ KEY_TYPE block_keys[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    __shared__ VALUE_TYPE block_values[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    block_pair_copy_kernel<KEY_TYPE, VALUE_TYPE>(block_keys, block_values, keys, values, rebalance_width);
    __syncthreads();

    // step2: sort by key with value on shared memory
    typedef cub::BlockLoad<KEY_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockKeyLoadT;
    typedef cub::BlockLoad<VALUE_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockValueLoadT;
    typedef cub::BlockStore<KEY_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockKeyStoreT;
    typedef cub::BlockStore<VALUE_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockValueStoreT;
    typedef cub::BlockRadixSort<KEY_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, VALUE_TYPE> BlockRadixSortT;

    __shared__ union
    {
        typename BlockKeyLoadT::TempStorage key_load;
        typename BlockValueLoadT::TempStorage value_load;
        typename BlockKeyStoreT::TempStorage key_store;
        typename BlockValueStoreT::TempStorage value_store;
        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    KEY_TYPE thread_keys[ITEM_PER_THREAD];
    VALUE_TYPE thread_values[ITEM_PER_THREAD];
    BlockKeyLoadT(temp_storage.key_load).Load(block_keys, thread_keys);
    BlockValueLoadT(temp_storage.value_load).Load(block_values, thread_values);
    __syncthreads();

    BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_values);
    __syncthreads();

    BlockKeyStoreT(temp_storage.key_store).Store(block_keys, thread_keys);
    BlockValueStoreT(temp_storage.value_store).Store(block_values, thread_values);
    __syncthreads();

    // step3: evenly re-dispatch KVs to leaf segments
    KEY_TYPE frac = rebalance_width / seg_length;
    KEY_TYPE deno = merge_size;
    for (SIZE_TYPE i = threadIdx.x; i < merge_size; i += blockDim.x)
    {
        keys[i] = KEY_NONE;
    }
    __syncthreads();

    for (SIZE_TYPE i = threadIdx.x; i < merge_size; i += blockDim.x)
    {
        SIZE_TYPE seg_idx = (SIZE_TYPE)(frac * i / deno);
        SIZE_TYPE seg_lane = (SIZE_TYPE)(frac * i % deno / frac);
        SIZE_TYPE proj_location = seg_idx * seg_length + seg_lane;

        KEY_TYPE cur_key = block_keys[i];
        VALUE_TYPE cur_value = block_values[i];
        keys[proj_location] = cur_key;
        values[proj_location] = cur_value;

        // addition for csr
        if ((cur_key & COL_IDX_NONE) == COL_IDX_NONE)
        {
            SIZE_TYPE cur_row = (SIZE_TYPE)(cur_key >> 32);
            row_offset[cur_row + 1] = proj_location + update_node;
        }
    }
}

template <SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__global__ void block_rebalancing_kernel(SIZE_TYPE seg_length, SIZE_TYPE level, KEY_TYPE *keys, VALUE_TYPE *values,
                                         SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE *unique_update_nodes,
                                         SIZE_TYPE *update_offset, SIZE_TYPE lower_bound, SIZE_TYPE upper_bound, SIZE_TYPE *row_offset)
{

    SIZE_TYPE update_id = blockIdx.x;
    SIZE_TYPE update_node = unique_update_nodes[update_id];
    KEY_TYPE *key = keys + update_node;
    VALUE_TYPE *value = values + update_node;
    SIZE_TYPE rebalance_width = seg_length << level;

    // compact
    __shared__ SIZE_TYPE compacted_size;
    block_compact_kernel<THREAD_PER_BLOCK, ITEM_PER_THREAD>(key, value, compacted_size);
    __syncthreads();

    // judge whether fit the density threshold
    SIZE_TYPE interval_a = update_offset[update_id];
    SIZE_TYPE interval_b = update_offset[update_id + 1];
    SIZE_TYPE interval_size = interval_b - interval_a;
    SIZE_TYPE merge_size = compacted_size + interval_size;
    __syncthreads();

    if (lower_bound <= merge_size && merge_size <= upper_bound)
    {
        // move
        block_pair_copy_kernel<KEY_TYPE, VALUE_TYPE>(key + compacted_size, value + compacted_size,
                                                     update_keys + interval_a, update_values + interval_a, interval_size);
        __syncthreads();

        // set SIZE_NONE for executed update
        for (SIZE_TYPE i = interval_a + threadIdx.x; i < interval_b; i += blockDim.x)
        {
            update_nodes[i] = SIZE_NONE;
        }

        // re-dispatch
        block_redispatch_kernel<THREAD_PER_BLOCK, ITEM_PER_THREAD>(key, value, rebalance_width, seg_length,
                                                                   merge_size, row_offset, update_node);
    }
}

__global__ void label_key_whether_none_kernel(SIZE_TYPE *label, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size)
{
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset)
    {
        label[i] = (keys[i] == KEY_NONE || values[i] == VALUE_NONE) ? 0 : 1;
    }
}

__global__ void copy_compacted_kv(SIZE_TYPE *exscan, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size, KEY_TYPE *tmp_keys,
                                  VALUE_TYPE *tmp_values, SIZE_TYPE *compacted_size)
{

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset)
    {
        if (i == size - 1)
            continue;
        if (exscan[i] != exscan[i + 1])
        {
            SIZE_TYPE loc = exscan[i];
            tmp_keys[loc] = keys[i];
            tmp_values[loc] = values[i];
        }
    }

    if (0 == global_thread_id)
    {
        SIZE_TYPE loc = exscan[size - 1];
        if (keys[size - 1] == KEY_NONE || values[size - 1] == VALUE_NONE)
        {
            *compacted_size = loc;
        }
        else
        {
            *compacted_size = loc + 1;
            tmp_keys[loc] = keys[size - 1];
            tmp_values[loc] = values[size - 1];
        }
    }
}

__device__ void compact_kernel(SIZE_TYPE size, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *compacted_size,
                               KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, SIZE_TYPE *exscan, SIZE_TYPE *label)
{

    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
    label_key_whether_none_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(label, keys, values, size);
    cErr(cudaDeviceSynchronize());

    // exscan
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, label, exscan, size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, label, exscan, size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(d_temp_storage));

    // copy compacted kv to tmp, and set the original to none
    copy_compacted_kv<<<BLOCKS_NUM, THREADS_NUM>>>(exscan, keys, values, size, tmp_keys, tmp_values, compacted_size);
    cErr(cudaDeviceSynchronize());
}

__global__ void redispatch_kernel(KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, KEY_TYPE *keys, VALUE_TYPE *values,
                                  SIZE_TYPE update_width, SIZE_TYPE seg_length, SIZE_TYPE merge_size, SIZE_TYPE *row_offset,
                                  SIZE_TYPE update_node)
{

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    KEY_TYPE frac = update_width / seg_length;
    KEY_TYPE deno = merge_size;

    for (SIZE_TYPE i = global_thread_id; i < merge_size; i += block_offset)
    {
        SIZE_TYPE seg_idx = (SIZE_TYPE)(frac * i / deno);
        SIZE_TYPE seg_lane = (SIZE_TYPE)(frac * i % deno / frac);
        SIZE_TYPE proj_location = seg_idx * seg_length + seg_lane;
        KEY_TYPE cur_key = tmp_keys[i];
        VALUE_TYPE cur_value = tmp_values[i];
        keys[proj_location] = cur_key;
        values[proj_location] = cur_value;

        // addition for csr
        if ((cur_key & COL_IDX_NONE) == COL_IDX_NONE)
        {
            SIZE_TYPE cur_row = (SIZE_TYPE)(cur_key >> 32);
            row_offset[cur_row + 1] = proj_location + update_node;
        }
    }
}

__global__ void rebalancing_kernel(SIZE_TYPE unique_update_size, SIZE_TYPE seg_length, SIZE_TYPE level, KEY_TYPE *keys,
                                   VALUE_TYPE *values, SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values,
                                   SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE lower_bound, SIZE_TYPE upper_bound,
                                   SIZE_TYPE *row_offset)
{

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    SIZE_TYPE update_width = seg_length << level;

    SIZE_TYPE *compacted_size;
    cErr(cudaMalloc(&compacted_size, sizeof(SIZE_TYPE)));
    cErr(cudaDeviceSynchronize());

    KEY_TYPE *tmp_keys;
    VALUE_TYPE *tmp_values;
    SIZE_TYPE *tmp_exscan;
    SIZE_TYPE *tmp_label;

    cErr(cudaMalloc(&tmp_keys, update_width * sizeof(KEY_TYPE)));
    cErr(cudaMalloc(&tmp_values, update_width * sizeof(VALUE_TYPE)));
    cErr(cudaMalloc(&tmp_exscan, update_width * sizeof(SIZE_TYPE)));
    cErr(cudaMalloc(&tmp_label, update_width * sizeof(SIZE_TYPE)));
    cErr(cudaDeviceSynchronize());

    for (SIZE_TYPE i = global_thread_id; i < unique_update_size; i += block_offset)
    {
        SIZE_TYPE update_node = unique_update_nodes[i];
        KEY_TYPE *key = keys + update_node;
        VALUE_TYPE *value = values + update_node;

        // compact
        compact_kernel(update_width, key, value, compacted_size, tmp_keys, tmp_values, tmp_exscan, tmp_label);
        cErr(cudaDeviceSynchronize());

        // judge whether fit the density threshold
        SIZE_TYPE interval_a = update_offset[i];
        SIZE_TYPE interval_b = update_offset[i + 1];
        SIZE_TYPE interval_size = interval_b - interval_a;
        SIZE_TYPE merge_size = (*compacted_size) + interval_size;

        if (lower_bound <= merge_size && merge_size <= upper_bound)
        {
            SIZE_TYPE THREADS_NUM = 32;
            SIZE_TYPE BLOCKS_NUM;

            // move
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, interval_size);
            memcpy_kernel<KEY_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(tmp_keys + (*compacted_size),
                                                                 update_keys + interval_a, interval_size);
            memcpy_kernel<VALUE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(tmp_values + (*compacted_size),
                                                                   update_values + interval_a, interval_size);
            cErr(cudaDeviceSynchronize());

            // set SIZE_NONE for executed updates
            memset_kernel<SIZE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(update_nodes + interval_a, SIZE_NONE, interval_size);
            cErr(cudaDeviceSynchronize());

            cub_sort_key_value(tmp_keys, tmp_values, merge_size, key, value);

            // re-dispatch
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_width);
            memset_kernel<KEY_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(key, KEY_NONE, update_width);
            cErr(cudaDeviceSynchronize());

            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, merge_size);
            redispatch_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(tmp_keys, tmp_values, key, value, update_width, seg_length,
                                                           merge_size, row_offset, update_node);
            cErr(cudaDeviceSynchronize());
        }

        cErr(cudaDeviceSynchronize());
    }

    cErr(cudaFree(compacted_size));
    cErr(cudaFree(tmp_keys));
    cErr(cudaFree(tmp_values));
    cErr(cudaFree(tmp_exscan));
    cErr(cudaFree(tmp_label));
}

__host__ void rebalance_batch(SIZE_TYPE level, SIZE_TYPE seg_length, KEY_TYPE *keys, VALUE_TYPE *values,
                              SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size,
                              SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE unique_update_size,
                              SIZE_TYPE lower_bound, SIZE_TYPE upper_bound, SIZE_TYPE *row_offset)
{

    SIZE_TYPE update_width = seg_length << level;

    if (update_width <= 1024)
    {
        // func pointer for each template
        void (*func_arr[10])(SIZE_TYPE, SIZE_TYPE, KEY_TYPE *, VALUE_TYPE *, SIZE_TYPE *, KEY_TYPE *, VALUE_TYPE *,
                             SIZE_TYPE *, SIZE_TYPE *, SIZE_TYPE, SIZE_TYPE, SIZE_TYPE *);
        func_arr[0] = block_rebalancing_kernel<2, 1>;
        func_arr[1] = block_rebalancing_kernel<4, 1>;
        func_arr[2] = block_rebalancing_kernel<8, 1>;
        func_arr[3] = block_rebalancing_kernel<16, 1>;
        func_arr[4] = block_rebalancing_kernel<32, 1>;
        func_arr[5] = block_rebalancing_kernel<32, 2>;
        func_arr[6] = block_rebalancing_kernel<32, 4>;
        func_arr[7] = block_rebalancing_kernel<32, 8>;
        func_arr[8] = block_rebalancing_kernel<32, 16>;
        func_arr[9] = block_rebalancing_kernel<32, 32>;

        // operate each tree node by cuda-block
        SIZE_TYPE THREADS_NUM = update_width > 32 ? 32 : update_width;
        SIZE_TYPE BLOCKS_NUM = unique_update_size;

        func_arr[fls(update_width) - 2]<<<BLOCKS_NUM, THREADS_NUM>>>(seg_length, level, keys, values, update_nodes,
                                                                     update_keys, update_values, unique_update_nodes, update_offset, lower_bound, upper_bound, row_offset);
    }
    else
    {
        // operate each tree node by cub-kernel (dynamic parallelsim)
        SIZE_TYPE BLOCKS_NUM = min(2048, unique_update_size);
        rebalancing_kernel<<<BLOCKS_NUM, 1>>>(unique_update_size, seg_length, level, keys, values, update_nodes,
                                              update_keys, update_values, unique_update_nodes, update_offset, lower_bound, upper_bound, row_offset);
    }
    cErr(cudaDeviceSynchronize());
}

struct three_tuple_first_none
{
    typedef thrust::tuple<SIZE_TYPE, KEY_TYPE, VALUE_TYPE> Tuple;
    __host__ __device__ bool operator()(const Tuple &a)
    {
        return SIZE_NONE == thrust::get<0>(a);
    }
};
__host__ void compact_insertions(DEV_VEC_SIZE &update_nodes, DEV_VEC_KEY &update_keys, DEV_VEC_VALUE &update_values,
                                 SIZE_TYPE &update_size)
{

    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(update_nodes.begin(), update_keys.begin(), update_values.begin()));
    auto zip_end = thrust::remove_if(zip_begin, zip_begin + update_size, three_tuple_first_none());
    cErr(cudaDeviceSynchronize());
    update_size = zip_end - zip_begin;
}

__host__ SIZE_TYPE group_insertion_by_node(SIZE_TYPE *update_nodes, SIZE_TYPE update_size,
                                           SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset)
{

    // step1: encode
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    SIZE_TYPE *tmp_offset;
    cErr(cudaMalloc(&tmp_offset, sizeof(SIZE_TYPE) * update_size));

    SIZE_TYPE *num_runs_out;
    cErr(cudaMalloc(&num_runs_out, sizeof(SIZE_TYPE)));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, update_nodes,
                                            unique_update_nodes, tmp_offset, num_runs_out, update_size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, update_nodes,
                                            unique_update_nodes, tmp_offset, num_runs_out, update_size));
    cErr(cudaDeviceSynchronize());

    SIZE_TYPE unique_node_size[1];
    cErr(cudaMemcpy(unique_node_size, num_runs_out, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(num_runs_out));
    cErr(cudaFree(d_temp_storage));

    // step2: exclusive scan
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp_offset,
                                       update_offset, unique_node_size[0]));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp_offset,
                                       update_offset, unique_node_size[0]));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(d_temp_storage));

    cErr(cudaMemcpy(update_offset + unique_node_size[0], &update_size, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(tmp_offset));

    return unique_node_size[0];
}

__host__ void compress_insertions_by_node(DEV_VEC_SIZE &update_nodes, SIZE_TYPE update_size,
                                          DEV_VEC_SIZE &unique_update_nodes, DEV_VEC_SIZE &update_offset, SIZE_TYPE &unique_node_size)
{
    unique_node_size = group_insertion_by_node(RAW_PTR(update_nodes), update_size, RAW_PTR(unique_update_nodes),
                                               RAW_PTR(update_offset));
    cErr(cudaDeviceSynchronize());
}

__global__ void up_level_kernel(SIZE_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE update_width)
{
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;

    for (SIZE_TYPE i = global_thread_id; i < update_size; i += block_offset)
    {
        SIZE_TYPE node = update_nodes[i];
        update_nodes[i] = node & ~update_width;
    }
}

__host__ void up_level_batch(SIZE_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE update_width)
{
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size);
    up_level_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(update_nodes, update_size, update_width);
    cErr(cudaDeviceSynchronize());
}

struct kv_tuple_none
{
    typedef thrust::tuple<KEY_TYPE, VALUE_TYPE> Tuple;
    __host__ __device__ bool operator()(const Tuple &a)
    {
        return KEY_NONE == thrust::get<0>(a) || VALUE_NONE == thrust::get<1>(a);
    }
};
__host__ int resize_gpma(GPMA &gpma, DEV_VEC_KEY &update_keys, DEV_VEC_VALUE &update_values, SIZE_TYPE update_size)
{
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(gpma.keys.begin(), gpma.values.begin()));
    auto zip_end = thrust::remove_if(zip_begin, zip_begin + gpma.keys.size(), kv_tuple_none());
    cErr(cudaDeviceSynchronize());
    SIZE_TYPE compacted_size = zip_end - zip_begin;
    thrust::fill(gpma.keys.begin() + compacted_size, gpma.keys.end(), KEY_NONE);
    cErr(cudaDeviceSynchronize());

    SIZE_TYPE merge_size = compacted_size + update_size;
    SIZE_TYPE original_tree_size = gpma.keys.size();

    SIZE_TYPE tree_size = 4;
    while (floor(gpma.density_upper_thres_root * tree_size) < merge_size)
        tree_size <<= 1;
    gpma.segment_length = 1 << (fls(fls(tree_size)) - 1);
    gpma.tree_height = fls(tree_size / gpma.segment_length) - 1;

    gpma.keys.resize(tree_size, KEY_NONE);
    gpma.values.resize(tree_size);
    cErr(cudaDeviceSynchronize());
    recalculate_density(gpma);

    return compacted_size;
}

__host__ void significant_insert(GPMA &gpma, DEV_VEC_KEY &update_keys, DEV_VEC_VALUE &update_values, int update_size)
{
    int valid_size = resize_gpma(gpma, update_keys, update_values, update_size);
    thrust::copy(update_keys.begin(), update_keys.begin() + update_size, gpma.keys.begin() + valid_size);
    thrust::copy(update_values.begin(), update_values.begin() + update_size, gpma.values.begin() + valid_size);

    DEV_VEC_KEY tmp_update_keys(gpma.get_size());
    DEV_VEC_VALUE tmp_update_values(gpma.get_size());
    cErr(cudaDeviceSynchronize());

    int merge_size = valid_size + update_size;
    thrust::sort_by_key(gpma.keys.begin(), gpma.keys.begin() + merge_size, gpma.values.begin());
    cErr(cudaDeviceSynchronize());

    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM;
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, merge_size);
    redispatch_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(tmp_update_keys),
                                                   RAW_PTR(tmp_update_values), gpma.get_size(), gpma.segment_length, merge_size, RAW_PTR(gpma.row_offset), 0);
    cErr(cudaDeviceSynchronize());

    gpma.keys = tmp_update_keys;
    gpma.values = tmp_update_values;
    cErr(cudaDeviceSynchronize());
}

__host__ void update_gpma(GPMA &gpma, DEV_VEC_KEY &update_keys, DEV_VEC_VALUE &update_values)
{
    SIZE_TYPE ous = update_keys.size();

    // step1: sort update keys with values
    thrust::sort_by_key(update_keys.begin(), update_keys.end(), update_values.begin());
    cErr(cudaDeviceSynchronize());

    // step2: get leaf node of each update (execute del and mod)
    DEV_VEC_SIZE update_nodes(update_keys.size());
    cErr(cudaDeviceSynchronize());
    locate_leaf_batch(RAW_PTR(gpma.keys), RAW_PTR(gpma.values), gpma.keys.size(), gpma.segment_length, gpma.tree_height,
                      RAW_PTR(update_keys), RAW_PTR(update_values), update_keys.size(), RAW_PTR(update_nodes));
    cErr(cudaDeviceSynchronize());

    // step3: extract insertions
    DEV_VEC_SIZE unique_update_nodes(update_keys.size());
    DEV_VEC_SIZE update_offset(update_keys.size() + 1);
    cErr(cudaDeviceSynchronize());
    SIZE_TYPE update_size = update_nodes.size();
    SIZE_TYPE unique_node_size = 0;
    compact_insertions(update_nodes, update_keys, update_values, update_size);
    compress_insertions_by_node(update_nodes, update_size, unique_update_nodes, update_offset, unique_node_size);
    cErr(cudaDeviceSynchronize());

    // step4: rebuild for significant update
    int threshold = 5 * 1000 * 1000;
    if (update_size >= threshold)
    {
        significant_insert(gpma, update_keys, update_values, update_size);
        return;
    }

    // step5: rebalance each tree level
    for (SIZE_TYPE level = 0; level <= gpma.tree_height && update_size; level++)
    {
        SIZE_TYPE lower_bound = gpma.lower_element[level];
        SIZE_TYPE upper_bound = gpma.upper_element[level];

        // re-balance
        rebalance_batch(level, gpma.segment_length, RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(update_nodes),
                        RAW_PTR(update_keys), RAW_PTR(update_values), update_size, RAW_PTR(unique_update_nodes),
                        RAW_PTR(update_offset), unique_node_size, lower_bound, upper_bound, RAW_PTR(gpma.row_offset));

        // compact
        compact_insertions(update_nodes, update_keys, update_values, update_size);

        // up level
        up_level_batch(RAW_PTR(update_nodes), update_size, gpma.segment_length << level);

        // re-compress
        compress_insertions_by_node(update_nodes, update_size, unique_update_nodes, update_offset,
                                    unique_node_size);
    }

    // step6: rebalance the root node if necessary
    if (update_size > 0)
    {
        resize_gpma(gpma, update_keys, update_values, update_size);

        SIZE_TYPE level = gpma.tree_height;
        SIZE_TYPE lower_bound = gpma.lower_element[level];
        SIZE_TYPE upper_bound = gpma.upper_element[level];

        // re-balance
        cErr(cudaDeviceSynchronize());
        rebalance_batch(level, gpma.segment_length, RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(update_nodes),
                        RAW_PTR(update_keys), RAW_PTR(update_values), update_size, RAW_PTR(unique_update_nodes),
                        RAW_PTR(update_offset), unique_node_size, lower_bound, upper_bound, RAW_PTR(gpma.row_offset));
    }

    cErr(cudaDeviceSynchronize());
}

__host__ void build_gpma(GPMA &gpma)
{
    // we are creating the smallest possible GPMA tree
    // height = 1
    // levels = 2
    // leaf segment length = 2
    gpma.keys.resize(4, KEY_NONE);
    gpma.values.resize(4);

    cErr(cudaDeviceSynchronize());

    gpma.segment_length = 2;
    gpma.tree_height = 1;

    // the minimal tree structure has 2 levels with 4 elements' space, and the leaf segment's length is 2
    // put two MAX_KEY to keep minimal valid structure
    gpma.keys[0] = gpma.keys[2] = KEY_MAX;
    gpma.values[0] = gpma.values[2] = 1;

    recalculate_density(gpma);
}

template <typename T>
struct col_idx_none
{
    typedef T argument_type;
    typedef T result_type;
    __host__ __device__
        T
        operator()(const T &x) const
    {
        return (x << 32) + COL_IDX_NONE;
    }
};
__host__ void init_gpma(GPMA &gpma, SIZE_TYPE row_num)
{
    // gpma     (GPMA)          : The GPMA object whose CSR arrays are to initialised
    // row_num  (unsized int)   : Number of total nodes in the graph

    // initialising the row_offset vector with all 0 value
    gpma.row_num = row_num;
    gpma.row_offset.resize(row_num + 1, 0);

    // initialising in_degree and out_degree arrays
    // with all zero values
    gpma.in_degree.resize(row_num, 0);
    gpma.out_degree.resize(row_num, 0);

    // creates a device vector of size row_num
    DEV_VEC_KEY row_wall(row_num);

    // creates a device vector of size row_num all
    // initialised to the value 1
    DEV_VEC_VALUE tmp_value(row_num, 1);
    cErr(cudaDeviceSynchronize());

    thrust::tabulate(row_wall.begin(), row_wall.end(), col_idx_none<KEY_TYPE>());
    build_gpma(gpma);
    cErr(cudaDeviceSynchronize());
    update_gpma(gpma, row_wall, tmp_value);
    cErr(cudaDeviceSynchronize());
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void print_gpma_info(GPMA &gpma, int node)
{
    // Pretty prints the row_offset range and column_indices values
    // for the given node
    //
    // row_offset:  Row offset vector
    // col_indices: Column indices PMA
    // node:        Node whose row_offset and column_indices is to be printed

    // getting the (beg,end) range of the node

    thrust::host_vector<SIZE_TYPE> row_offset = gpma.row_offset;
    thrust::host_vector<KEY_TYPE> col_indices = gpma.keys;
    thrust::host_vector<VALUE_TYPE> edge_values = gpma.values;

    unsigned int beg = row_offset[node];
    unsigned int end = row_offset[node + 1];
    int print_count = 0;

    std::cout << "\n🌟 Column Indices for node: " << node << "(" << beg << ", " << end << ")"
              << "\n\n";
    std::cout << std::setw(6);

    for (int i = beg; i < end; ++i)
    {
        KEY_TYPE mask = (KEY_TYPE)node << 32;
        unsigned int dst = (col_indices[i] - mask);
        VALUE_TYPE val = edge_values[i];

        if (dst != COL_IDX_NONE && val != VALUE_NONE)
        {
            std::cout << dst << "(" << val << ")" << std::setw(6);
        }
        else
        {
            std::cout << "-" << std::setw(6);
        }
        print_count += 1;

        if (print_count > 10)
        {
            printf("\n");
            print_count = 0;
        }
    }
    printf("\n\n");
}

void load_data(const char *file_path, thrust::host_vector<int> &host_x, thrust::host_vector<int> &host_y,
               int &node_size, int &edge_size)
{

    FILE *fp;
    fp = fopen(file_path, "r");
    if (not fp)
    {
        printf("Open graph file failed.\n");
        exit(0);
    }

    // gets the number of nodes and edges from the first line
    // [num_nodes] [num_edges]
    fscanf(fp, "%d %d", &node_size, &edge_size);
    printf("node_num: %d, edge_num: %d\n", node_size, edge_size);

    // dynamic resizing to fit all edges
    host_x.resize(edge_size);
    host_y.resize(edge_size);

    // loads the src and dst nodes into x_host
    // and y_host respectively
    for (int i = 0; i < edge_size; i++)
    {
        int x, y;

        // [src_node] [dst_node]
        fscanf(fp, "%d %d", &x, &y);
        host_x[i] = x;
        host_y[i] = y;
    }

    printf("Graph file is loaded.\n");
    fclose(fp);
}

void load_graph(GPMA &gpma, const char *file_path)
{
    // sets the GPU Malloc Heap Size to 1GB
    // we could change the limit accordingly
    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024ll * 1024);

    // controls the maximum nesting depth of a grid at which
    // a thread can safely call cudaDeviceSynchronize()
    // cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 5);

    // we are defining two host_vectors on the CPU
    // host_x will contain the src nodes and
    // host_y will contain the destination nodes of the edge
    thrust::host_vector<int> host_x;
    thrust::host_vector<int> host_y;

    int node_size; // num of nodes
    int edge_size; // num of edges

    // loads the graph edges
    load_data(file_path, host_x, host_y, node_size, edge_size);

    // we are starting out with half the number of edges
    // as the initial graph. After this is done, the other
    // half of edges are added dynamically to the graph in
    // batches of 100 edges
    int half = edge_size / 2;

    // h_base_key is a host_vector to maintain a unique ID
    // for each edge. KEY_TYPE = unsigned long long.
    // The unique ID is a 64-bit number, where the first 32 bits
    // represent the source node and the last 32-bits the
    // destination node.
    thrust::host_vector<KEY_TYPE> h_base_keys(half);

    // looping over half the edges to create the unique edge ID.
    // It shifts the source node ID (64-bit after casting) by 32 bits
    // to the left and the 32-bit destination node ID is added
    // to the 32-bits on the left of source node ID
    for (int i = 0; i < half; i++)
        h_base_keys[i] = ((KEY_TYPE)host_x[i] << 32) + host_y[i];

    // device_vector<unsigned long long>. The unique edge IDs are
    // copied into the GPU device through a device_vector
    DEV_VEC_KEY base_keys = h_base_keys;

    // device_vector<double>. device_vector of size
    // half = num_edges/2. Contains the value of the edges.
    // Initialised to 1 and stored in GPU device
    DEV_VEC_VALUE base_values(half, 1);

    // the base keys and values are being transferred to
    // GPU device. Host is waiting for that task to complete
    // before continuing with the rest.
    cudaDeviceSynchronize();

    init_gpma(gpma, node_size); //
    cudaDeviceSynchronize();

    update_gpma(gpma, base_keys, base_values);
    cudaDeviceSynchronize();

    printf("Graph is updated.\n");
}

void edge_update_list(GPMA &gpma, std::vector<std::tuple<int, int>> edge_list, bool is_delete = false, bool is_reverse_edge = false)
{
    // NOTE:: Should we set these limits every single time?
    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024ll * 1024);
    // cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 5);

    thrust::host_vector<int> host_src;
    thrust::host_vector<int> host_dst;

    int edge_count = 0;

    // iterating through a vector of tuples
    // each tuple is of the form (src_node, dst_node)

    for (auto &edge : edge_list)
    {

        int src = (is_reverse_edge == true) ? std::get<1>(edge) : std::get<0>(edge);
        int dst = (is_reverse_edge == true) ? std::get<0>(edge) : std::get<1>(edge);

        // updating the in-degree and out-degree of
        // destination and source node respectively
        if (is_delete)
        {
            gpma.in_degree[dst] -= 1;
            gpma.out_degree[src] -= 1;
        }
        else
        {
            gpma.in_degree[dst] += 1;
            gpma.out_degree[src] += 1;
        }

        host_src.push_back(src);
        host_dst.push_back(dst);

        ++edge_count;
    }

    gpma.edge_count = is_delete ? gpma.edge_count - edge_count : gpma.edge_count + edge_count;

    thrust::host_vector<KEY_TYPE> h_base_keys(edge_count);

    for (int i = 0; i < edge_count; i++)
        h_base_keys[i] = ((KEY_TYPE)host_src[i] << 32) + host_dst[i];

    DEV_VEC_KEY base_keys = h_base_keys;
    cudaDeviceSynchronize();

    DEV_VEC_VALUE base_values(edge_count);

    if (is_delete)
        thrust::fill(base_values.begin(), base_values.end(), VALUE_NONE);
    else
        thrust::fill(base_values.begin(), base_values.end(), 1);

    cudaDeviceSynchronize();

    update_gpma(gpma, base_keys, base_values);
    cudaDeviceSynchronize();
}

void label_edges(GPMA &gpma)
{
    int edge_label_counter = 1;

    thrust::host_vector<SIZE_TYPE> h_row_offset = gpma.row_offset;
    thrust::host_vector<KEY_TYPE> h_keys = gpma.keys;
    thrust::host_vector<VALUE_TYPE> h_values = gpma.values;

    for (int node = 0; node < h_row_offset.size() - 1; ++node)
    {
        unsigned int beg = h_row_offset[node];
        unsigned int end = h_row_offset[node + 1];

        for (int i = beg; i < end; ++i)
        {
            KEY_TYPE mask = (KEY_TYPE)node << 32;
            unsigned int dst = (h_keys[i] - mask);
            if (dst != COL_IDX_NONE && h_values[i] != VALUE_NONE)
            {
                h_values[i] = edge_label_counter;
                edge_label_counter += 1;
            }
        }
    }

    gpma.row_offset = h_row_offset;
    gpma.keys = h_keys;
    gpma.values = h_values;
}

void copy_label_edges(GPMA &gpma, GPMA &ref_gpma)
{
    int edge_counter = 0;

    // creating the host vector version for keys
    // and values for the new gpma
    thrust::host_vector<KEY_TYPE> h_new_keys(ref_gpma.edge_count);
    thrust::host_vector<VALUE_TYPE> h_new_values(ref_gpma.edge_count);

    // creating the host vectors for the CSR arrays of the
    // reference GPMA from it's device vectors
    thrust::host_vector<SIZE_TYPE> h_ref_row_offset = ref_gpma.row_offset;
    thrust::host_vector<KEY_TYPE> h_ref_keys = ref_gpma.keys;
    thrust::host_vector<VALUE_TYPE> h_ref_values = ref_gpma.values;

    DEV_VEC_KEY d_new_keys(ref_gpma.edge_count);
    DEV_VEC_VALUE d_new_values(ref_gpma.edge_count);

    for (int node = 0; node < h_ref_row_offset.size() - 1; ++node)
    {
        unsigned int beg = h_ref_row_offset[node];
        unsigned int end = h_ref_row_offset[node + 1];

        for (int i = beg; i < end; ++i)
        {
            KEY_TYPE mask = (KEY_TYPE)node << 32;
            unsigned int dst = (h_ref_keys[i] - mask);
            if (dst != COL_IDX_NONE && h_ref_values[i] != VALUE_NONE)
            {
                h_new_keys[edge_counter] = (h_ref_keys[i] << 32) + node;
                h_new_values[edge_counter] = h_ref_values[i];
                edge_counter += 1;
            }
        }
    }

    d_new_keys = h_new_keys;
    d_new_values = h_new_values;

    // NOTE: Verify if sorting is really required or not
    thrust::sort_by_key(d_new_keys.begin(), d_new_keys.end(), d_new_values.begin());
    cudaDeviceSynchronize();

    locate_leaf_batch(RAW_PTR(gpma.keys), RAW_PTR(gpma.values), gpma.keys.size(), gpma.segment_length, gpma.tree_height,
                      RAW_PTR(d_new_keys), RAW_PTR(d_new_values), d_new_keys.size(), NULL, false);

    cudaDeviceSynchronize();
}

void build_reverse_gpma(GPMA &gpma, GPMA &ref_gpma)
{
    // This funtion expects an empty initialized GPMA
    // as first param and the reference GPMA from which
    // edges are to be added as the second param

    int edge_counter = 0;

    // creating host_vectors for the new GPMA
    thrust::host_vector<KEY_TYPE> h_new_keys(ref_gpma.edge_count);
    thrust::host_vector<VALUE_TYPE> h_new_values(ref_gpma.edge_count);

    // creating the host vectors for the CSR arrays of the
    // reference GPMA from it's device vectors
    thrust::host_vector<SIZE_TYPE> h_ref_row_offset = ref_gpma.row_offset;
    thrust::host_vector<KEY_TYPE> h_ref_keys = ref_gpma.keys;
    thrust::host_vector<VALUE_TYPE> h_ref_values = ref_gpma.values;

    DEV_VEC_KEY d_new_keys(ref_gpma.edge_count);
    DEV_VEC_VALUE d_new_values(ref_gpma.edge_count);

    for (int node = 0; node < h_ref_row_offset.size() - 1; ++node)
    {
        unsigned int beg = h_ref_row_offset[node];
        unsigned int end = h_ref_row_offset[node + 1];

        for (int i = beg; i < end; ++i)
        {
            KEY_TYPE mask = (KEY_TYPE)node << 32;
            unsigned int dst = (h_ref_keys[i] - mask);
            if (dst != COL_IDX_NONE && h_ref_values[i] != VALUE_NONE)
            {
                h_new_keys[edge_counter] = (h_ref_keys[i] << 32) + node;
                h_new_values[edge_counter] = h_ref_values[i];
                edge_counter += 1;
            }
        }
    }

    d_new_keys = h_new_keys;
    d_new_values = h_new_values;

    update_gpma(gpma, d_new_keys, d_new_values);
    cudaDeviceSynchronize();
}

std::tuple<std::size_t, std::size_t, std::size_t> get_csr_ptrs(GPMA &gpma)
{
    std::tuple<std::size_t, std::size_t, std::size_t> t;
    std::get<0>(t) = (std::size_t)RAW_PTR(gpma.row_offset);
    std::get<1>(t) = (std::size_t)RAW_PTR(gpma.keys);
    std::get<2>(t) = (std::size_t)RAW_PTR(gpma.values);
    return t;
}

std::tuple<int, int> get_graph_attr(GPMA &gpma)
{
    std::tuple<int, int> t;

    // Since row_offset has size num_nodes + 1
    std::get<0>(t) = gpma.row_offset.size() - 1;
    std::get<1>(t) = gpma.edge_count;
    return t;
}

PYBIND11_MODULE(gpma, m)
{
    m.doc() = "CPython module for GPMA"; // optional module docstring

    m.def("init_gpma", &init_gpma, "Initialises the CSR arrays using GPMA");
    m.def("print_gpma_info", &print_gpma_info, "Prints row_offset and col_indices for a given node");
    m.def("load_graph", &load_graph, "Loads a graph data into a GPMA");
    m.def("edge_update_list", &edge_update_list, "Updates the GPMA by adding/deleting edges from the edge list", py::arg("gpma"), py::arg("edge_list"), py::arg("is_delete") = false, py::arg("is_reverse_edge") = false);
    m.def("label_edges", &label_edges, "Creates edge labels for the current GPMA");
    m.def("copy_label_edges", &copy_label_edges, "Label edges of a GPMA based on another GPMA");
    m.def("build_reverse_gpma", &build_reverse_gpma, "Builds the reverse GPMA based on another GPMA");
    m.def("get_csr_ptrs", &get_csr_ptrs, "Returns the pointers to row_offset, col_indices and edge_ids");
    m.def("get_graph_attr", &get_graph_attr, "Returns the [num_nodes, num_edges]");

    py::class_<GPMA>(m, "GPMA")
        .def(py::init<>())
        .def("get_size", &GPMA::get_size)
        .def_readwrite("keys", &GPMA::keys)
        .def_readwrite("values", &GPMA::values)
        .def_readwrite("segment_length", &GPMA::segment_length)
        .def_readwrite("tree_height", &GPMA::tree_height)
        .def_readwrite("density_lower_thres_leaf", &GPMA::density_lower_thres_leaf)
        .def_readwrite("density_lower_thres_root", &GPMA::density_lower_thres_root)
        .def_readwrite("density_upper_thres_root", &GPMA::density_upper_thres_root)
        .def_readwrite("density_upper_thres_leaf", &GPMA::density_upper_thres_leaf)
        .def_readwrite("lower_element", &GPMA::lower_element)
        .def_readwrite("upper_element", &GPMA::upper_element)
        .def_readwrite("row_num", &GPMA::row_num)
        .def_readwrite("row_offset", &GPMA::row_offset)
        .def_readwrite("in_degree", &GPMA::in_degree)
        .def_readwrite("out_degree", &GPMA::out_degree)
        .def("__copy__", [](const GPMA &self)
             { return GPMA(self); })
        .def(
            "__deepcopy__", [](const GPMA &self, py::dict)
            { return GPMA(self); },
            "memo"_a);
}

// Command used:
// /usr/local/cuda-11.7/bin/nvcc $(python3 -m pybind11 --includes) -shared -rdc=true --compiler-options '-fPIC'  -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -o gpma.so gpma.cu