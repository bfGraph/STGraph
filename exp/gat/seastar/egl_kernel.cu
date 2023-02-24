
extern "C" __global__ void K4(float *Velinb, float *Vercen, float *V36, float *V37, 
  int *row_offsets,
  int *eids,
  int *column_indices,
  int num_nodes,
  int max_dimx,
  int max_dimy,
  int thrs_per_group,
  int nodes_per_block) {
    int dst_id = nodes_per_block*blockIdx.x + threadIdx.x/thrs_per_group;;
    if (dst_id < num_nodes) {
        int feat_len = max_dimx * max_dimy;
        int beg = __ldg(row_offsets + dst_id);
        int end = __ldg(row_offsets + dst_id + 1);
        int tx = threadIdx.x % thrs_per_group;
        for (; tx<feat_len; tx+=blockDim.x) {
            float V37_tmp = 0;
            int offset1 = dst_id * 1 + tx;
            for (int e=beg;e<end;++e) {
                int src_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                int offset0 = src_id * 1 + tx;int offset2 = eid * 1 + tx;
                
                
                float V34_tmp = Velinb[offset0] + Vercen[offset1];
                
                
                float V35_tmp=V34_tmp>0?V34_tmp:0.2*V34_tmp;
                
                
                float V36_tmp = exp(V35_tmp);
                V36[offset2] = V36_tmp;
                
                V37_tmp += V36_tmp;
                
            }
            
            V37[offset1] = V37_tmp;
            
        }
    }
}
extern "C" __global__ void K5(float *V36, float *V37, float *Vfeat_srcinb, float *V40, 
  int *row_offsets,
  int *eids,
  int *column_indices,
  int num_nodes,
  int max_dimx,
  int max_dimy,
  int thrs_per_group,
  int nodes_per_block) {
    int dst_id = nodes_per_block*blockIdx.x + threadIdx.x/thrs_per_group;;
    if (dst_id < num_nodes) {
        int feat_len = max_dimx * max_dimy;
        int beg = __ldg(row_offsets + dst_id);
        int end = __ldg(row_offsets + dst_id + 1);
        int tx = threadIdx.x % thrs_per_group;
        for (; tx<feat_len; tx+=blockDim.x) {
            float V40_tmp = 0;
            int offset1 = dst_id * 1 + tx/2;int offset3 = dst_id * 2 + tx;
            for (int e=beg;e<end;++e) {
                int src_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                int offset2 = src_id * 2 + tx;int offset0 = eid * 1 + tx/2;
                
                
                float V38_tmp = V36[offset0]/V37[offset1];
                
                
                float V39_tmp = V38_tmp*Vfeat_srcinb[offset2];
                
                
                V40_tmp += V39_tmp;
                
            }
            
            V40[offset3] = V40_tmp;
            
        }
    }
}
extern "C" __global__ void K6(float *V36, float *V37, float *V40, float *V41, float *Velinb, float *Vercen, float *Vfeat_srcinb, float *V46, float *V60, float *V62, 
  int *row_offsets,
  int *eids,
  int *column_indices,
  int num_nodes,
  int max_dimx,
  int max_dimy,
  int thrs_per_group,
  int nodes_per_block) {
    int src_id = nodes_per_block*blockIdx.x + threadIdx.x/thrs_per_group;;
    if (src_id < num_nodes) {
        int feat_len = max_dimx * max_dimy;
        int beg = __ldg(row_offsets + src_id);
        int end = __ldg(row_offsets + src_id + 1);
        int tx = threadIdx.x % thrs_per_group;
        for (; tx<feat_len; tx+=blockDim.x) {
            float V60_tmp = 0;float V62_tmp = 0;float V46_tmp = 0;
            int offset3 = src_id * 1 + tx/2;int offset4 = src_id * 2 + tx;
            for (int e=beg;e<end;++e) {
                int dst_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                int offset1 = dst_id * 1 + tx/2;int offset2 = dst_id * 2 + tx;int offset0 = eid * 1 + tx/2;
                
                
                float V38_tmp = V36[offset0]/V37[offset1];
                
                
                float V45_tmp = V41[offset2]*V38_tmp;
                
                
                float V34_tmp = Velinb[offset3] + Vercen[offset1];
                
                
                float V43_tmp = V41[offset2]*Vfeat_srcinb[offset4];
                
                
                float V47_tmp = 1/V37[offset1];
                
                
                float V48_tmp = V43_tmp*V47_tmp;
                
                
                float V49_tmp = V41[offset2]/V37[offset1];
                
                
                float V50_tmp = V49_tmp*V40[offset2];
                
                
                float V51_tmp = -1*V50_tmp;
                
                
                float V55_tmp = V48_tmp + V51_tmp;
                
                
                float V56_tmp = V55_tmp*V36[offset0];
                
                
                float V57_tmp = V34_tmp>0?1:0.2;
                
                
                float V58_tmp = V56_tmp*V57_tmp;
                
                
                V60_tmp += V58_tmp;
                
                V62_tmp = V58_tmp;
                atomicAdd(V62+offset1, V62_tmp);
                V46_tmp += V45_tmp;
                
            }
            
            atomicAdd(V60+offset3, V60_tmp);
            
            V46[offset4] = V46_tmp;
            
        }
    }
}