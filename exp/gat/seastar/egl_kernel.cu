
extern "C" __global__ void K0(float *Velinb, float *Vercen, float *V2, float *V3, 
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
            float V3_tmp = 0;
            int offset1 = dst_id * 8 + tx;
            for (int e=beg;e<end;++e) {
                int src_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                int offset0 = src_id * 8 + tx;int offset2 = eid * 8 + tx;
                
                
                float V0_tmp = Velinb[offset0] + Vercen[offset1];
                
                
                float V1_tmp=V0_tmp>0?V0_tmp:0.2*V0_tmp;
                
                
                float V2_tmp = exp(V1_tmp);
                V2[offset2] = V2_tmp;
                
                V3_tmp += V2_tmp;
                
            }
            
            V3[offset1] = V3_tmp;
            
        }
    }
}
extern "C" __global__ void K1(float *V2, float *V3, float *Vfeat_srcinb, float *V6, 
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
            float V6_tmp = 0;
            int offset1 = dst_id * 8 + tx;
            for (int e=beg;e<end;++e) {
                int src_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                int offset2 = src_id * 8 + tx;int offset0 = eid * 8 + tx;
                
                
                float V4_tmp = V2[offset0]/V3[offset1];
                
                
                float V5_tmp = V4_tmp*Vfeat_srcinb[offset2];
                
                
                V6_tmp += V5_tmp;
                
            }
            
            V6[offset1] = V6_tmp;
            
        }
    }
}
extern "C" __global__ void K2(float *V2, float *V3, float *V6, float *V7, float *Velinb, float *Vercen, float *Vfeat_srcinb, float *V11, float *V25, float *V27, 
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
            float V25_tmp = 0;float V27_tmp = 0;float V11_tmp = 0;
            int offset2 = src_id * 8 + tx;
            for (int e=beg;e<end;++e) {
                int dst_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                int offset1 = dst_id * 8 + tx;int offset0 = eid * 8 + tx;
                
                
                float V4_tmp = V2[offset0]/V3[offset1];
                
                
                float V10_tmp = V7[offset1]*V4_tmp;
                
                
                float V0_tmp = Velinb[offset2] + Vercen[offset1];
                
                
                float V9_tmp = V7[offset1]*Vfeat_srcinb[offset2];
                
                
                float V12_tmp = 1/V3[offset1];
                
                
                float V13_tmp = V9_tmp*V12_tmp;
                
                
                float V14_tmp = V7[offset1]/V3[offset1];
                
                
                float V15_tmp = V14_tmp*V6[offset1];
                
                
                float V16_tmp = -1*V15_tmp;
                
                
                float V20_tmp = V13_tmp + V16_tmp;
                
                
                float V21_tmp = V20_tmp*V2[offset0];
                
                
                float V22_tmp = V0_tmp>0?1:0.2;
                
                
                float V23_tmp = V21_tmp*V22_tmp;
                
                
                V25_tmp += V23_tmp;
                
                V27_tmp = V23_tmp;
                atomicAdd(V27+offset1, V27_tmp);
                V11_tmp += V10_tmp;
                
            }
            
            V25[offset2] = V25_tmp;
            
            V11[offset2] = V11_tmp;
            
        }
    }
}