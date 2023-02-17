
extern "C" __global__ void K0(float *Vhinb, float *Vnormcen, float *Vnorminb, float *V2, 
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
            float V1_tmp = 0;
            int offset2 = dst_id * 16 + tx;int offset3 = dst_id * 1 + tx/16;
            for (int e=beg;e<end;++e) {
                int src_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                int offset0 = src_id * 16 + tx;int offset1 = src_id * 1 + tx/16;
                
                
                float V0_tmp = Vhinb[offset0]*Vnorminb[offset1];
                
                
                V1_tmp += V0_tmp;
                
            }
            
            
            
            float V2_tmp = V1_tmp*Vnormcen[offset3];V2[offset2] = V2_tmp;
        }
    }
}
extern "C" __global__ void K1(float *V3, float *Vnormcen, float *Vnorminb, float *V7, 
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
            float V6_tmp = 0;
            int offset2 = src_id * 1 + tx/16;int offset3 = src_id * 16 + tx;
            for (int e=beg;e<end;++e) {
                int dst_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                int offset0 = dst_id * 1 + tx/16;int offset1 = dst_id * 16 + tx;
                
                
                float V4_tmp = V3[offset1]*Vnormcen[offset0];
                
                
                V6_tmp += V4_tmp;
                
            }
            
            
            
            float V7_tmp = V6_tmp*Vnorminb[offset2];V7[offset3] = V7_tmp;
        }
    }
}