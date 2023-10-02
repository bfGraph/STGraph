extern "C" __global__ void K4
(float *Vedge_weight, float *Vhinb, float *Vnormcen, float *Vnorminb, float *V23, 
  int *row_offsets,
  int *eids,
  int *column_indices,
  int *node_ids,
  int num_nodes,
  int max_dimx,
  int max_dimy,
  int thrs_per_group,
  int nodes_per_block) {
      
    int dst_id = nodes_per_block*blockIdx.x + threadIdx.x/thrs_per_group;

    if (dst_id < num_nodes) {
        
        int feat_len = max_dimx * max_dimy;
        int beg = __ldg(row_offsets + dst_id);
        int end = __ldg(row_offsets + dst_id + 1);
        int tx = threadIdx.x % thrs_per_group;
        
        for (; tx<feat_len; tx+=blockDim.x) {
            
            float V22_tmp = 0;
            int offset3 = dst_id * 96 + tx;int offset4 = dst_id * 1 + tx/96;
            
            for (int e=beg;e<end;++e) {
                
                int src_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                
                int offset0 = src_id * 96 + tx;int offset1 = src_id * 1 + tx/96;int offset2 = eid * 1 + tx/96;
                
                
                
                float V20_tmp = Vnorminb[offset1]*Vhinb[offset0];
                
                
                
                float V21_tmp = V20_tmp*Vedge_weight[offset2];
                
                
                
                
                V22_tmp += V21_tmp;
                
                
            }
            
            
            
            
            
            
            
            float V23_tmp = V22_tmp*Vnormcen[offset4];
            V23[offset3] = V23_tmp;
            
        }
    }
}extern "C" __global__ void K5
(float *V24, float *Vedge_weight, float *Vnormcen, float *Vnorminb, float *V29, 
  int *row_offsets,
  int *eids,
  int *column_indices,
  int *node_ids,
  int num_nodes,
  int max_dimx,
  int max_dimy,
  int thrs_per_group,
  int nodes_per_block) {
      
    int src_id = nodes_per_block*blockIdx.x + threadIdx.x/thrs_per_group;

    if (src_id < num_nodes) {
        
        int feat_len = max_dimx * max_dimy;
        int beg = __ldg(row_offsets + src_id);
        int end = __ldg(row_offsets + src_id + 1);
        int tx = threadIdx.x % thrs_per_group;
        
        for (; tx<feat_len; tx+=blockDim.x) {
            
            float V28_tmp = 0;
            int offset3 = src_id * 1 + tx/96;int offset4 = src_id * 96 + tx;
            
            for (int e=beg;e<end;++e) {
                
                int dst_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                
                int offset0 = dst_id * 96 + tx;int offset1 = dst_id * 1 + tx/96;int offset2 = eid * 1 + tx/96;
                
                
                
                float V25_tmp = V24[offset0]*Vnormcen[offset1];
                
                
                
                float V27_tmp = V25_tmp*Vedge_weight[offset2];
                
                
                
                
                V28_tmp += V27_tmp;
                
                
            }
            
            
            
            
            
            
            
            float V29_tmp = V28_tmp*Vnorminb[offset3];
            V29[offset4] = V29_tmp;
            
        }
    }
}