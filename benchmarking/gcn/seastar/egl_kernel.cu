extern "C" __global__ void K2
(float *Vhinb, float *Vnormcen, float *Vnorminb, float *V10, 
  int *row_offsets,
  int *eids,
  int *column_indices,
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
            
            float V9_tmp = 0;
            int offset2 = dst_id * 7 + tx;int offset3 = dst_id * 1 + tx/7;
            
            for (int e=beg;e<end;++e) {
                
                int src_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                
                int offset0 = src_id * 7 + tx;int offset1 = src_id * 1 + tx/7;
                
                
                
                float V8_tmp = Vhinb[offset0]*Vnorminb[offset1];
                
                
                
                
                V9_tmp += V8_tmp;
                
                
            }
            
            
            
            
            
            
            
            float V10_tmp = V9_tmp*Vnormcen[offset3];
            V10[offset2] = V10_tmp;
            
        }
    }
}extern "C" __global__ void K3
(float *V11, float *Vnormcen, float *Vnorminb, float *V15, 
  int *row_offsets,
  int *eids,
  int *column_indices,
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
            
            float V14_tmp = 0;
            int offset2 = src_id * 7 + tx;int offset3 = src_id * 1 + tx/7;
            
            for (int e=beg;e<end;++e) {
                
                int dst_id = __ldg(column_indices + e);
                int eid = __ldg(eids + e);
                
                int offset0 = dst_id * 1 + tx/7;int offset1 = dst_id * 7 + tx;
                
                
                
                float V12_tmp = V11[offset1]*Vnormcen[offset0];
                
                
                
                
                V14_tmp += V12_tmp;
                
                
            }
            
            
            
            
            
            
            
            float V15_tmp = V14_tmp*Vnorminb[offset3];
            V15[offset2] = V15_tmp;
            
        }
    }
}