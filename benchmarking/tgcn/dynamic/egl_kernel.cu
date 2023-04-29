
extern "C" __global__ void K4(float *Vhinb, float *Vnormcen, float *Vnorminb, float *Vweight, float *V23, 
  unsigned int *row_offsets,
  double *eids,
  unsigned long long *column_indices,
  long long int num_nodes,
  long long int max_dimx,
  long long int max_dimy,
  long long int thrs_per_group,
  long long int nodes_per_block) {
    long long int dst_id = nodes_per_block*blockIdx.x + threadIdx.x/thrs_per_group;;
    if (dst_id < num_nodes) {
        long long int feat_len = max_dimx * max_dimy;
        long long int beg = __ldg(row_offsets + dst_id);
        long long int end = __ldg(row_offsets + dst_id + 1);
        long long int tx = threadIdx.x % thrs_per_group;
        for (; tx<feat_len; tx+=blockDim.x) {
            float V22_tmp = 0;
            long long int offset3 = dst_id * 32 + tx;long long int offset4 = dst_id * 1 + tx/32;
            for (long long int e=beg;e<end;++e) {
                long long int src_id = __ldg(column_indices + e);
                long long int eid = __ldg(eids + e);
                
                unsigned long long mask = (unsigned long long)dst_id << 32;
                unsigned int dst_check = (src_id - mask);
                src_id = (src_id - mask);
                if(dst_check != 0xFFFFFFFF && eid != 0){
                    long long int offset0 = src_id * 1 + tx/32;long long int offset1 = src_id * 32 + tx;long long int offset2 = eid * 1 + tx/32;
                    
                    
                    float V20_tmp = Vnorminb[offset0]*Vhinb[offset1];
                    
                    
                    float V21_tmp = V20_tmp*Vweight[offset2];
                    
                    
                    V22_tmp += V21_tmp;
                        
                }
            }
            
            
            
            float V23_tmp = V22_tmp*Vnormcen[offset4];V23[offset3] = V23_tmp;
        }
    }
}
extern "C" __global__ void K5(float *V24, float *Vnormcen, float *Vnorminb, float *Vweight, float *V29, 
  unsigned int *row_offsets,
  double *eids,
  unsigned long long *column_indices,
  long long int num_nodes,
  long long int max_dimx,
  long long int max_dimy,
  long long int thrs_per_group,
  long long int nodes_per_block) {
    long long int src_id = nodes_per_block*blockIdx.x + threadIdx.x/thrs_per_group;;
    if (src_id < num_nodes) {
        long long int feat_len = max_dimx * max_dimy;
        long long int beg = __ldg(row_offsets + src_id);
        long long int end = __ldg(row_offsets + src_id + 1);
        long long int tx = threadIdx.x % thrs_per_group;
        for (; tx<feat_len; tx+=blockDim.x) {
            float V28_tmp = 0;
            long long int offset3 = src_id * 32 + tx;long long int offset4 = src_id * 1 + tx/32;
            for (long long int e=beg;e<end;++e) {
                long long int dst_id = __ldg(column_indices + e);
                long long int eid = __ldg(eids + e);
                
                unsigned long long mask = (unsigned long long)src_id << 32;
                unsigned int dst_check = (dst_id - mask);
                dst_id = (dst_id - mask);
                if(dst_check != 0xFFFFFFFF && eid != 0){
                    long long int offset0 = dst_id * 32 + tx;long long int offset1 = dst_id * 1 + tx/32;long long int offset2 = eid * 1 + tx/32;
                    
                    
                    float V25_tmp = V24[offset0]*Vnormcen[offset1];
                    
                    
                    float V27_tmp = V25_tmp*Vweight[offset2];
                    
                    
                    V28_tmp += V27_tmp;
                        
                }
            }
            
            
            
            float V29_tmp = V28_tmp*Vnorminb[offset4];V29[offset3] = V29_tmp;
        }
    }
}