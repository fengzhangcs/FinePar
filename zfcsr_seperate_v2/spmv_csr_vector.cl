
#define VEC2DWIDTH 1024
#define CSR_VEC_GROUP_SIZE 64
//#define WARPSIZE 64
#define WARPSIZE 32


//__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num )
__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num , __global unsigned long* bitmap)
{

    int row = get_global_id(0);
    int getnumsum=get_global_size(0);
    
    for(;row<row_num; row+=getnumsum){
      //int row=rowall[r];
      if(bitmap[row>>6]&(1ul<<(row&0x3f))){
        float sum = result[row];
        int start = rowptr[row];
        int end = rowptr[row+1];
        for (int i = start; i < end; i++)
        {
          int col = colid[i];
          sum = mad(data[i], vec[col], sum);
        }
        result[row] = sum;
      }  
    }



}


__kernel void cpu_csr(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int rowtotal, __global int* rowall, int rowallsize){

    //int row = get_global_id(0);
    int r = get_global_id(0);
    int getnumsum=get_global_size(0);
    
    for(;r<rowallsize; r+=getnumsum){
      int row=rowall[r];
//      if (row < row_num)
 //     {   
        float sum = result[row];
        int start = rowptr[row];
        int end = rowptr[row+1];
        for (int i = start; i < end; i++)
        {
          int col = colid[i];
          sum = mad(data[i], vec[col], sum);
        }
        result[row] = sum;
  //    }  
    }

}
/*
__kernel void cpu_csr(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int rowtotal, __global int* rowall){

    //int row = get_global_id(0);
    int r = get_global_id(0);
    int row=rowall[r];
    
    //if (row < row_num)
    {   
        float sum = result[row];
        int start = rowptr[row];
        int end = rowptr[row+1];
        for (int i = start; i < end; i++)
        {
            int col = colid[i];
            sum = mad(data[i], vec[col], sum);
        }
        result[row] = sum;
    }  

}
*/
