
#define VEC2DWIDTH 1024
#define CSR_VEC_GROUP_SIZE 64
//#define WARPSIZE 64
#define WARPSIZE 32
#define DAM 0.85


__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num , int midrow, 
    __global int* rowinfo, int start)
//__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num , __global unsigned long* bitmap, __global float* prold, __global float* prnew, int midrow)
{


    int row = get_global_id(0);
    int getnumsum=get_global_size(0);
//    r+=midrow;//zf
 
    int startRow=rowinfo[start];
    int endRow=rowinfo[start+1];   
    row+=startRow;


    for(;row<endRow; row+=getnumsum){
 //   for(;row<midrow; row+=getnumsum){
        float sum = 0;
        //float sum = result[row];
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

__kernel void cpu_csr_ve_slm_pm_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int rowallsize, int midrow, 
    __global int* rowinfo, int start){
//__kernel void cpu_csr(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int rowtotal, __global int* rowall, int rowallsize, __global float* prold, __global float* prnew,  int midrow){

    int row = get_global_id(0);
    //int row = get_global_id(0)+midrow;
    int getnumsum=get_global_size(0);
//    r+=midrow;//zf
  
    
    int startRow=rowinfo[start];
    int endRow=rowinfo[start+1];   
    row+=startRow;

  
    for(;row<endRow; row+=getnumsum){
      
    //row += midrow;
    //for(;row<rowallsize; row+=getnumsum){
    //  int row=r;
      //int row=rowall[r];
//      if (row < row_num)
 //     {   
        float sum = 0;
        //float sum = result[row];
        int start = rowptr[row];
        int end = rowptr[row+1];
        for (int i = start; i < end; i++)
        {
          int col = colid[i];
          sum = mad(data[i], vec[col], sum);
        }
        //sum=sum*DAM + (1.0-DAM)*vec[row];
        result[row] = sum;
        //result[row] = sum;
  //    }  
    }

}



