#define SINGLE_PRECISION
#ifdef SINGLE_PRECISION
#define FPTYPE float
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FPTYPE double
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define FPTYPE double
#endif


#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif





__kernel void
spmv_ellpackr_kernel(__global const FPTYPE * restrict val,
                     __global const  FPTYPE * restrict vec,
                     __global const int * restrict cols,
                     __global const int * restrict rowLengths,
                     const int dim, __global FPTYPE * restrict out, 
                      __global unsigned long* bitmap,    const int rowsetzf)
{
    int t = get_global_id(0);

//    if ((bitmap[t>>6]&(1ul<<(t&0x3f)))&&(t < dim))
    {   
        FPTYPE result = 0.0;
        int max = rowLengths[t];
        for (int i = 0; i < max; i++)
        {
            int ind = i * dim + t;
                  result += val[ind] * vec[cols[ind]];
        }
        out[t] = result;
    }   
}


__kernel void cpu_csr(__global const FPTYPE * restrict val,
                     __global const  FPTYPE * restrict vec,
                     __global const int * restrict cols,
                     __global const int * restrict rowLengths,
                     const int dim, __global FPTYPE * restrict out,
                     __global int* rowall,
                     const int rowallsize,
                     const int maxrl,    const int rowsetzf
){

    //int row = get_global_id(0);
    int r = get_global_id(0)+rowsetzf;
    int getnumsum=get_global_size(0);
    
    for(;r<dim; r+=getnumsum){
    //for(;r<rowallsize; r+=getnumsum){
    
      int row=r;
      //int row=rowall[r];
        FPTYPE result = 0.0;
        int max = rowLengths[row];
        for (int i = 0; i < max; i++)
        {
            int ind = i  + row*maxrl;
            //int ind = i * dim + row;
            result += val[ind] * vec[cols[ind]];
        }
        out[row] = result;
    }
    
}

/*
__kernel void cpu_csr(__global const FPTYPE * restrict val,
                     __global const  FPTYPE * restrict vec,
                     __global const int * restrict cols,
                     __global const int * restrict rowLengths,
                     const int dim, __global FPTYPE * restrict out,
                     __global int* rowall,
                     const int rowallsize
){

    //int row = get_global_id(0);
    int r = get_global_id(0);
    int getnumsum=get_global_size(0);
    
    for(;r<rowallsize; r+=getnumsum){
    
      int row=rowall[r];
        FPTYPE result = 0.0;
        int max = rowLengths[row];
        for (int i = 0; i < max; i++)
        {
            int ind = i * dim + row;
                  result += val[ind] * vec[cols[ind]];
        }
        out[row] = result;
    }
    
}
*/


__kernel void cpu_csr_only(__global const FPTYPE * restrict val,
                     __global const  FPTYPE * restrict vec,
                     __global const int * restrict cols,
                     __global const int * restrict rowLengths,
                     const int dim, __global FPTYPE * restrict out,
                     __global int* rowall,
                     const int rowallsize,
                     const int maxrl
){

    //int row = get_global_id(0);
    int r = get_global_id(0);
    int getnumsum=get_global_size(0);
    
    for(;r<dim; r+=getnumsum){
    //for(;r<rowallsize; r+=getnumsum){
    
      int row=r;
        FPTYPE result = 0.0;
        int max = rowLengths[row];
        for (int i = 0; i < max; i++)
        {
            int ind = i  + row*maxrl;
            //int ind = i * dim + row;
            result += val[ind] * vec[cols[ind]];
        }
        out[row] = result;
    }
    
}


