
#define VEC2DWIDTH 1024
#define CSR_VEC_GROUP_SIZE 64
//#define WARPSIZE 64
#define WARPSIZE 32





__kernel void dckernel(__global int* rowptr,  __global int* colid, volatile __global int* resin, __global int* resout, int rownum){

    int row = get_global_id(0);
    int getnumsum=get_global_size(0);
   // int r = get_global_id(0);
   // int row=rowall[r];
    
    //if (row < row_num)
    for(;row<rownum; row+=getnumsum)
    {   
        //float sum = result[row];
        int start = rowptr[row];
        int end = rowptr[row+1];
        resout[row]=end-start;

        for (int i = start; i < end; i++)
        {
            int col = colid[i];
            atomic_add(&resin[col],1);
        }
    }  
}


