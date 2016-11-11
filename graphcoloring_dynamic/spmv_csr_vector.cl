
#define VEC2DWIDTH 1024
#define CSR_VEC_GROUP_SIZE 64
//#define WARPSIZE 64
#define WARPSIZE 32
#define DAM 0.85

#define MY_INFINITY    0xffffff00


__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  
    __global int* colid, 
    __global int* randlist,
    __global int* vplist,
    int color,
    __global char* over,
    int rownum,
    int mid, 
    __global int* rowinfo, int start){
//__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num , __global unsigned long* bitmap, __global float* prold, __global float* prnew, int midrow){
  int tid=get_global_id(0);
  int getnumsum=get_global_size(0);

    int startRow=rowinfo[start];
    int endRow=rowinfo[start+1];
    tid+=startRow;

  for(;tid<endRow; tid+=getnumsum){
  //for(;tid<mid; tid+=getnumsum){
  //for(;tid<rownum; tid+=getnumsum){
    if(vplist[tid]!=MY_INFINITY)
      continue;
    int vid=tid;
    int start=rowptr[vid];
    int end=rowptr[vid+1];
    int local_rand=randlist[vid];
    char found_larger=0;
    for(int i=start; i<end; i++){
      int dest=colid[i];
      if((vplist[dest]<color)&&vplist[dest]>=0)
        continue;
      if(  (randlist[dest]>local_rand)  ||
          (   (randlist[dest]==local_rand) && (dest<vid)) ){
        found_larger=1;
        break;
      }
    }
    if(found_larger==0){
      vplist[vid]=color;
      //if(vid==0)printf("%d\n",vplist[vid]);
    }
    else{
      *over=0;
      //printf("*over=%d\n",*over);
    }
      
  }

}


__kernel void cpu_csr(__global int* rowptr,  
    __global int* colid, 
    __global int* randlist,
    __global int* vplist,
    int color,
    __global int* over,
    int rownum,
    int mid, 
    __global int* rowinfo, int start){
//__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num , __global unsigned long* bitmap, __global float* prold, __global float* prnew, int midrow){
  int tid=get_global_id(0);
//  tid+=mid;
  int getnumsum=get_global_size(0);

    int startRow=rowinfo[start];
    int endRow=rowinfo[start+1];
    tid+=startRow;


  for(;tid<endRow; tid+=getnumsum){
  //for(;tid<rownum; tid+=getnumsum){
    if(vplist[tid]!=MY_INFINITY)
      continue;
    int vid=tid;
    int start=rowptr[vid];
    int end=rowptr[vid+1];
    int local_rand=randlist[vid];
    char found_larger=0;
    for(int i=start; i<end; i++){
      int dest=colid[i];
     // if(vplist[dest]<color)
      if((vplist[dest]<color)&&vplist[dest]>=0)
        continue;
      if(  (randlist[dest]>local_rand)  ||
          (   (randlist[dest]==local_rand) && (dest<vid)) ){
        found_larger=1;
        break;
      }
    }
    if(found_larger==0)
      vplist[vid]=color;
    else
      *over=0;
      
   
  }

}

