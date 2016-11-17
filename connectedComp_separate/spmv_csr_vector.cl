
#define VEC2DWIDTH 1024
#define CSR_VEC_GROUP_SIZE 64
//#define WARPSIZE 64
#define WARPSIZE 32
#define DAM 0.85

#define MY_INFINITY    0xffffff00
#define CHUNK_SZ 2



// checking all edges in parallel
__kernel void kernel_hooking(
        __global int * d_parents, 
        __global int * d_shadow, 
        __global char* d_mask, 
        __global int * d_edge_src, 
        __global char* d_over,
        int iter,
        int edge_cnt,

        __global int* rowptr,  
        __global int* colid,
        int vertex_cnt,

        __global  unsigned long* bitmap){
        //int mid){
        //cudaGraph graph){
  // int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid=get_global_id(0);
  int getnumsum=get_global_size(0);//zf

    for(; tid<vertex_cnt; tid+=getnumsum){//zf
      if(!(bitmap[tid>>6]&(1ul<<(tid&0x3f))))
        continue;

    if (tid >= vertex_cnt) return;
    int start = rowptr[tid];
    int end = rowptr[tid+1];
    for(int eid=start; eid<end; eid++){
      if (d_mask[eid]) continue;
      int src = tid;
      //int src = d_edge_src[eid];
      int dest = colid[eid];
      if (d_parents[src] != d_parents[dest]){
        int min, max;
        //int mn, mx;
        if (d_parents[src]>d_parents[dest])
        {
          // max = d_parents[src];
          //min = d_parents[dest];
          d_shadow[tid] = d_parents[dest];
        }
        else
        {
          //max = d_parents[dest];
          //min = d_parents[src];
          d_mask[eid] = true;//zf
        }
        *d_over = false;

      }
      else
      {
        d_mask[eid] = true;
      }

    }

  }
  }


// checking all edges in parallel
__kernel void kernel_hooking_cpu(
        __global int * d_parents, 
        __global int * d_shadow, 
        __global char* d_mask, 
        __global int * d_edge_src, 
        __global char* d_over,
        int iter,
        int edge_cnt,

        __global int* rowptr,  
        __global int* colid,
        int vertex_cnt,

        __global int* rowall,
        int rowallsize)
        //cudaGraph graph)
{
   // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int t=get_global_id(0);
    int getnumsum=get_global_size(0);//zf

    for(; t<rowallsize; t+=getnumsum){//zf
      int tid=rowall[t];

      if (tid >= vertex_cnt) return;
      int start = rowptr[tid];
      int end = rowptr[tid+1];
      for(int eid=start; eid<end; eid++){
        if (d_mask[eid]) continue;
        int src = tid;
        //int src = d_edge_src[eid];
        int dest = colid[eid];
        if (d_parents[src] != d_parents[dest]){
            int min, max;
            //int mn, mx;
            if (d_parents[src]>d_parents[dest])
            {
               // max = d_parents[src];
                //min = d_parents[dest];
                d_shadow[tid] = d_parents[dest];
            }
            else
            {
                //max = d_parents[dest];
                //min = d_parents[src];
                d_mask[eid] = true;//zf
            }
            *d_over = false;

        }
        else
        {
            d_mask[eid] = true;
        }
 
      }

    }







/*




    for(; tid<((edge_cnt+1)/CHUNK_SZ); tid+=getnumsum){//zf
//      if(tid==0)        printf("hah=%d  edge=%d  getnum=%d\n",((edge_cnt+1)/CHUNK_SZ),edge_cnt,getnumsum);


    int chunk_id = tid * CHUNK_SZ;

    if (chunk_id >= edge_cnt) return;
   
    int end_id = chunk_id + CHUNK_SZ;
    if (end_id > edge_cnt) 
       end_id = edge_cnt; 
    for (int eid=chunk_id;eid<end_id;eid++)
    {
        if (d_mask[eid]) continue;

        int src = d_edge_src[eid];
        int dest = colid[eid];
        //int dest = graph.get_edge_dest(eid);

        if (d_parents[src] != d_parents[dest])
        {
            int min, max;
            //int mn, mx;
            if (d_parents[src]>d_parents[dest])
            {
                max = d_parents[src];
                min = d_parents[dest];
                //mx = src;
                //mn = dest;
            }
            else
            {
                max = d_parents[dest];
                min = d_parents[src];
                //mx = dest;
                //mn = src;
            }

            if ((iter%2)==0) // even iterations
            {
                d_shadow[min] = max;
           //     if(min==0)printf("max=%d\n",max);
                //d_parents[mn] = max;
            }
            
            else // odd iterations
            {
                d_shadow[max] = min;
                //d_parents[mx] = min;
            }
           
            *d_over = false;
        }
        else
        {
            d_mask[eid] = true;
        }
    }
    }
    */
}

__kernel void kernel_update(
        __global int * d_parents, 
        __global int * d_shadow, 
        int vertex_cnt) 
{
//	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid=get_global_id(0);
    int getnumsum=get_global_size(0);//zf

    for(; tid<vertex_cnt; tid+=getnumsum){//zf

    if (tid >= vertex_cnt) return;
    d_parents[tid] = d_shadow[tid];

    }
}

__kernel void kernel_pointer_jumping(
        __global int * d_parents, 
        __global int * d_shadow, 
        int vertex_cnt)
{
//	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid=get_global_id(0);

    int getnumsum=get_global_size(0);//zf
    for(; tid<vertex_cnt; tid+=getnumsum){//zf


    if (tid >= vertex_cnt) return;

    int parent = d_parents[tid];
    while(d_parents[parent]!=parent)
    {
        parent = d_parents[parent];
    }

    d_shadow[tid] = parent; 

    }
}























/*
__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  
    __global int* colid, 
    __global int* randlist,
    __global int* vplist,
    int color,
    __global char* over,
    int rownum,
    int mid){
//__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num , __global unsigned long* bitmap, __global float* prold, __global float* prnew, int midrow){
  int tid=get_global_id(0);
  int getnumsum=get_global_size(0);

  for(;tid<mid; tid+=getnumsum){
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
    int mid){
//__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num , __global unsigned long* bitmap, __global float* prold, __global float* prnew, int midrow){
  int tid=get_global_id(0);
  tid+=mid;
  int getnumsum=get_global_size(0);

  for(;tid<rownum; tid+=getnumsum){
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
*/
