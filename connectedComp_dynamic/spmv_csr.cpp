#include "spmv_util.h"
#define MY_INFINITY    0xffffff00
#define TOTALNUMber 100
int cpuoffset;

#include <pthread.h>
int rowInfo[11];
#define MAX 10

pthread_t thread[2];
pthread_mutex_t mut;
int number1=0;
int number2=0;
cl_int errorCode ;
cl_kernel kernel_hooking;
cl_kernel kernel_hooking_cpu;
int iter;
 cl_command_queue cmdQueue ;
 cl_command_queue cmdQueue_cpu ;
cl_uint work_dim;
        cl_uint work_dimcpu;
size_t globalsizecpu;
size_t localsizecpu;
size_t globalsize[2] ;
size_t blocksize[2] ;
void *thread1(void *)
{
  int i;
  //printf ("thread1 : I'm thread 1\n");

  for (i = 0; i < MAX; i++)
  {
    //printf("thread1 : number1 = %d\n",number1);
    pthread_mutex_lock(&mut);
    if(number2>number1+1){
     number1++;

	    errorCode = clSetKernelArg(kernel_hooking, 5, sizeof(int), &iter); CHECKERROR;
        errorCode = clSetKernelArg(kernel_hooking, 12, sizeof(int), &number1); CHECKERROR;
//	    errorCode = clSetKernelArg(kernel_hooking_cpu, 5, sizeof(int), &iter); CHECKERROR;
            errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_hooking, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
 //           errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, kernel_hooking_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;



  //    errorCode = clSetKernelArg(csrKernel, 9, sizeof(int), &number1); CHECKERROR;
   //   errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
    pthread_mutex_unlock(&mut);
      clFinish(cmdQueue);
    }
    else{
      pthread_mutex_unlock(&mut);
      pthread_exit(NULL);
    }


 //   sleep(1);
  }


  //printf("thread1 :主函数在等我完成任务吗？\n");
  pthread_exit(NULL);
//return NULL;
}
void *thread2(void *)
{
  int i;
//  printf("thread2 : I'm thread 2\n");

  for (i = 0; i < MAX; i++)
  {
 //   printf("thread2 : number2 = %d\n",number2);
    pthread_mutex_lock(&mut);
    if(number2>number1+1){
      number2--;

//	    errorCode = clSetKernelArg(kernel_hooking, 5, sizeof(int), &iter); CHECKERROR;
	    errorCode = clSetKernelArg(kernel_hooking_cpu, 5, sizeof(int), &iter); CHECKERROR;
        errorCode = clSetKernelArg(kernel_hooking_cpu, 12, sizeof(int), &number2); CHECKERROR;
 //           errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_hooking, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, kernel_hooking_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;


 //       errorCode = clSetKernelArg(csrKernelcpu, 11, sizeof(int), &number2); CHECKERROR;
//      errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
    pthread_mutex_unlock(&mut);
      clFinish(cmdQueue_cpu);
    }
    else{
      pthread_mutex_unlock(&mut);
      pthread_exit(NULL);
    }
//    sleep(1);
  }


  //printf("thread2 :主函数在等我完成任务吗？\n");
  pthread_exit(NULL);
//return NULL;
}
void thread_create(void)
{
  int temp;
    number1=0; number2=9;
    number1--;
    number2++;
  memset(&thread, 0, sizeof(thread));          //comment1
  /*创建线程*/
  if((temp = pthread_create(&thread[0], NULL, thread1, NULL)) != 0)  //comment2     
    printf("线程1创建失败!\n");
  //else
   // printf("线程1被创建\n");

  if((temp = pthread_create(&thread[1], NULL, thread2, NULL)) != 0)  //comment3
    printf("线程2创建失败");
//  else
 //   printf("线程2被创建\n");
}
void thread_wait(void)
{
        /*等待线程结束*/
        if(thread[0] !=0)
           {             //comment4    
                pthread_join(thread[0],NULL);
  //              printf("线程1已经结束\n");
          }   
        if(thread[1] !=0) 
           {   
                //comment5
               pthread_join(thread[1],NULL);
   //             printf("线程2已经结束\n");
         }   
}




    //spmv_csr_vector_ocl(&csrmat, parents, shadow, mask, edge_src, 0,  opttime1, optmethod1, clfilename, ntimes);//zf method1
void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, int* parents, int* shadow, char* mask, int* edge_src, int padNum, double& opttime, int& optmethod, char* oclfilename, int ntimes, int* rescoo)
//void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, int* vplist, int* randlist, int padNum, double& opttime, int& optmethod, char* oclfilename, int* vplistres, int ntimes)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cmdQueue = NULL;
    cmdQueue_cpu = NULL;
//    cl_command_queue cmdQueue = NULL;
 //   cl_command_queue cmdQueue_cpu = NULL;


    cl_program program = NULL;

    assert(initialization2(devices, &context, &cmdQueue, &program, oclfilename, &cmdQueue_cpu) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devRowPtr;//r
    cl_mem devColId;//r
    cl_mem devparents;
    cl_mem devshadow;//w
    cl_mem devmask;//w
    cl_mem devedge_src;//r

    cl_mem devRowPtr_cpu;
    cl_mem devColId_cpu;
    cl_mem devedge_src_cpu;//r

    cl_mem devover;

    int vertex_cnt=mat->matinfo.height;
    int edge_cnt=mat->matinfo.nnz;

    //int* rescoo=(int*)malloc(sizeof(int)*vertex_cnt);
    int* rescoo2=(int*)malloc(sizeof(int)*vertex_cnt);

    //Initialize values
    int nnz = mat->matinfo.nnz;
    int vecsize = mat->matinfo.width;
    int rownum = mat->matinfo.height;
    int rowptrsize = rownum + 1;
    ALLOCATE_GPU_READ(devRowPtr, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ(devColId, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ(devedge_src, edge_src, sizeof(int)*edge_cnt);

    
    /*
    for(int i=0; i<15; i++)
      printf("%d   ",mat->csr_row_ptr[i]);
    printf("\n\n");
    for(int i=0; i<15; i++)
      printf("%d   ",mat->csr_col_id[i]);
    printf("\n\n");
    for(int i=0; i<15; i++)
      printf("%d   ",edge_src[i]);
    printf("\n\n");
    for(int i=1988; i<2100; i++)
      printf("%d   ",mask[i]);
    printf("\n\n");
    */
    

    ALLOCATE_GPU_READ_cpu(devparents, parents, sizeof(int)*vertex_cnt);
    ALLOCATE_GPU_READ_cpu(devshadow, shadow, sizeof(int)*vertex_cnt);
    ALLOCATE_GPU_READ_cpu(devmask, mask, sizeof(char)*edge_cnt);

//    devvplist= clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(int)*rownum, NULL, &errorCode); CHECKERROR;
 //   errorCode = clEnqueueWriteBuffer(cmdQueue, devvplist, CL_TRUE, 0, sizeof(int)*rownum, vplist, 0, NULL, NULL); CHECKERROR;


    ALLOCATE_GPU_READ_cpu(devRowPtr_cpu, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ_cpu(devColId_cpu, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ_cpu(devedge_src_cpu, edge_src, sizeof(int)*edge_cnt);
    devover= clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(char), NULL, &errorCode); CHECKERROR;

    opttime = 10000.0f;
    optmethod = 0;
    int dim2 = 1;


    {
	int methodid = 7;
        
//////////////////////////////////////////////////////////////////////////////////////
/*
        printf("sizeofulong=%d bytes",sizeof(unsigned long));
        unsigned long *bitmap=(unsigned long*)malloc(sizeof(unsigned long)*(((padrowsize)>>6) + 1));
	int* rowforcpu = (int*)malloc(sizeof(int)*(padrowsize));
        int rowforcpusum=0;
        memset(bitmap,0,(sizeof(unsigned long)*(((padrowsize)>>6) + 1)));

        for(int i=0 ; i<padrowsize; i++){
          int numi=rowptrpad[i+1]-rowptrpad[i];
          if(numi>=64){
          //if(numi>=WARPSIZE){
            bitmap[(i>>6)]=bitmap[(i>>6)]|(1ul<<((i)&0x3f));
          }
          else if(numi!=0) {
            rowforcpu[rowforcpusum]=i;
            rowforcpusum++;
          }
        }

	cl_mem devbitmap;
	ALLOCATE_GPU_READ(devbitmap, bitmap, sizeof(unsigned long)*(((padrowsize)>>6)+1));

	cl_mem devrowforcpu;
	ALLOCATE_GPU_READ_cpu(devrowforcpu, rowforcpu, sizeof(int)*(rowforcpusum));
  */     




//////////////////////////////////////////////////////////////////////////////////////




	//printf("\nRow Num %d padded size %d\n", rownum, padrowsize);
	work_dim = 1;
	//cl_uint work_dim = 1;
	//cl_uint work_dim = 2;
	//int dim2 = 16;
	blocksize[0] = CSR_VEC_GROUP_SIZE;
	blocksize[1] = 1;
	//size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};
        //int threadsinrow=16;

	kernel_hooking= NULL;
	//cl_kernel kernel_hooking= NULL;
	kernel_hooking = clCreateKernel(program, "kernel_hooking", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking, 0, sizeof(cl_mem), &devparents); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking, 1, sizeof(cl_mem), &devshadow); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking, 2, sizeof(cl_mem), &devmask); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking, 3, sizeof(cl_mem), &devedge_src); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking, 4, sizeof(cl_mem), &devover); CHECKERROR;
//	errorCode = clSetKernelArg(kernel_hooking, 5, sizeof(int), &iter); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking, 6, sizeof(int), &edge_cnt); CHECKERROR;

	errorCode = clSetKernelArg(kernel_hooking, 7, sizeof(cl_mem), &devRowPtr); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking, 8, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking, 9, sizeof(int), &vertex_cnt); CHECKERROR;//zf

        int rowsetzf=(double)cpuoffset/100*vertex_cnt;
        printf("rowsetzf=%d  cpuoffset=%d%%\n",rowsetzf,cpuoffset);
	errorCode = clSetKernelArg(kernel_hooking, 10, sizeof(int), &rowsetzf); CHECKERROR;//zf

        for(int i=0; i<11; i++){
          rowInfo[i]=(double)i/10*rownum;
         // printf("row %d = %d\n",i,rowInfo[i]);
        }
        //printf("rownum=%d\n",rownum);exit(0);
	cl_mem devrowInfo;
	ALLOCATE_GPU_READ(devrowInfo, rowInfo, sizeof(int)*(11));
        errorCode = clSetKernelArg(kernel_hooking, 11, sizeof(cl_mem), &devrowInfo); CHECKERROR;
        errorCode = clSetKernelArg(kernel_hooking, 12, sizeof(int), &number1); CHECKERROR;





        globalsizecpu=3;
        localsizecpu=1;
        //size_t globalsizecpu=3;
        //size_t localsizecpu=1;

	kernel_hooking_cpu= NULL;
	//cl_kernel kernel_hooking_cpu= NULL;
	kernel_hooking_cpu = clCreateKernel(program, "kernel_hooking_cpu", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking_cpu, 0, sizeof(cl_mem), &devparents); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking_cpu, 1, sizeof(cl_mem), &devshadow); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking_cpu, 2, sizeof(cl_mem), &devmask); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking_cpu, 3, sizeof(cl_mem), &devedge_src_cpu); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking_cpu, 4, sizeof(cl_mem), &devover); CHECKERROR;
//	errorCode = clSetKernelArg(kernel_hooking_cpu, 5, sizeof(int), &iter); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking_cpu, 6, sizeof(int), &edge_cnt); CHECKERROR;

	errorCode = clSetKernelArg(kernel_hooking_cpu, 7, sizeof(cl_mem), &devRowPtr_cpu); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking_cpu, 8, sizeof(cl_mem), &devColId_cpu); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking_cpu, 9, sizeof(int), &vertex_cnt); CHECKERROR;//zf

	errorCode = clSetKernelArg(kernel_hooking_cpu, 10, sizeof(int), &rowsetzf); CHECKERROR;//zf
        errorCode = clSetKernelArg(kernel_hooking_cpu, 11, sizeof(cl_mem), &devrowInfo); CHECKERROR;
        errorCode = clSetKernelArg(kernel_hooking_cpu, 12, sizeof(int), &number2); CHECKERROR;






	cl_kernel kernel_update= NULL;
	kernel_update = clCreateKernel(program, "kernel_update", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(kernel_update, 0, sizeof(cl_mem), &devparents); CHECKERROR;
	errorCode = clSetKernelArg(kernel_update, 1, sizeof(cl_mem), &devshadow); CHECKERROR;
	errorCode = clSetKernelArg(kernel_update, 2, sizeof(int), &vertex_cnt); CHECKERROR;
	
	cl_kernel kernel_pointer_jumping= NULL;
	kernel_pointer_jumping = clCreateKernel(program, "kernel_pointer_jumping", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(kernel_pointer_jumping, 0, sizeof(cl_mem), &devparents); CHECKERROR;
	errorCode = clSetKernelArg(kernel_pointer_jumping, 1, sizeof(cl_mem), &devshadow); CHECKERROR;
	errorCode = clSetKernelArg(kernel_pointer_jumping, 2, sizeof(int), &vertex_cnt); CHECKERROR;



	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;
        work_dimcpu=1;
        //cl_uint work_dimcpu=1;

        //for (int groupnum = 48; groupnum <= 288; groupnum+= 24)
        for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
        {
          globalsize[0] = groupnum*CSR_VEC_GROUP_SIZE;
          globalsize[1] =  dim2;
          //size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};
    errorCode = clEnqueueWriteBuffer(cmdQueue, devparents, CL_TRUE, 0, sizeof(int)*vertex_cnt, parents, 0, NULL, NULL); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devshadow, CL_TRUE, 0, sizeof(int)*vertex_cnt, shadow, 0, NULL, NULL); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devmask, CL_TRUE, 0, sizeof(char)*edge_cnt, mask, 0, NULL, NULL); CHECKERROR;
    clFinish(cmdQueue);
    
         // errorCode = clEnqueueWriteBuffer(cmdQueue, devvplist, CL_TRUE, 0, sizeof(float)*rownum, vplist, 0, NULL, NULL); CHECKERROR;
          //clFinish(cmdQueue);

          char stop=0;
          iter=0;
          //int iter=0;

          do{
            stop=1;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);

thread_create();
thread_wait();


            /*
	    errorCode = clSetKernelArg(kernel_hooking, 5, sizeof(int), &iter); CHECKERROR;
	    errorCode = clSetKernelArg(kernel_hooking_cpu, 5, sizeof(int), &iter); CHECKERROR;
            errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_hooking, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, kernel_hooking_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);
            clFinish(cmdQueue_cpu);
            */

            errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueReadBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;

            iter++;
            
            errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_pointer_jumping, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

//          }while(iter==1);
          }while(!stop);




          int* tmpresult = (int*)malloc(sizeof(int)*vertex_cnt);
          errorCode = clEnqueueReadBuffer(cmdQueue, devparents, CL_TRUE, 0, sizeof(int)*vertex_cnt, tmpresult, 0, NULL, NULL); CHECKERROR;
          clFinish(cmdQueue);

          memset(rescoo2,0,sizeof(int)*vertex_cnt);
          for(int i=0; i<vertex_cnt; i++){
            if(tmpresult[i]>vertex_cnt){
              printf("tmpresult[%d]=%d\n",i,tmpresult[i]);
              exit(-1);
            }
            rescoo2[tmpresult[i]]=1;
          }
          int sumcomp=0;
          for(int i=0; i<vertex_cnt; i++){
            sumcomp+=rescoo2[i];
          }
          printf("\n\nThe number of components= %d iters= %d\n",sumcomp,iter);

          
          /*
          //if(groupnum==48)
          if(groupnum==24)
            for(int i=0; i<vertex_cnt; i++)
              rescoo2[i]=tmpresult[i];
          */

          //
          two_vec_compare(rescoo, tmpresult, vertex_cnt);
              
          free(tmpresult);
 /*
        }
    }
}
*/

//#if 0
         

          /*
          for (int k = 0; k < 3; k++)
          {


            errorCode = clEnqueueWriteBuffer(cmdQueue, devparents, CL_TRUE, 0, sizeof(int)*vertex_cnt, parents, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devshadow, CL_TRUE, 0, sizeof(int)*vertex_cnt, shadow, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devmask, CL_TRUE, 0, sizeof(char)*edge_cnt, mask, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);

            char stop=0;
            int iter=0;

            do{
              stop=1;
              errorCode = clSetKernelArg(kernel_hooking, 5, sizeof(int), &iter); CHECKERROR;
              errorCode = clSetKernelArg(kernel_hooking_cpu, 5, sizeof(int), &iter); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;
              clFinish(cmdQueue);

              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_hooking, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, kernel_hooking_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
              clFinish(cmdQueue);
              clFinish(cmdQueue_cpu);

              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueReadBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;

              iter++;

              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_pointer_jumping, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

            }while(!stop);



          }
*/

          double time_in_sec=0;
          for (int i = 0; i < ntimes; i++)
          {



            errorCode = clEnqueueWriteBuffer(cmdQueue, devparents, CL_TRUE, 0, sizeof(int)*vertex_cnt, parents, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devshadow, CL_TRUE, 0, sizeof(int)*vertex_cnt, shadow, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devmask, CL_TRUE, 0, sizeof(char)*edge_cnt, mask, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);

            char stop=0;
            int iter=0;


          double teststart = timestamp();
            do{
              stop=1;

              errorCode = clEnqueueWriteBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;
              clFinish(cmdQueue);



thread_create();
thread_wait();


/*
              errorCode = clSetKernelArg(kernel_hooking, 5, sizeof(int), &iter); CHECKERROR;
              errorCode = clSetKernelArg(kernel_hooking_cpu, 5, sizeof(int), &iter); CHECKERROR;
              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_hooking, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, kernel_hooking_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
              clFinish(cmdQueue);
              clFinish(cmdQueue_cpu);
              */

              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueReadBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;

              iter++;

              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_pointer_jumping, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

            }while(!stop);


         double testend = timestamp();
          time_in_sec += (testend - teststart);

          }


          //double time_in_sec = (testend - teststart)/(double)dim2;
          double gflops = (double)nnz*2*iter/(time_in_sec/(double)ntimes)/(double)1e9;
          printf("\nCSR vector SLM row ptr groupnum:%d cpu time %lf ms GFLOPS %lf code %d \n\n", groupnum,   time_in_sec / (double) ntimes * 1000, gflops, methodid);

          double onetime = time_in_sec / (double) ntimes;
          if (onetime < opttime)
          {
            opttime = onetime;
            optmethod = methodid;
          }
          if (onetime < minlooptime)
          {
            minlooptime = onetime;
            maxloopsize = groupnum;
          }
          
        }
        printf("******* Min time %f groupnum %d **********", minlooptime, maxloopsize);

        //if (devRowPtrPad)
         // clReleaseMemObject(devRowPtrPad);
        if (kernel_hooking)
          clReleaseKernel(kernel_hooking);
        if (kernel_hooking_cpu)
          clReleaseKernel(kernel_hooking_cpu);
        if (kernel_update)
          clReleaseKernel(kernel_update);
          if(kernel_pointer_jumping)
          clReleaseKernel(kernel_pointer_jumping);
        //free(rowptrpad);


        //free(bitmap);free(rowforcpu);//zf
        //	clReleaseMemObject(devbitmap);//zf
        }



        if (devRowPtr)
          clReleaseMemObject(devRowPtr);
        if (devColId)
          clReleaseMemObject(devColId);


        freeObjects(devices, &context, &cmdQueue, &program);


        return;
    }


int main(int argc, char* argv[])
{
    if (argc < 2){
	printf("\nUsage: ./main *.mtx percentage(such as 40)\n");
        return 1;
    }

    char* filename = argv[1];
    cpuoffset = 50;//atoi(argv[2]);
//    printf("cpuoffset=%d\n",cpuoffset);
    int ntimes = 1;
    //int ntimes = 20;

    coo_matrix<int, float> mat;
    init_coo_matrix(mat);
    ReadMMF(filename, &mat);


    //char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    
   {
	sprintf(clfilename, "%s%s", ".", "/spmv_csr_vector.cl");
	//sprintf(clfilename, "%s%s", "/home/pacman/zf/wubo/apu_corun/connectedComp_dynamic", "/spmv_csr_vector.cl");
    printMatInfo(&mat);


    if(mat.matinfo.width != mat.matinfo.height){
      printf("width != height\n");
      exit(-1);
    }



    csr_matrix<int, float> csrmat;
    coo2csr<int, float>(&mat, &csrmat);


////////////////////////////////////////////////////////////////////////////////////////////////////////

    int vertex_cnt=mat.matinfo.height;
    int edge_cnt=csrmat.matinfo.nnz;
    int* parents=(int*)malloc(sizeof(int)*vertex_cnt);
    int* shadow=(int*)malloc(sizeof(int)*vertex_cnt);
    char* mask=(char*)malloc(sizeof(char)*edge_cnt);
    int* edge_src=(int*)malloc(sizeof(int)*edge_cnt);
//    char* over=(char*)malloc(sizeof(char));

    for(int tid=0; tid<vertex_cnt; tid++){
      parents[tid]=tid;
      shadow[tid]=tid;
      int start=csrmat.csr_row_ptr[tid];
      int end=csrmat.csr_row_ptr[tid+1];
      for(int i=start; i<end; i++){
        mask[i]=0;
        edge_src[i]=tid;
      }
    }


    int curr=0, color;
    char stop=0;
    do{
      stop=1;
      color=curr;

      for(int tid=0; tid<vertex_cnt; tid++){
        int start=csrmat.csr_row_ptr[tid];
        int end=csrmat.csr_row_ptr[tid+1];
        for(int eid=start; eid<end; eid++){
          if(mask[eid])continue;
          int src=tid;
          int dest=csrmat.csr_col_id[eid];
          if(parents[src]!=parents[dest]){
            if(parents[src]>parents[dest])
              shadow[tid]=parents[dest];
            else
              mask[eid]=true;
            stop=false;
          }
          else
            mask[eid]=true;
          }
        }

        for(int tid=0; tid<vertex_cnt; tid++){
          parents[tid]=shadow[tid];
        }

        for(int tid=0; tid<vertex_cnt; tid++){
          int tmp=parents[tid];
          while(parents[tmp]!=tmp)
            tmp=parents[tmp];
          shadow[tid]=tmp;
        }

        for(int tid=0; tid<vertex_cnt; tid++){
            parents[tid]=shadow[tid];
        }
        curr++;
      }while(!stop);


    int* rescoo=(int*)malloc(sizeof(int)*vertex_cnt);
          memset(rescoo,0,sizeof(int)*vertex_cnt);
          for(int i=0; i<vertex_cnt; i++){
            rescoo[parents[i]]=1;
          }
          int sumcomp=0;
          for(int i=0; i<vertex_cnt; i++){
            sumcomp+=rescoo[i];
          }
          printf("\n\nCPU RESULT: The number of components= %d iters= %d\n",sumcomp,curr);
          for(int i=0; i<vertex_cnt; i++){
            rescoo[i]=parents[i];
          }
       
//initialzed
    for(int tid=0; tid<vertex_cnt; tid++){
      parents[tid]=tid;
      shadow[tid]=tid;
      int start=csrmat.csr_row_ptr[tid];
      int end=csrmat.csr_row_ptr[tid+1];
      for(int i=start; i<end; i++){
        mask[i]=0;
        edge_src[i]=tid;
      }
    }





////////////////////////////////////////////////////////////////////////////////////////////////////////


    double opttime1 = 10000.0f;
    int optmethod1 = 0;




    spmv_csr_vector_ocl(&csrmat, parents, shadow, mask, edge_src, 0,  opttime1, optmethod1, clfilename, ntimes, rescoo);//zf method1

	double opttime2 = 10000.0f;
	int optmethod2 = 0;

	csr_matrix<int, float> padcsr;
	pad_csr(&csrmat, &padcsr, WARPSIZE / 2);
	printf("\nNNZ Before %d After %d\n", csrmat.matinfo.nnz, padcsr.matinfo.nnz);
//    spmv_csr_vector_ocl(&padcsr, parents, shadow, mask, edge_src, 0,  opttime2, optmethod2, clfilename, ntimes);//zf method1
//        spmv_csr_vector_ocl(&padcsr, vplistori, randlist, 0,  opttime2, optmethod2, clfilename, vplist, ntimes);//zf method1
//	spmv_csr_vector_ocl(&padcsr, pr0, res, 16, opttime2, optmethod2, clfilename, coores, ntimes);
	free_csr_matrix(padcsr);

	int nnz = mat.matinfo.nnz;
	double gflops = (double)nnz*2*TOTALNUMber/opttime1/(double)1e9;
	printf("\n------------------------------------------------------------------------\n");
	printf("CSR VEC without padding best time %f ms best method %d gflops %f", opttime1*1000.0, optmethod1, gflops);
//	printf("\n------------------------------------------------------------------------\n");
//	gflops = (double)nnz*2*TOTALNUMber/opttime2/(double)1e9;
//	printf("CSR VEC with padding best time %f ms best method %d gflops %f", opttime2*1000.0, optmethod2, gflops);
	printf("\n------------------------------------------------------------------------\n");
        printf("CAUTTION: kernel time(ms): %f\n",opttime1*1000.0);
  

    //free(vec);
    //free(res);
    free_csr_matrix(csrmat);
    //free(vplist);
    //free(vplistori);
    //free(randlist);

    free(parents);free(shadow);free(mask);free(edge_src);

    }

    free_coo_matrix(mat);

    return 0;
}

