#include "spmv_util.h"
#define TOTALNUMber 5
int cpuoffset;

#include <pthread.h>
int rowInfo[11];
#define MAX 10

pthread_t thread[2];
pthread_mutex_t mut;
int number1=0;
int number2=0;
cl_int errorCode ;
cl_kernel csrKernel ;
cl_kernel csrKernelcpu ;
cl_command_queue cmdQueue ;
cl_command_queue cmdQueue_cpu ;
cl_uint work_dim ;
cl_uint work_dimcpu;
size_t globalsizecpu;
size_t globalsize[2] ;
		size_t blocksize[2];
        size_t localsizecpu;
     cl_mem devprnew;
    cl_mem devprold;


 
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

              errorCode = clSetKernelArg(csrKernel, 7, sizeof(cl_mem), &devprold); CHECKERROR;//added
              errorCode = clSetKernelArg(csrKernel, 8, sizeof(cl_mem), &devprnew); CHECKERROR;//added
        errorCode = clSetKernelArg(csrKernel, 11, sizeof(int), &number1); CHECKERROR;


              errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;





//      errorCode = clSetKernelArg(csrKernel, 9, sizeof(int), &number1); CHECKERROR;
 //     errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
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

              errorCode = clSetKernelArg(csrKernelcpu, 8, sizeof(cl_mem), &devprold); CHECKERROR;//added
              errorCode = clSetKernelArg(csrKernelcpu, 9, sizeof(cl_mem), &devprnew); CHECKERROR;//added
        errorCode = clSetKernelArg(csrKernelcpu, 12, sizeof(int), &number2); CHECKERROR;


              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
 



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


void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, float* vec, float* result, int padNum, double& opttime, int& optmethod, char* oclfilename, float* coores, int ntimes)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cmdQueue = NULL;
    cmdQueue_cpu = NULL;
    //cl_command_queue cmdQueue = NULL;
    //cl_command_queue cmdQueue_cpu = NULL;
    cl_program program = NULL;

    assert(initialization2(devices, &context, &cmdQueue, &program, oclfilename, &cmdQueue_cpu) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devRowPtr;
    cl_mem devColId;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devRowPtr_cpu;
    cl_mem devColId_cpu;
    cl_mem devData_cpu;
    cl_mem devVec_cpu;


    cl_mem devRes;

    //Initialize values
    int nnz = mat->matinfo.nnz;
    int vecsize = mat->matinfo.width;
    int rownum = mat->matinfo.height;
    int rowptrsize = rownum + 1;
    ALLOCATE_GPU_READ(devRowPtr, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ(devColId, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ(devData, mat->csr_data, sizeof(float)*nnz);
    ALLOCATE_GPU_READ(devVec, vec, sizeof(float)*vecsize);
    //ALLOCATE_GPU_READ(devprnew, vec, sizeof(float)*vecsize);
    //ALLOCATE_GPU_READ(devprold, vec, sizeof(float)*vecsize);
    devprold= clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devprold, CL_TRUE, 0, sizeof(float)*rownum, vec, 0, NULL, NULL); CHECKERROR;
    devprnew= clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devprnew, CL_TRUE, 0, sizeof(float)*rownum, vec, 0, NULL, NULL); CHECKERROR;

    ALLOCATE_GPU_READ_cpu(devRowPtr_cpu, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ_cpu(devColId_cpu, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ_cpu(devData_cpu, mat->csr_data, sizeof(float)*nnz);
    ALLOCATE_GPU_READ_cpu(devVec_cpu, vec, sizeof(float)*vecsize);

   
    int paddedres = findPaddedSize(rownum, 16);
    devRes = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;
    //errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;

	const cl_image_format floatFormat =
	{
	    CL_R,
	    CL_FLOAT,
	};
		
	int width = VEC2DWIDTH;
	int height = (vecsize + VEC2DWIDTH - 1)/VEC2DWIDTH;
/*
	float* image2dVec = (float*)malloc(sizeof(float)*width*height);
	memset(image2dVec, 0, sizeof(float)*width*height);
	for (int i = 0; i < vecsize; i++)
	{
	    image2dVec[i] = vec[i];
	}
*/
	size_t origin[] = {0, 0, 0};
	size_t vectorSize[] = {width, height, 1};


    opttime = 10000.0f;
    optmethod = 0;
    int dim2 = 1;


    {
	int methodid = 7;
	cl_mem devRowPtrPad;
	cl_mem devRowPtrPad_cpu;
	int padrowsize = findPaddedSize(rownum, 8);
	int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	for (int i = 0; i <= mat->matinfo.height; i++)
	    rowptrpad[i] = mat->csr_row_ptr[i];


	ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));
	ALLOCATE_GPU_READ_cpu(devRowPtrPad_cpu, rowptrpad, sizeof(int)*(padrowsize+1));
	clFinish(cmdQueue);

        cl_mem devdistance;
        float distance=0;
//        ALLOCATE_GPU_READ(devdistance, &distance, sizeof(float));
    devdistance= clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(float), NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devdistance, CL_TRUE, 0, sizeof(float), &distance, 0, NULL, NULL); CHECKERROR;


        clFinish(cmdQueue);

       
//////////////////////////////////////////////////////////////////////////////////////

        printf("sizeofulong=%d bytes",sizeof(unsigned long));
        unsigned long *bitmap=(unsigned long*)malloc(sizeof(unsigned long)*(((padrowsize)>>6) + 1));
	int* rowforcpu = (int*)malloc(sizeof(int)*(padrowsize));
        int rowforcpusum=0;
        memset(bitmap,0,(sizeof(unsigned long)*(((padrowsize)>>6) + 1)));

        for(int i=0 ; i<padrowsize; i++){
          int numi=rowptrpad[i+1]-rowptrpad[i];
          //if(numi>=16){
          //if(numi>=0){
          if(numi>=64){
          //if(numi>=WARPSIZE){
            bitmap[(i>>6)]=bitmap[(i>>6)]|(1ul<<((i)&0x3f));
          }
          else if(numi!=0) {
            rowforcpu[rowforcpusum]=i;
            rowforcpusum++;
//            printf("%d, ",i);
          }
//          if((i==275)||(i==688))
 //           printf("num%d=%d\n",i,numi);
        }
        /*
        printf("\nrowforcpusum=%d\n",rowforcpusum);
        int row=275;
        printf("(bitmap[row>>6]&(1ul<<(row&0x3f)))=%lu\n",(bitmap[row>>6]&(1ul<<(row&0x3f))));
        printf("((1ul<<(row&0x3f)))=%x, %d\n",(bitmap[row>>6]&(1ul<<(row&0x3f))),(bitmap[row>>6]&(1ul<<(row&0x3f))));
        */
//        exit(-1);
 
	cl_mem devbitmap;
	ALLOCATE_GPU_READ(devbitmap, bitmap, sizeof(unsigned long)*(((padrowsize)>>6)+1));

	cl_mem devrowforcpu;
	ALLOCATE_GPU_READ_cpu(devrowforcpu, rowforcpu, sizeof(int)*(rowforcpusum));
       




//////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////



	printf("\nRow Num %d padded size %d\n", rownum, padrowsize);
	work_dim = 1;
	//cl_uint work_dim = 1;
	//cl_uint work_dim = 2;
	//int dim2 = 16;
	blocksize[0] = CSR_VEC_GROUP_SIZE;
	blocksize[1] = 1;
	//size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};
        //int threadsinrow=16;

	csrKernel = NULL;
	//cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_ve_slm_pm_fs", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devbitmap); CHECKERROR;


        errorCode = clSetKernelArg(csrKernel, 7, sizeof(cl_mem), &devprold); CHECKERROR;//added
        errorCode = clSetKernelArg(csrKernel, 8, sizeof(cl_mem), &devprnew); CHECKERROR;//added

        int rowsetzf=(double)cpuoffset/100*rownum;//zf
        printf("rowsetzf=%d\n",rowsetzf);
	errorCode = clSetKernelArg(csrKernel, 9, sizeof(int), &rowsetzf); CHECKERROR;


        for(int i=0; i<11; i++){
          rowInfo[i]=(double)i/10*rownum;
         // printf("row %d = %d\n",i,rowInfo[i]);
        }
        //printf("rownum=%d\n",rownum);exit(0);
	cl_mem devrowInfo;
	ALLOCATE_GPU_READ(devrowInfo, rowInfo, sizeof(int)*(11));
        errorCode = clSetKernelArg(csrKernel, 10, sizeof(cl_mem), &devrowInfo); CHECKERROR;
        errorCode = clSetKernelArg(csrKernel, 11, sizeof(int), &number1); CHECKERROR;







	csrKernelcpu = NULL;
	//cl_kernel csrKernelcpu = NULL;
        globalsizecpu=3;
        //size_t globalsizecpu=3;
        //size_t globalsizecpu=4;
        //size_t globalsizecpu=rowforcpusum;
        localsizecpu=1;
        //size_t localsizecpu=1;
	csrKernelcpu = clCreateKernel(program, "cpu_csr", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 0, sizeof(cl_mem), &devRowPtrPad_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 1, sizeof(cl_mem), &devColId_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 2, sizeof(cl_mem), &devData_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 3, sizeof(cl_mem), &devVec_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 4, sizeof(cl_mem), &devRes); CHECKERROR;
        
	errorCode = clSetKernelArg(csrKernelcpu, 5, sizeof(int), &rownum); CHECKERROR;
	//wrong !!errorCode = clSetKernelArg(csrKernelcpu, 5, sizeof(int), &rowforcpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 6, sizeof(cl_mem), &devrowforcpu); CHECKERROR;
	
	errorCode = clSetKernelArg(csrKernelcpu, 7, sizeof(int), &rowforcpusum); CHECKERROR;//newadded

        errorCode = clSetKernelArg(csrKernelcpu, 8, sizeof(cl_mem), &devprold); CHECKERROR;//added
        errorCode = clSetKernelArg(csrKernelcpu, 9, sizeof(cl_mem), &devprnew); CHECKERROR;//added

	errorCode = clSetKernelArg(csrKernelcpu, 10, sizeof(int), &rowsetzf); CHECKERROR;

        errorCode = clSetKernelArg(csrKernelcpu, 11, sizeof(cl_mem), &devrowInfo); CHECKERROR;
        errorCode = clSetKernelArg(csrKernelcpu, 12, sizeof(int), &number2); CHECKERROR;


        //new kernel
        cl_kernel disKernel = NULL;
        disKernel = clCreateKernel(program, "caldistance", &errorCode); CHECKERROR;
        errorCode = clSetKernelArg(disKernel, 0, sizeof(int), &rownum); CHECKERROR;
        errorCode = clSetKernelArg(disKernel, 1, sizeof(cl_mem), &devprold); CHECKERROR;
        errorCode = clSetKernelArg(disKernel, 2, sizeof(cl_mem), &devprnew); CHECKERROR;
        errorCode = clSetKernelArg(disKernel, 3, sizeof(cl_mem), &devdistance); CHECKERROR;





	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;
        work_dimcpu=1;
        //cl_uint work_dimcpu=1;
	
	for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
	{
	    //size_t globalsize[] = {totalsize, dim2};
	    globalsize[0] = groupnum*CSR_VEC_GROUP_SIZE;
	    globalsize[1] = dim2;
	    //size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};

	    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devprold, CL_TRUE, 0, sizeof(float)*rownum, vec, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devprnew, CL_TRUE, 0, sizeof(float)*rownum, vec, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);


            for(int order=0; order<TOTALNUMber; order++){
thread_create();
thread_wait();



/*
              errorCode = clSetKernelArg(csrKernel, 7, sizeof(cl_mem), &devprold); CHECKERROR;//added
              errorCode = clSetKernelArg(csrKernel, 8, sizeof(cl_mem), &devprnew); CHECKERROR;//added


              errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

              errorCode = clSetKernelArg(csrKernelcpu, 8, sizeof(cl_mem), &devprold); CHECKERROR;//added
              errorCode = clSetKernelArg(csrKernelcpu, 9, sizeof(cl_mem), &devprnew); CHECKERROR;//added


              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
              */

              clFinish(cmdQueue);
              clFinish(cmdQueue_cpu);

              cl_mem prtmpt=devprold;
              devprold=devprnew;
              devprnew=prtmpt;

            }
              errorCode = clSetKernelArg(disKernel, 1, sizeof(cl_mem), &devprold); CHECKERROR;
              errorCode = clSetKernelArg(disKernel, 2, sizeof(cl_mem), &devprnew); CHECKERROR;

              size_t workdimcpu=1, sizecpu=1;
              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, disKernel , work_dim, NULL, &workdimcpu, &workdimcpu, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueReadBuffer(cmdQueue_cpu, devdistance, CL_TRUE, 0, sizeof(float), &distance, 0, NULL, NULL); CHECKERROR;
              clFinish(cmdQueue_cpu);
              distance=sqrt(distance);
              printf("distance=%f\n",distance);
              //      exit(0);







	    float* tmpresult = (float*)malloc(sizeof(float)*rownum);
	    errorCode = clEnqueueReadBuffer(cmdQueue, devprold, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	    //errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);

            
            /*
        for(int i=0; i<rowforcpusum; i++){//zfadded
        //  printf("rowforcpu[%d]=%d\n",i,rowforcpu[i]);
          tmpresult[rowforcpu[i]]=coores[rowforcpu[i]];
        }
        */
        
        


	    two_vec_compare(coores, tmpresult, rownum);
	    free(tmpresult);

	    for (int k = 0; k < 3; k++)
	    {

              for(int order=0; order<TOTALNUMber; order++){


                errorCode = clSetKernelArg(csrKernel, 7, sizeof(cl_mem), &devprold); CHECKERROR;//added
                errorCode = clSetKernelArg(csrKernel, 8, sizeof(cl_mem), &devprnew); CHECKERROR;//added


                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelcpu, 8, sizeof(cl_mem), &devprold); CHECKERROR;//added
                errorCode = clSetKernelArg(csrKernelcpu, 9, sizeof(cl_mem), &devprnew); CHECKERROR;//added


                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;

                clFinish(cmdQueue);
                clFinish(cmdQueue_cpu);

                cl_mem prtmpt=devprold;
                devprold=devprnew;
                devprnew=prtmpt;

              }

	    }
	    double teststart = timestamp();
	    for (int i = 0; i < ntimes; i++)
	    {
              for(int order=0; order<TOTALNUMber; order++){


                errorCode = clSetKernelArg(csrKernel, 7, sizeof(cl_mem), &devprold); CHECKERROR;//added
                errorCode = clSetKernelArg(csrKernel, 8, sizeof(cl_mem), &devprnew); CHECKERROR;//added


                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelcpu, 8, sizeof(cl_mem), &devprold); CHECKERROR;//added
                errorCode = clSetKernelArg(csrKernelcpu, 9, sizeof(cl_mem), &devprnew); CHECKERROR;//added


                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;

                clFinish(cmdQueue);
                clFinish(cmdQueue_cpu);

                cl_mem prtmpt=devprold;
                devprold=devprnew;
                devprnew=prtmpt;

              }

	    }
	    double testend = timestamp();
	    double time_in_sec = (testend - teststart)/(double)dim2;
	    double gflops = (double)nnz*2*TOTALNUMber/(time_in_sec/(double)ntimes)/(double)1e9;
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

	if (devRowPtrPad)
	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	free(rowptrpad);


        //free(bitmap);free(rowforcpu);//zf
//	clReleaseMemObject(devbitmap);//zf
    }

        //exit(-1);
    
    
   
    



    //Clean up
 //   if (image2dVec)
//	free(image2dVec);

    if (devRowPtr)
	clReleaseMemObject(devRowPtr);
    if (devColId)
	clReleaseMemObject(devColId);
    if (devData)
	clReleaseMemObject(devData);
    if (devVec)
	clReleaseMemObject(devVec);
    if (devRes)
	clReleaseMemObject(devRes);

    freeObjects(devices, &context, &cmdQueue, &program);


        return;
}


int main(int argc, char* argv[])
{
    if (argc < 2){
	printf("\nUsage: ./main *.mtx \n");
        return 1;
    }

    char* filename = argv[1];
    //cpuoffset = atoi(argv[2]);
    cpuoffset = 50;
    //printf("cpuoffset=%d\n",cpuoffset);
    int ntimes = 5;

    coo_matrix<int, float> mat;
    init_coo_matrix(mat);
    ReadMMF(filename, &mat);

    //////////////////////////////////////up/////////////////////////////////////////////////
    csr_matrix<int, float> csrmattmp;
    coo2csr<int, float>(&mat, &csrmattmp);
    float* tmp=(float*)malloc(sizeof(float)*mat.matinfo.width);
    for(int i=0; i<mat.matinfo.width; i++){
      tmp[i]=csrmattmp.csr_row_ptr[i+1]-csrmattmp.csr_row_ptr[i];
    }   
    for(int i=0; i<mat.matinfo.nnz; i++){
      mat.coo_data[i]=1.0/tmp[mat.coo_row_id[i]];
    }   
    free_csr_matrix(csrmattmp);
    free(tmp);



    for(int i=0; i<mat.matinfo.nnz; i++){
      int tmpt=mat.coo_row_id[i];
      mat.coo_row_id[i]=mat.coo_col_id[i];
      mat.coo_col_id[i]=tmpt;
    }



    ////////////////////////////////////////down///////////////////////////////////////////////




    //char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    
   {
	sprintf(clfilename, "%s%s", "/home/pacman/zf/wubo/apu_corun/pagerank_dynamic", "/spmv_csr_vector.cl");
    printMatInfo(&mat);


    if(mat.matinfo.width != mat.matinfo.height){
      printf("width != height\n");
      exit(-1);
    }



    csr_matrix<int, float> csrmat;
    coo2csr<int, float>(&mat, &csrmat);


////////////////////////////////////////////////////////////////////////////////////////////////////////
    float* prnew = (float*)malloc(sizeof(float)*mat.matinfo.width);
    float* prold = (float*)malloc(sizeof(float)*mat.matinfo.width);
    float* pr0 = (float*)malloc(sizeof(float)*mat.matinfo.width);
    float tmpt=1.0;
    //float tmpt=1.0/(float)mat.matinfo.width;
    printf("cpu start initialization\n");
    for(int i=0; i<mat.matinfo.width; i++){
      prnew[i]=tmpt;
      prold[i]=tmpt;
      pr0[i]=tmpt;
    }
    int height=mat.matinfo.width;


    float d=0.85;
    float e=0.99;
    float distance=0;
    printf("cpu start computing\n");
    //while(distance < e){
    for(int order=0; order<TOTALNUMber; order++){
      for(int row=0; row<height; row++){
        int start = csrmat.csr_row_ptr[row];
        int end = csrmat.csr_row_ptr[row+1];
        float accumulant = 0;
//        printf("row=%d start\n",row);
        for(int j=start; j<end; j++){
          int col=csrmat.csr_col_id[j];
          float data=csrmat.csr_data[j];
          accumulant += data*prold[col];
          //accumulant += data*prold[j];
        }
        //printf("row=%d start=%d end=%d\n",row,start,end);
        accumulant = accumulant*d + (1-d)*pr0[row];
        prnew[row]=accumulant;
      //  if(row<15)
      // printf("cpu prold[%d]=%f, prnew[%d]=%f\n",row,prold[row],row,prnew[row]);

      }

      float *prtmpt=prnew;
      prnew=prold;
      prold=prtmpt;

    }

      for(int i=0; i<height; i++){
        distance+=(prnew[i]-prold[i])*(prnew[i]-prold[i]);
      }
      distance=sqrt(distance);
      printf("number=%d   distance=%f\n",TOTALNUMber ,distance);


//    exit(0);
    float* coores = (float*)malloc(sizeof(float)*mat.matinfo.height);
    for(int i=0; i<mat.matinfo.height; i++)
      coores[i]=prold[i];

    float* res = (float*)malloc(sizeof(float)*mat.matinfo.height);




////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    float* vec = (float*)malloc(sizeof(float)*mat.matinfo.width);
    float* res = (float*)malloc(sizeof(float)*mat.matinfo.height);
    initVectorOne<int, float>(vec, mat.matinfo.width);	
    initVectorZero<int, float>(res, mat.matinfo.height);
    float* coores = (float*)malloc(sizeof(float)*mat.matinfo.height);
    spmv_only(&mat, vec, coores);
*/
    double opttime1 = 10000.0f;
    int optmethod1 = 0;

    spmv_csr_vector_ocl(&csrmat, pr0, res, 0,  opttime1, optmethod1, clfilename, coores, ntimes);//zf method1
    //spmv_csr_vector_ocl(&csrmat, vec, res, 0,  opttime1, optmethod1, clfilename, coores, ntimes);//zf method1

	double opttime2 = 10000.0f;
	int optmethod2 = 0;

	csr_matrix<int, float> padcsr;
	pad_csr(&csrmat, &padcsr, WARPSIZE / 2);
	printf("\nNNZ Before %d After %d\n", csrmat.matinfo.nnz, padcsr.matinfo.nnz);
	spmv_csr_vector_ocl(&padcsr, pr0, res, 16, opttime2, optmethod2, clfilename, coores, ntimes);
	free_csr_matrix(padcsr);

	int nnz = mat.matinfo.nnz;
	double gflops = (double)nnz*2*TOTALNUMber/opttime1/(double)1e9;
	printf("\n------------------------------------------------------------------------\n");
	printf("CSR VEC without padding best time %f ms best method %d gflops %f", opttime1*1000.0, optmethod1, gflops);
	printf("\n------------------------------------------------------------------------\n");
	gflops = (double)nnz*2*TOTALNUMber/opttime2/(double)1e9;
	printf("CSR VEC with padding best time %f ms best method %d gflops %f", opttime2*1000.0, optmethod2, gflops);
	printf("\n------------------------------------------------------------------------\n");
         double optfinal=opttime1;
        if(opttime2<optfinal)
          optfinal=opttime2;
        printf("CAUTTION: kernel time(ms): %f\n",optfinal*1000.0);
  

    //free(vec);
    //free(res);
    free_csr_matrix(csrmat);
    free(coores);


    }

    free_coo_matrix(mat);

    return 0;
}

