#include "spmv_util.h"
#define TOTALNUMber 5
int cpuoffset;
double costtime0,costtime1,costtime2,costtime3,costtime4,costtime5;

double get_time2(cl_event gpuEvt, cl_event cpuEvt, int ret){
  cl_int status;
  cl_int eventStatus = CL_QUEUED;

  cl_ulong startTimeall, startTime, endTime;
  cl_ulong overlapstartTime, overlapendTime;
  cl_ulong startTime1, endTime1;
  cl_ulong startTime0, endTime0;
  while(eventStatus != CL_COMPLETE)
  {
    status = clGetEventInfo(
        cpuEvt,
        CL_EVENT_COMMAND_EXECUTION_STATUS,
        sizeof(cl_int),
        &eventStatus,
        NULL);
    if (CL_SUCCESS != status)printf("error! %d\n", status);
  }
  eventStatus = CL_QUEUED;
  while(eventStatus != CL_COMPLETE)
  {
    status = clGetEventInfo(
        gpuEvt,
        CL_EVENT_COMMAND_EXECUTION_STATUS,
        sizeof(cl_int),
        &eventStatus,
        NULL);
    if (CL_SUCCESS != status)printf("error! %d\n", status);
  }

  status = clGetEventProfilingInfo(cpuEvt,
      CL_PROFILING_COMMAND_START,
      sizeof(cl_ulong),
      &startTime1,
      0);  
  if (CL_SUCCESS != status)printf("error! %d\n", status);
  status = clGetEventProfilingInfo(cpuEvt,
      CL_PROFILING_COMMAND_END,
      sizeof(cl_ulong),
      &endTime1,
      0);  
  if (CL_SUCCESS != status)printf("error! %d\n", status);//zf

  status = clGetEventProfilingInfo(gpuEvt,
      CL_PROFILING_COMMAND_START,
      sizeof(cl_ulong),
      &startTime0,
      0);
  if (CL_SUCCESS != status)printf("error! %d\n", status);
  status = clGetEventProfilingInfo(gpuEvt,
      CL_PROFILING_COMMAND_END,
      sizeof(cl_ulong),
      &endTime0,
      0);
  if (CL_SUCCESS != status)printf("error! %d\n", status);
//0 is for gpuevt while 1 is for cpuevt
  startTime=(startTime1<startTime0)?startTime1: startTime0;
  endTime=(endTime0<endTime1)?endTime1: endTime0;

  overlapstartTime=(startTime1>startTime0)?startTime1: startTime0;
  overlapendTime=(endTime0>endTime1)?endTime1: endTime0;
  //status = clReleaseEvent(cpuEvt);
  
  //printf("ms gpuevent %lf | cpuevent %lf\n", (double)((endTime0-startTime0)*1e-6), (double)((endTime1-startTime1)*1e-6));

  //printf("###, starttime, endtime, gpustarttime, gpuendtime, cpustarttime, cpuendtime, totaltime, gputime, cputime, device, overlapstart, overlapend, overlaptime ,overlappercent, longestpercent, shortestpercent\n");
  startTimeall =startTime; 
  startTime -= startTimeall;
  endTime -= startTimeall;
  startTime0 -= startTimeall;
  endTime0 -= startTimeall;
  startTime1 -= startTimeall;
  endTime1 -= startTimeall;
  overlapstartTime -= startTimeall;
  overlapendTime -= startTimeall;

  
  double gputime = (double)((endTime0-startTime0)*1e-6);
  double cputime = (double)((endTime1-startTime1)*1e-6);
  double longest = (gputime>cputime)?gputime:cputime;
  double shortest = (gputime<cputime)?gputime:cputime;
  /*
  printf("LOOKOUT, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, ",(double)(startTime*1e-6), (double)(endTime*1e-6), (double)(startTime0*1e-6), (double)(endTime0*1e-6), (double)(startTime1*1e-6), (double)(endTime1*1e-6),
      (double)((endTime-startTime)*1e-6),  (double)((endTime0-startTime0)*1e-6), (double)((endTime1-startTime1)*1e-6));
  if(startTime1<startTime0)printf("cpu, ");
  else printf("gpu, ");
  printf("%lf, %lf, %lf, ", (double)(overlapstartTime*1e-6),(double)(overlapendTime*1e-6),(double)((overlapendTime-overlapstartTime)*1e-6));
  if(overlapstartTime<overlapendTime){
    printf("%lf, ",(double)((double)(overlapendTime-overlapstartTime)/(endTime-startTime)));
  }
  else{
    printf("NULL, ");
  }
  printf("%lf, %lf\n", longest/((endTime-startTime)*1e-6), shortest/((endTime-startTime)*1e-6));
  */
  
  


  //return (double)((endTime-startTime)*1e-6);

  if(ret==0)
    return (double)((endTime0-startTime0)*1e-6);
  if(ret==1){
//    printf("cpu ms %lf\n",(endTime1-startTime1)*1e-6);
    return (double)((endTime1-startTime1)*1e-6);
  }
  else
    return 1;
}



void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, float* vec, float* result, int padNum, double& opttime, int& optmethod, char* oclfilename, float* coores, int ntimes)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_command_queue cmdQueue_cpu = NULL;
    cl_program program = NULL;

    assert(initialization2(devices, &context, &cmdQueue, &program, oclfilename, &cmdQueue_cpu) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devRowPtr;
    cl_mem devColId;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devprnew;
    cl_mem devprold;

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

  costtime3=timestamp();//costtime
        printf("sizeofulong=%d bytes",sizeof(unsigned long));
        unsigned long *bitmap=(unsigned long*)malloc(sizeof(unsigned long)*(((padrowsize)>>6) + 1));
	int* rowforcpu = (int*)malloc(sizeof(int)*(padrowsize));
        int rowforcpusum=0;
        memset(bitmap,0,(sizeof(unsigned long)*(((padrowsize)>>6) + 1)));

        //cpuoffset=2048;
        for(int i=0 ; i<padrowsize; i++){
          int numi=rowptrpad[i+1]-rowptrpad[i];
          if(numi<=cpuoffset){
          //if(numi>=cpuoffset){
          //if(numi>=64){
          //if(numi>=WARPSIZE){
            bitmap[(i>>6)]=bitmap[(i>>6)]|(1ul<<((i)&0x3f));
          }
          else{
          //else if(numi!=0) {
            rowforcpu[rowforcpusum]=i;
            rowforcpusum++;
            if(i==1){printf("wrong!\n");}
//            printf("%d, ",i);
          }
 //         if((i==1)||(i==2))
//            printf("\nnum%d=%d\n",i,numi);
        }
        /*
          for(int i=0; i<5; i++)
            printf("rowforcpu[%d]=%d\n",i,rowforcpu[i]);
          printf("\n");
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
       



  costtime4=timestamp();//costtime

//////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////



	printf("\nRow Num %d padded size %d\n", rownum, padrowsize);
	cl_uint work_dim = 1;
	//cl_uint work_dim = 2;
	//int dim2 = 16;
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};
        //int threadsinrow=16;

	cl_kernel csrKernel = NULL;
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




	cl_kernel csrKernelcpu = NULL;
        size_t globalsizecpu=3;
        //size_t globalsizecpu=4;
        //size_t globalsizecpu=rowforcpusum;
        size_t localsizecpu=1;
	csrKernelcpu = clCreateKernel(program, "cpu_csr", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 0, sizeof(cl_mem), &devRowPtrPad_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 1, sizeof(cl_mem), &devColId_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 2, sizeof(cl_mem), &devData_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 3, sizeof(cl_mem), &devVec_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 4, sizeof(cl_mem), &devRes); CHECKERROR;
        
	//wrong errorCode = clSetKernelArg(csrKernelcpu, 5, sizeof(int), &rowforcpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 5, sizeof(int), &rownum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 6, sizeof(cl_mem), &devrowforcpu); CHECKERROR;
	
	errorCode = clSetKernelArg(csrKernelcpu, 7, sizeof(int), &rowforcpusum); CHECKERROR;//newadded

        errorCode = clSetKernelArg(csrKernelcpu, 8, sizeof(cl_mem), &devprold); CHECKERROR;//added
        errorCode = clSetKernelArg(csrKernelcpu, 9, sizeof(cl_mem), &devprnew); CHECKERROR;//added

        //new kernel
        cl_kernel disKernel = NULL;
        disKernel = clCreateKernel(program, "caldistance", &errorCode); CHECKERROR;
        errorCode = clSetKernelArg(disKernel, 0, sizeof(int), &rownum); CHECKERROR;
        errorCode = clSetKernelArg(disKernel, 1, sizeof(cl_mem), &devprold); CHECKERROR;
        errorCode = clSetKernelArg(disKernel, 2, sizeof(cl_mem), &devprnew); CHECKERROR;
        errorCode = clSetKernelArg(disKernel, 3, sizeof(cl_mem), &devdistance); CHECKERROR;





	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;
	double mincpulooptime = 1000.0f;
	double mingpulooptime = 1000.0f;
        cl_uint work_dimcpu=1;
	
	for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
	{
	    //size_t globalsize[] = {totalsize, dim2};
	    size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};

	    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devprold, CL_TRUE, 0, sizeof(float)*rownum, vec, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devprnew, CL_TRUE, 0, sizeof(float)*rownum, vec, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);



  costtime5=timestamp();//costtime
  printf(" simple:totalms: %lf ioms: %lf ioms%%: %lf bitmapms: %lf bitmapms%%: %lf otherms: %lf otherms%%: %lf %lf %lf %lf %lf %lf %lf\n",
    (costtime5-costtime0)*1000,
      (costtime2-costtime1)*1000,
      (costtime2-costtime1)/(costtime5-costtime0),
      (costtime4-costtime3)*1000,
      (costtime4-costtime3)/(costtime5-costtime0),
      ((costtime5-costtime0)-(costtime2-costtime1)-(costtime4-costtime3))*1000,
      ((costtime5-costtime0)-(costtime2-costtime1)-(costtime4-costtime3))/(costtime5-costtime0),
      costtime0,costtime1,costtime2,costtime3,costtime4,costtime5);
  printf(" percent:totalms: %lf ioms: %lf ioms%%: %lf bitmapms: %lf bitmapms%%: %lf otherms: %lf otherms%%: %lf %lf %lf %lf %lf %lf %lf\n",
    (costtime5-costtime0)*1000,
      (costtime2-costtime1)*1000,
      (costtime2-costtime1)/(costtime5-costtime0)*100,
      (costtime4-costtime3)*1000,
      (costtime4-costtime3)/(costtime5-costtime0)*100,
      ((costtime5-costtime0)-(costtime2-costtime1)-(costtime4-costtime3))*1000,
      ((costtime5-costtime0)-(costtime2-costtime1)-(costtime4-costtime3))/(costtime5-costtime0)*100,
      costtime0,costtime1,costtime2,costtime3,costtime4,costtime5);


  //exit(0);//costtime



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

            /*
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
            */
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





time_in_sec = 0;//missed
        cl_event e[2];

	    teststart = timestamp();
	    for (int i = 0; i < ntimes; i++)
	    {
              for(int order=0; order<TOTALNUMber; order++){


                errorCode = clSetKernelArg(csrKernel, 7, sizeof(cl_mem), &devprold); CHECKERROR;//added
                errorCode = clSetKernelArg(csrKernel, 8, sizeof(cl_mem), &devprnew); CHECKERROR;//added
                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, &(e[0])); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelcpu, 8, sizeof(cl_mem), &devprold); CHECKERROR;//added
                errorCode = clSetKernelArg(csrKernelcpu, 9, sizeof(cl_mem), &devprnew); CHECKERROR;//added


                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, &(e[1])); CHECKERROR;

                clFinish(cmdQueue);
                clFinish(cmdQueue_cpu);
              time_in_sec += get_time2(e[0],e[1],1);

                cl_mem prtmpt=devprold;
                devprold=devprnew;
                devprnew=prtmpt;

              }

	    }
	    testend = timestamp();
	    //time_in_sec = (testend - teststart)/(double)dim2;
	    double onecputime = time_in_sec*1 / (double) ntimes;




time_in_sec = 0;//missed
	    teststart = timestamp();
	    for (int i = 0; i < ntimes; i++)
	    {
              for(int order=0; order<TOTALNUMber; order++){


                errorCode = clSetKernelArg(csrKernel, 7, sizeof(cl_mem), &devprold); CHECKERROR;//added
                errorCode = clSetKernelArg(csrKernel, 8, sizeof(cl_mem), &devprnew); CHECKERROR;//added


                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, &(e[0])); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelcpu, 8, sizeof(cl_mem), &devprold); CHECKERROR;//added
                errorCode = clSetKernelArg(csrKernelcpu, 9, sizeof(cl_mem), &devprnew); CHECKERROR;//added


                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, &(e[1])); CHECKERROR;

                clFinish(cmdQueue);
                clFinish(cmdQueue_cpu);
              time_in_sec += get_time2(e[0],e[1],0);

                cl_mem prtmpt=devprold;
                devprold=devprnew;
                devprnew=prtmpt;

              }

	    }
	    testend = timestamp();
	   // time_in_sec = (testend - teststart)/(double)dim2;
	    double onegputime = time_in_sec*1 / (double) ntimes;








	    if (onetime < opttime)
	    {
		opttime = onetime;
		optmethod = methodid;
	    }
	    if (onetime < minlooptime)
	    {
		minlooptime = onetime;
            mincpulooptime = onecputime;
            mingpulooptime = onegputime;
 
		maxloopsize = groupnum;
	    }
	}
	printf("******* Min time %f groupnum %d **********", minlooptime, maxloopsize);
        printf("\n\nCPUTIMEinms: %f GPUTIMEinms: %f cpuoffset: %d\n\n",mincpulooptime, mingpulooptime, cpuoffset);

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
  costtime0=timestamp();//costtime
    if (argc < 3){
	printf("\nUsage: ./main *.mtx percent\n");
        return 1;
    }

    char* filename = argv[1];
    cpuoffset = atoi(argv[2]);
    int ntimes = 5;

    coo_matrix<int, float> mat;
    init_coo_matrix(mat);
  costtime1=timestamp();//costtime
    ReadMMF(filename, &mat);
  costtime2=timestamp();//costtime

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
	sprintf(clfilename, "%s%s", ".", "/spmv_csr_vector.cl");
	//sprintf(clfilename, "%s%s", "/home/pacman/zf/wubo/apu_corun/pagerank_seperate_v2", "/spmv_csr_vector.cl");
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
//	spmv_csr_vector_ocl(&padcsr, pr0, res, 16, opttime2, optmethod2, clfilename, coores, ntimes);
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

