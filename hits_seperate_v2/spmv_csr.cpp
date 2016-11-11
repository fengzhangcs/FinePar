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



    //spmv_csr_vector_ocl(&csrmat, &csrmatT, cooresa, cooresh, 0,  opttime1, optmethod1, clfilename, ntimes);//zf method1
void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, csr_matrix<int, float>* matT, float* resa, float* resh, float* res, int padNum, double& opttime, int& optmethod, char* oclfilename, int ntimes)
//void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, float* vec, float* result, int padNum, double& opttime, int& optmethod, char* oclfilename, float* coores, int ntimes)
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
    cl_mem devRowPtrT;
    cl_mem devColIdT;
    cl_mem devDataT;

    cl_mem devaold;
    cl_mem devanew;
    cl_mem devhold;
    cl_mem devhnew;

    /*
    cl_mem devVec;
    cl_mem devprnew;
    cl_mem devprold;
    */

    cl_mem devRowPtr_cpu;
    cl_mem devColId_cpu;
    cl_mem devData_cpu;
    cl_mem devRowPtrT_cpu;
    cl_mem devColIdT_cpu;
    cl_mem devDataT_cpu;

    //cl_mem devVec_cpu;


    //cl_mem devRes;

    //Initialize values
    int nnz = mat->matinfo.nnz;
    int vecsize = mat->matinfo.width;
    int rownum = mat->matinfo.height;
    int rowptrsize = rownum + 1;
    ALLOCATE_GPU_READ(devRowPtr, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ(devColId, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ(devData, mat->csr_data, sizeof(float)*nnz);
    ALLOCATE_GPU_READ(devRowPtrT, matT->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ(devColIdT, matT->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ(devDataT, matT->csr_data, sizeof(float)*nnz);


    /*
    devprold= clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devprold, CL_TRUE, 0, sizeof(float)*rownum, vec, 0, NULL, NULL); CHECKERROR;
    devprnew= clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devprnew, CL_TRUE, 0, sizeof(float)*rownum, vec, 0, NULL, NULL); CHECKERROR;
    */

    ALLOCATE_GPU_READ_cpu(devRowPtr_cpu, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ_cpu(devColId_cpu, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ_cpu(devData_cpu, mat->csr_data, sizeof(float)*nnz);
    ALLOCATE_GPU_READ_cpu(devRowPtrT_cpu, matT->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ_cpu(devColIdT_cpu, matT->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ_cpu(devDataT_cpu, matT->csr_data, sizeof(float)*nnz);

    ALLOCATE_GPU_READ_cpu(devaold, res, sizeof(float)*vecsize);
    ALLOCATE_GPU_READ_cpu(devanew, res, sizeof(float)*vecsize);
    ALLOCATE_GPU_READ_cpu(devhold, res, sizeof(float)*vecsize);
    ALLOCATE_GPU_READ_cpu(devhnew, res, sizeof(float)*vecsize);

   
    //int paddedres = findPaddedSize(rownum, 16);
    //devRes = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devaold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devanew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devhold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devhnew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;




    opttime = 10000.0f;
    optmethod = 0;
    int dim2 = 1;


    {
	int methodid = 7;


       
//////////////////////////////////////////////////////////////////////////////////////

  costtime3=timestamp();//costtime
        
        printf("sizeofulong=%d bytes",sizeof(unsigned long));
        unsigned long *bitmap=(unsigned long*)malloc(sizeof(unsigned long)*(((rownum)>>6) + 1));
	int* rowforcpu = (int*)malloc(sizeof(int)*(rownum));
        int rowforcpusum=0;
        memset(bitmap,0,(sizeof(unsigned long)*(((rownum)>>6) + 1)));

        //cpuoffset=2048;
        for(int i=0 ; i<rownum; i++){
          int numi=(mat->csr_row_ptr[i+1])-(mat->csr_row_ptr[i]);
          //if(numi>=16){
          //if(numi>=0){
          if(numi<=cpuoffset){
          //if(numi>=cpuoffset){
          //if(numi>=WARPSIZE){
            bitmap[(i>>6)]=bitmap[(i>>6)]|(1ul<<((i)&0x3f));
          }
          else {
          //else if(numi>=0) {
          //else if(numi!=0) {
            rowforcpu[rowforcpusum]=i;
            rowforcpusum++;
//            printf("%d, ",i);
          }
//          if((i==275)||(i==688))
 //           printf("num%d=%d\n",i,numi);
        }

	cl_mem devbitmap;
	ALLOCATE_GPU_READ(devbitmap, bitmap, sizeof(unsigned long)*(((rownum)>>6)+1));
	cl_mem devrowforcpu;
	ALLOCATE_GPU_READ_cpu(devrowforcpu, rowforcpu, sizeof(int)*(rowforcpusum));
        free(bitmap);
        free(rowforcpu);
       

        unsigned long *bitmapT=(unsigned long*)malloc(sizeof(unsigned long)*(((rownum)>>6) + 1));
	int* rowforcpuT = (int*)malloc(sizeof(int)*(rownum));
        int rowforcpusumT=0;
        memset(bitmapT,0,(sizeof(unsigned long)*(((rownum)>>6) + 1)));

        for(int i=0 ; i<rownum; i++){
          int numi=(matT->csr_row_ptr[i+1])-(matT->csr_row_ptr[i]);
          if(numi<=cpuoffset){
          //if(numi>=64){
          //if(numi>=WARPSIZE){
            bitmapT[(i>>6)]=bitmapT[(i>>6)]|(1ul<<((i)&0x3f));
          }
          else{
          //else if(numi!=0) {
            rowforcpuT[rowforcpusumT]=i;
            rowforcpusumT++;
//            printf("%d, ",i);
          }
//          if((i==275)||(i==688))
 //           printf("num%d=%d\n",i,numi);
        }

	cl_mem devbitmapT;
	ALLOCATE_GPU_READ(devbitmapT, bitmapT, sizeof(unsigned long)*(((rownum)>>6)+1));
	cl_mem devrowforcpuT;
	ALLOCATE_GPU_READ_cpu(devrowforcpuT, rowforcpuT, sizeof(int)*(rowforcpusumT));
        free(bitmapT);
        free(rowforcpuT);
       

  costtime4=timestamp();//costtime

//////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////



	//printf("\nRow Num %d padded size %d\n", rownum, padrowsize);
	cl_uint work_dim = 1;
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};
        //int threadsinrow=16;

        int rowsetzf=(double)cpuoffset/100*rownum;//zf
        printf("rowsetzf=%d\n",rowsetzf);


	cl_kernel csrKernela = NULL;
	csrKernela = clCreateKernel(program, "gpu_csr_ve_slm_pm_fs", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela, 0, sizeof(cl_mem), &devRowPtrT); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela, 1, sizeof(cl_mem), &devColIdT); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela, 2, sizeof(cl_mem), &devDataT); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela, 3, sizeof(cl_mem), &devhold); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela, 4, sizeof(cl_mem), &devanew); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clSetKernelArg(csrKernela, 6, sizeof(cl_mem), &devbitmapT); CHECKERROR;
	//errorCode = clSetKernelArg(csrKernela, 6, sizeof(int), &rowsetzf); CHECKERROR;

	cl_kernel csrKernelh = NULL;
	csrKernelh = clCreateKernel(program, "gpu_csr_ve_slm_pm_fs", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh, 0, sizeof(cl_mem), &devRowPtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh, 3, sizeof(cl_mem), &devaold); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh, 4, sizeof(cl_mem), &devhnew); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clSetKernelArg(csrKernelh, 6, sizeof(cl_mem), &devbitmap); CHECKERROR;
//	errorCode = clSetKernelArg(csrKernelh, 6, sizeof(int), &rowsetzf); CHECKERROR;


	cl_kernel csrKernela_cpu = NULL;
	csrKernela_cpu = clCreateKernel(program, "cpu_csr_ve_slm_pm_fs", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela_cpu, 0, sizeof(cl_mem), &devRowPtrT_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela_cpu, 1, sizeof(cl_mem), &devColIdT_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela_cpu, 2, sizeof(cl_mem), &devDataT_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela_cpu, 3, sizeof(cl_mem), &devhold); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela_cpu, 4, sizeof(cl_mem), &devanew); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela_cpu, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clSetKernelArg(csrKernela_cpu, 6, sizeof(cl_mem), &devrowforcpuT); CHECKERROR;
	errorCode = clSetKernelArg(csrKernela_cpu, 7, sizeof(int), &rowforcpusumT); CHECKERROR;
	//errorCode = clSetKernelArg(csrKernela_cpu, 6, sizeof(int), &rowsetzf); CHECKERROR;

	cl_kernel csrKernelh_cpu = NULL;
	csrKernelh_cpu = clCreateKernel(program, "cpu_csr_ve_slm_pm_fs", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh_cpu, 0, sizeof(cl_mem), &devRowPtr_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh_cpu, 1, sizeof(cl_mem), &devColId_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh_cpu, 2, sizeof(cl_mem), &devData_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh_cpu, 3, sizeof(cl_mem), &devaold); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh_cpu, 4, sizeof(cl_mem), &devhnew); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh_cpu, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clSetKernelArg(csrKernelh_cpu, 6, sizeof(cl_mem), &devrowforcpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelh_cpu, 7, sizeof(int), &rowforcpusum); CHECKERROR;
//	errorCode = clSetKernelArg(csrKernelh_cpu, 6, sizeof(int), &rowsetzf); CHECKERROR;


	cl_kernel csrKernelcpu = NULL;
        size_t globalsizecpu=3;
        size_t localsizecpu=1;

	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;
	double mincpulooptime = 1000.0f;
	double mingpulooptime = 1000.0f;
        cl_uint work_dimcpu=1;
	
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


	for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
	{
	    size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};

            errorCode = clEnqueueWriteBuffer(cmdQueue, devaold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devanew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devhold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devhnew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;


            clFinish(cmdQueue);


            for(int order=0; order<TOTALNUMber; order++){

              errorCode = clSetKernelArg(csrKernela, 3, sizeof(cl_mem), &devhold); CHECKERROR;
              errorCode = clSetKernelArg(csrKernela, 4, sizeof(cl_mem), &devanew); CHECKERROR;

              errorCode = clSetKernelArg(csrKernelh, 3, sizeof(cl_mem), &devaold); CHECKERROR;
              errorCode = clSetKernelArg(csrKernelh, 4, sizeof(cl_mem), &devhnew); CHECKERROR;


              errorCode = clSetKernelArg(csrKernela_cpu, 3, sizeof(cl_mem), &devhold); CHECKERROR;
              errorCode = clSetKernelArg(csrKernela_cpu, 4, sizeof(cl_mem), &devanew); CHECKERROR;

              errorCode = clSetKernelArg(csrKernelh_cpu, 3, sizeof(cl_mem), &devaold); CHECKERROR;
              errorCode = clSetKernelArg(csrKernelh_cpu, 4, sizeof(cl_mem), &devhnew); CHECKERROR;



              errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernela, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernela_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;

              errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernelh, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelh_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
              clFinish(cmdQueue);
              clFinish(cmdQueue_cpu);

              cl_mem ptmpt=devaold;
              devaold=devanew;
              devanew=ptmpt;

              ptmpt=devhold;
              devhold=devhnew;
              devhnew=ptmpt;


            }






	    float* tmpresulta = (float*)malloc(sizeof(float)*rownum);
	    float* tmpresulth = (float*)malloc(sizeof(float)*rownum);
	    errorCode = clEnqueueReadBuffer(cmdQueue, devhold, CL_TRUE, 0, sizeof(float)*rownum, tmpresulth, 0, NULL, NULL); CHECKERROR;
	    errorCode = clEnqueueReadBuffer(cmdQueue, devaold, CL_TRUE, 0, sizeof(float)*rownum, tmpresulta, 0, NULL, NULL); CHECKERROR;
	    //errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);

            
            /*
        for(int i=0; i<rowforcpusum; i++){//zfadded
        //  printf("rowforcpu[%d]=%d\n",i,rowforcpu[i]);
          tmpresult[rowforcpu[i]]=coores[rowforcpu[i]];
        }
        */
        
        


	    two_vec_compare(resa, tmpresulta, rownum);
	    two_vec_compare(resh, tmpresulth, rownum);
	    free(tmpresulta);
	    free(tmpresulth);
      /*  }
        }
        }

#if 0
	    for (int k = 0; k < 3; k++)
	    {
              errorCode = clEnqueueWriteBuffer(cmdQueue, devaold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devanew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devhold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devhnew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;


              clFinish(cmdQueue);


              for(int order=0; order<TOTALNUMber; order++){

                errorCode = clSetKernelArg(csrKernela, 3, sizeof(cl_mem), &devhold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernela, 4, sizeof(cl_mem), &devanew); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelh, 3, sizeof(cl_mem), &devaold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernelh, 4, sizeof(cl_mem), &devhnew); CHECKERROR;


                errorCode = clSetKernelArg(csrKernela_cpu, 3, sizeof(cl_mem), &devhold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernela_cpu, 4, sizeof(cl_mem), &devanew); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelh_cpu, 3, sizeof(cl_mem), &devaold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernelh_cpu, 4, sizeof(cl_mem), &devhnew); CHECKERROR;



                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernela, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernela_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;

                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernelh, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelh_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
                clFinish(cmdQueue);
                clFinish(cmdQueue_cpu);

                cl_mem ptmpt=devaold;
                devaold=devanew;
                devanew=ptmpt;

                ptmpt=devhold;
                devhold=devhnew;
                devhnew=ptmpt;


              }

	    }
    */



            double time_in_sec=0;
	    for (int i = 0; i < ntimes; i++)
	    {
              errorCode = clEnqueueWriteBuffer(cmdQueue, devaold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devanew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devhold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devhnew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;

              clFinish(cmdQueue);

              double teststart = timestamp();
              for(int order=0; order<TOTALNUMber; order++){

                errorCode = clSetKernelArg(csrKernela, 3, sizeof(cl_mem), &devhold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernela, 4, sizeof(cl_mem), &devanew); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelh, 3, sizeof(cl_mem), &devaold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernelh, 4, sizeof(cl_mem), &devhnew); CHECKERROR;


                errorCode = clSetKernelArg(csrKernela_cpu, 3, sizeof(cl_mem), &devhold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernela_cpu, 4, sizeof(cl_mem), &devanew); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelh_cpu, 3, sizeof(cl_mem), &devaold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernelh_cpu, 4, sizeof(cl_mem), &devhnew); CHECKERROR;



                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernela, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernela_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;

                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernelh, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelh_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
                clFinish(cmdQueue);
                clFinish(cmdQueue_cpu);

                cl_mem ptmpt=devaold;
                devaold=devanew;
                devanew=ptmpt;

                ptmpt=devhold;
                devhold=devhnew;
                devhnew=ptmpt;

              }

              double testend = timestamp();
              time_in_sec += (testend - teststart)/(double)dim2;

            }
            double gflops = (double)nnz*2*2*TOTALNUMber/(time_in_sec/(double)ntimes)/(double)1e9;
	    printf("\nCSR vector SLM row ptr groupnum:%d cpu time %lf ms GFLOPS %lf code %d \n\n", groupnum,   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	    double onetime = time_in_sec / (double) ntimes;








        cl_event e[2];
time_in_sec=0;
	    for (int i = 0; i < ntimes; i++)
	    {
              errorCode = clEnqueueWriteBuffer(cmdQueue, devaold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devanew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devhold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devhnew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;

              clFinish(cmdQueue);

              double teststart = timestamp();
              for(int order=0; order<TOTALNUMber; order++){

                errorCode = clSetKernelArg(csrKernela, 3, sizeof(cl_mem), &devhold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernela, 4, sizeof(cl_mem), &devanew); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelh, 3, sizeof(cl_mem), &devaold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernelh, 4, sizeof(cl_mem), &devhnew); CHECKERROR;


                errorCode = clSetKernelArg(csrKernela_cpu, 3, sizeof(cl_mem), &devhold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernela_cpu, 4, sizeof(cl_mem), &devanew); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelh_cpu, 3, sizeof(cl_mem), &devaold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernelh_cpu, 4, sizeof(cl_mem), &devhnew); CHECKERROR;



                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernela, work_dim, NULL, globalsize, blocksize, 0, NULL, &(e[0])); CHECKERROR;
                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernela_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, &(e[1])); CHECKERROR;
                clFinish(cmdQueue);
                clFinish(cmdQueue_cpu);
              time_in_sec += get_time2(e[0],e[1],1);

                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernelh, work_dim, NULL, globalsize, blocksize, 0, NULL, &(e[0])); CHECKERROR;
                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelh_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, &(e[1])); CHECKERROR;
                clFinish(cmdQueue);
                clFinish(cmdQueue_cpu);
              time_in_sec += get_time2(e[0],e[1],1);

                cl_mem ptmpt=devaold;
                devaold=devanew;
                devanew=ptmpt;

                ptmpt=devhold;
                devhold=devhnew;
                devhnew=ptmpt;

              }

              double testend = timestamp();
              //time_in_sec += (testend - teststart)/(double)dim2;

            }
	    double onecputime = time_in_sec*1 / (double) ntimes;




time_in_sec=0;
	    for (int i = 0; i < ntimes; i++)
	    {
              errorCode = clEnqueueWriteBuffer(cmdQueue, devaold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devanew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devhold, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devhnew, CL_TRUE, 0, sizeof(float)*vecsize, res, 0, NULL, NULL); CHECKERROR;

              clFinish(cmdQueue);

              double teststart = timestamp();
              for(int order=0; order<TOTALNUMber; order++){

                errorCode = clSetKernelArg(csrKernela, 3, sizeof(cl_mem), &devhold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernela, 4, sizeof(cl_mem), &devanew); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelh, 3, sizeof(cl_mem), &devaold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernelh, 4, sizeof(cl_mem), &devhnew); CHECKERROR;


                errorCode = clSetKernelArg(csrKernela_cpu, 3, sizeof(cl_mem), &devhold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernela_cpu, 4, sizeof(cl_mem), &devanew); CHECKERROR;

                errorCode = clSetKernelArg(csrKernelh_cpu, 3, sizeof(cl_mem), &devaold); CHECKERROR;
                errorCode = clSetKernelArg(csrKernelh_cpu, 4, sizeof(cl_mem), &devhnew); CHECKERROR;



                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernela, work_dim, NULL, globalsize, blocksize, 0, NULL, &(e[0])); CHECKERROR;
                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernela_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, &(e[1])); CHECKERROR;
                //
                clFinish(cmdQueue);
                clFinish(cmdQueue_cpu);
              time_in_sec += get_time2(e[0],e[1],0);

                errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernelh, work_dim, NULL, globalsize, blocksize, 0, NULL, &(e[0])); CHECKERROR;
                errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelh_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, &(e[1])); CHECKERROR;
                clFinish(cmdQueue);
                clFinish(cmdQueue_cpu);
              time_in_sec += get_time2(e[0],e[1],0);

                cl_mem ptmpt=devaold;
                devaold=devanew;
                devanew=ptmpt;

                ptmpt=devhold;
                devhold=devhnew;
                devhnew=ptmpt;

              }

              double testend = timestamp();
              //time_in_sec += (testend - teststart)/(double)dim2;

            }
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

	if (csrKernela)
	    clReleaseKernel(csrKernela);
	if (csrKernela_cpu)
	    clReleaseKernel(csrKernela_cpu);

	if (csrKernelh)
	    clReleaseKernel(csrKernelh);
	if (csrKernelh_cpu)
	    clReleaseKernel(csrKernelh_cpu);




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
    if (devaold)
	clReleaseMemObject(devaold);
    if (devhold)
	clReleaseMemObject(devhold);
    if (devanew)
	clReleaseMemObject(devanew);
    if (devhnew)
	clReleaseMemObject(devhnew);


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
    //printf("cpuoffset=%d\n",cpuoffset);
    int ntimes = 5;

    coo_matrix<int, float> mat;
    init_coo_matrix(mat);
  costtime1=timestamp();//costtime
    ReadMMF(filename, &mat);
  costtime2=timestamp();//costtime

    if(mat.matinfo.width != mat.matinfo.height){
      printf("width != height\n");
      exit(-1);
    }
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

    csr_matrix<int, float> csrmat;
    coo2csr<int, float>(&mat, &csrmat);




    for(int i=0; i<mat.matinfo.nnz; i++){
      int tmpt=mat.coo_row_id[i];
      mat.coo_row_id[i]=mat.coo_col_id[i];
      mat.coo_col_id[i]=tmpt;
    }


    csr_matrix<int, float> csrmatT;
    coo2csr<int, float>(&mat, &csrmatT);



    ////////////////////////////////////////down///////////////////////////////////////////////




    //char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    
   {
	sprintf(clfilename, "%s%s", "/home/pacman/zf/wubo/apu_corun/hits_seperate_v2", "/spmv_csr_vector.cl");
    printMatInfo(&mat);





////////////////////////////////////////////////////////////////////////////////////////////////////////
    float* anew = (float*)malloc(sizeof(float)*mat.matinfo.width);
    float* aold = (float*)malloc(sizeof(float)*mat.matinfo.width);
    float* hnew = (float*)malloc(sizeof(float)*mat.matinfo.width);
    float* hold = (float*)malloc(sizeof(float)*mat.matinfo.width);



    //float* pr0 = (float*)malloc(sizeof(float)*mat.matinfo.width);
    float tmpt=1.0;
    //float tmpt=1.0/(float)mat.matinfo.width;
    printf("cpu start initialization\n");
    for(int i=0; i<mat.matinfo.width; i++){
      anew[i]=tmpt;
      aold[i]=tmpt;
      hnew[i]=tmpt;
      hold[i]=tmpt;
    }
    int height=mat.matinfo.width;


    //float d=0.85;
    //float e=0.99;
    //float distance=0;
    printf("cpu start computing\n");
    //while(distance < e){
    for(int order=0; order<TOTALNUMber; order++){
      for(int row=0; row<height; row++){
        int start = csrmatT.csr_row_ptr[row];
        int end = csrmatT.csr_row_ptr[row+1];
        float accumulant = 0;
        for(int j=start; j<end; j++){
          int col=csrmatT.csr_col_id[j];
          float data=csrmatT.csr_data[j];
          accumulant += data*hold[col];
        }
        anew[row]=accumulant;
      }

      for(int row=0; row<height; row++){
        int start = csrmat.csr_row_ptr[row];
        int end = csrmat.csr_row_ptr[row+1];
        float accumulant = 0;
        for(int j=start; j<end; j++){
          int col=csrmat.csr_col_id[j];
          float data=csrmat.csr_data[j];
          accumulant += data*aold[col];
        }
        hnew[row]=accumulant;
      }

      float *ptmpt=anew;
      anew=aold;
      aold=ptmpt;

      ptmpt=hnew;
      hnew=hold;
      hold=ptmpt;

    }


//    exit(0);
    float* cooresa = (float*)malloc(sizeof(float)*mat.matinfo.height);
    float* cooresh = (float*)malloc(sizeof(float)*mat.matinfo.height);
    float* res = (float*)malloc(sizeof(float)*mat.matinfo.height);
    for(int i=0; i<mat.matinfo.height; i++){
      cooresa[i]=aold[i];
      cooresh[i]=hold[i];
      res[i]=1;
    }





////////////////////////////////////////////////////////////////////////////////////////////////////////

    double opttime1 = 10000.0f;
    int optmethod1 = 0;

    spmv_csr_vector_ocl(&csrmat, &csrmatT, cooresa, cooresh, res, 0,  opttime1, optmethod1, clfilename, ntimes);//zf method1
    //spmv_csr_vector_ocl(&csrmat, pr0, res, 0,  opttime1, optmethod1, clfilename, coores, ntimes);//zf method1

	double opttime2 = 10000.0f;
	int optmethod2 = 0;

	csr_matrix<int, float> padcsr;
	csr_matrix<int, float> padcsrT;
	pad_csr(&csrmat, &padcsr, WARPSIZE / 2);
	pad_csr(&csrmatT, &padcsrT, WARPSIZE / 2);
	printf("\nNNZ Before %d After %d\n", csrmat.matinfo.nnz, padcsr.matinfo.nnz);
//    spmv_csr_vector_ocl(&padcsr, &padcsrT, cooresa, cooresh, res, 0,  opttime2, optmethod2, clfilename, ntimes);//zf method1
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
    //free(coores);


    }

    free_coo_matrix(mat);

    return 0;
}

