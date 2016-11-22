#include "spmv_util.h"
#define MY_INFINITY    0xffffff00
#define TOTALNUMber 100
int cpuoffset;
double costtime0,costtime1,costtime2,costtime3,costtime4,costtime5;

/*
    int vertex_cnt=mat.matinfo.height;
    int edge_cnt=csrmat.matinfo.nnz;
    int* parents=(int*)malloc(sizeof(int)*vertex_cnt);
    int* shadow=(int*)malloc(sizeof(int)*vertex_cnt);
    char* mask=(char*)malloc(sizeof(char)*edge_cnt);
    int* edge_src=(int*)malloc(sizeof(int)*edge_cnt);
*/



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



    //spmv_csr_vector_ocl(&csrmat, parents, shadow, mask, edge_src, 0,  opttime1, optmethod1, clfilename, ntimes);//zf method1
void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, int* parents, int* shadow, char* mask, int* edge_src, int padNum, double& opttime, int& optmethod, char* oclfilename, int ntimes, int* rescoo)
//void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, int* vplist, int* randlist, int padNum, double& opttime, int& optmethod, char* oclfilename, int* vplistres, int ntimes)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_command_queue cmdQueue_cpu = NULL;
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

  costtime3=timestamp();//costtime
        printf("sizeofulong=%d bytes\n",sizeof(unsigned long));
        unsigned long *bitmap=(unsigned long*)malloc(sizeof(unsigned long)*(((rownum)>>6) + 1));
	int* rowforcpu = (int*)malloc(sizeof(int)*(rownum));
        int rowforcpusum=0;
        memset(bitmap,0,(sizeof(unsigned long)*(((rownum)>>6) + 1)));

        //cpuoffset=2048;
        for(int i=0 ; i<rownum; i++){
          int numi=(mat->csr_row_ptr[i+1])-(mat->csr_row_ptr[i]);
          if(numi<=cpuoffset){
          //if(numi>=WARPSIZE){
            bitmap[(i>>6)]=bitmap[(i>>6)]|(1ul<<((i)&0x3f));
          }
          else if(numi!=0) {
            rowforcpu[rowforcpusum]=i;
            rowforcpusum++;
          }
        }

	cl_mem devbitmap;
	ALLOCATE_GPU_READ(devbitmap, bitmap, sizeof(unsigned long)*(((rownum)>>6)+1));

	cl_mem devrowforcpu;
	ALLOCATE_GPU_READ_cpu(devrowforcpu, rowforcpu, sizeof(int)*(rowforcpusum));
       

  costtime4=timestamp();//costtime



//////////////////////////////////////////////////////////////////////////////////////




	//printf("\nRow Num %d padded size %d\n", rownum, padrowsize);
	cl_uint work_dim = 1;
	//cl_uint work_dim = 2;
	//int dim2 = 16;
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};
        //int threadsinrow=16;

	cl_kernel kernel_hooking= NULL;
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

	errorCode = clSetKernelArg(kernel_hooking, 10, sizeof(cl_mem), &devbitmap); CHECKERROR;

        //int rowsetzf=(double)cpuoffset/100*vertex_cnt;
        //printf("rowsetzf=%d  cpuoffset=%d%%\n",rowsetzf,cpuoffset);
	//errorCode = clSetKernelArg(kernel_hooking, 10, sizeof(int), &rowsetzf); CHECKERROR;//zf


        size_t globalsizecpu=3;
        size_t localsizecpu=1;
        /*
    ALLOCATE_GPU_READ_cpu(devRowPtr_cpu, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ_cpu(devColId_cpu, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ_cpu(devedge_src_cpu, edge_src, sizeof(int)*edge_cnt);


         */

	cl_kernel kernel_hooking_cpu= NULL;
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

	errorCode = clSetKernelArg(kernel_hooking_cpu, 10, sizeof(cl_mem), &devrowforcpu); CHECKERROR;
	errorCode = clSetKernelArg(kernel_hooking_cpu, 11, sizeof(int), &rowforcpusum); CHECKERROR;//zf

	//errorCode = clSetKernelArg(kernel_hooking_cpu, 10, sizeof(int), &rowsetzf); CHECKERROR;//zf





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




        /*
        int rowsetzf=(double)cpuoffset/100*rownum;//zf
        printf("rowsetzf=%d\n",rowsetzf);
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int), &rowsetzf); CHECKERROR;



	cl_kernel csrKernelcpu = NULL;
        size_t globalsizecpu=3;
        //size_t globalsizecpu=4;
        //size_t globalsizecpu=rowforcpusum;
        size_t localsizecpu=1;
	csrKernelcpu = clCreateKernel(program, "cpu_csr", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 0, sizeof(cl_mem), &devRowPtr_cpu); CHECKERROR;
	//errorCode = clSetKernelArg(csrKernelcpu, 0, sizeof(cl_mem), &devRowPtrPad_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 1, sizeof(cl_mem), &devColId_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 2, sizeof(cl_mem), &devrandlist_cpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 3, sizeof(cl_mem), &devvplist); CHECKERROR;
//	errorCode = clSetKernelArg(csrKernelcpu, 4, sizeof(int), &color); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 5, sizeof(cl_mem), &devover); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 6, sizeof(int), &rownum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 7, sizeof(int), &rowsetzf); CHECKERROR;
        */

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
    errorCode = clEnqueueWriteBuffer(cmdQueue, devparents, CL_TRUE, 0, sizeof(int)*vertex_cnt, parents, 0, NULL, NULL); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devshadow, CL_TRUE, 0, sizeof(int)*vertex_cnt, shadow, 0, NULL, NULL); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devmask, CL_TRUE, 0, sizeof(char)*edge_cnt, mask, 0, NULL, NULL); CHECKERROR;
    clFinish(cmdQueue);
    
         // errorCode = clEnqueueWriteBuffer(cmdQueue, devvplist, CL_TRUE, 0, sizeof(float)*rownum, vplist, 0, NULL, NULL); CHECKERROR;
          //clFinish(cmdQueue);

          char stop=0;
          int iter=0;

          do{
            stop=1;
	    errorCode = clSetKernelArg(kernel_hooking, 5, sizeof(int), &iter); CHECKERROR;
	    errorCode = clSetKernelArg(kernel_hooking_cpu, 5, sizeof(int), &iter); CHECKERROR;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);
            clFinish(cmdQueue_cpu);//shan chu

            errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_hooking, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, kernel_hooking_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);
            clFinish(cmdQueue_cpu);

            errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueReadBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;

            iter++;
            
            errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_pointer_jumping, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
            errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

          //}while(iter==1);
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
          for(int i=0; i<vertex_cnt; i++)
            sumcomp+=rescoo2[i];
          printf("The number of components= %d iters= %d\n",sumcomp,iter);

          
          /*
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


            double testend = timestamp();
            time_in_sec += (testend - teststart);

          }


          //double time_in_sec = (testend - teststart)/(double)dim2;
          double gflops = (double)nnz*2*iter/(time_in_sec/(double)ntimes)/(double)1e9;
          printf("\nCSR vector SLM row ptr groupnum:%d cpu time %lf ms GFLOPS %lf code %d \n\n", groupnum,   time_in_sec / (double) ntimes * 1000, gflops, methodid);

          double onetime = time_in_sec / (double) ntimes;






        cl_event e[2];
          time_in_sec=0;
          for (int i = 0; i < ntimes; i++)
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

              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_hooking, work_dim, NULL, globalsize, blocksize, 0, NULL, &(e[0])); CHECKERROR;

//          double teststart = timestamp();
              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, kernel_hooking_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, &(e[1])); CHECKERROR;
  //          double testend = timestamp();
   //         time_in_sec += (testend - teststart);
              clFinish(cmdQueue);
              clFinish(cmdQueue_cpu);
              time_in_sec += get_time2(e[0],e[1],1);
 
              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueReadBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;

              iter++;

              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_pointer_jumping, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

            }while(!stop);
         }
          double onecputime = time_in_sec*1 / (double) ntimes;


          time_in_sec=0;
          for (int i = 0; i < ntimes; i++)
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

 //         double teststart = timestamp();
              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_hooking, work_dim, NULL, globalsize, blocksize, 0, NULL, &(e[0])); CHECKERROR;
//  double testend = timestamp();
            //time_in_sec += (testend - teststart);

              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, kernel_hooking_cpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, &(e[1])); CHECKERROR;
              clFinish(cmdQueue);
              clFinish(cmdQueue_cpu);
              time_in_sec += get_time2(e[0],e[1],0);

              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueReadBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;

              iter++;

              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_pointer_jumping, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
              errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel_update, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

            }while(!stop);
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
  costtime0=timestamp();//costtime
    if (argc < 3){
	printf("\nUsage: ./main *.mtx percent\n");
        return 1;
    }

    char* filename = argv[1];
    cpuoffset = atoi(argv[2]);
    //printf("cpuoffset=%d\n",cpuoffset);
    int ntimes = 1;
    //int ntimes = 20;

    coo_matrix<int, float> mat;
    init_coo_matrix(mat);
  costtime1=timestamp();//costtime
    ReadMMF(filename, &mat);
  costtime2=timestamp();//costtime


    //char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    
   {
	sprintf(clfilename, "%s%s", ".", "/spmv_csr_vector.cl");
	//sprintf(clfilename, "%s%s", "/home/pacman/zf/wubo/apu_corun/connectedComp_seperate", "/spmv_csr_vector.cl");
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

