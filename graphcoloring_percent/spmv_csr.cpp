#include "spmv_util.h"
#define MY_INFINITY    0xffffff00
#define TOTALNUMber 100
int cpuoffset;

    //spmv_csr_vector_ocl(&csrmat, vplistori, randlist, 0,  opttime1, optmethod1, clfilename, vplist, ntimes);//zf method1
void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, int* vplist, int* randlist, int padNum, double& opttime, int& optmethod, char* oclfilename, int* vplistres, int ntimes)
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
    cl_mem devrandlist;
    cl_mem devvplist;
    //cl_mem devData;
    //cl_mem devVec;
    //cl_mem devprnew;
    //cl_mem devprold;

    cl_mem devRowPtr_cpu;
    cl_mem devColId_cpu;
    cl_mem devrandlist_cpu;
    //cl_mem devData_cpu;
    //cl_mem devVec_cpu;


    cl_mem devover;

    //Initialize values
    int nnz = mat->matinfo.nnz;
    int vecsize = mat->matinfo.width;
    int rownum = mat->matinfo.height;
    int rowptrsize = rownum + 1;
    ALLOCATE_GPU_READ(devRowPtr, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ(devColId, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ(devrandlist, randlist, sizeof(int)*rownum);

    devvplist= clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(int)*rownum, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devvplist, CL_TRUE, 0, sizeof(int)*rownum, vplist, 0, NULL, NULL); CHECKERROR;
    devover= clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(char), NULL, &errorCode); CHECKERROR;

    ALLOCATE_GPU_READ_cpu(devRowPtr_cpu, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ_cpu(devColId_cpu, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ_cpu(devrandlist_cpu, randlist, sizeof(int)*rownum);

    opttime = 10000.0f;
    optmethod = 0;
    int dim2 = 1;


    {
	int methodid = 7;
        /*
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
        */


       
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
	cl_uint work_dim = 1;
	//cl_uint work_dim = 2;
	//int dim2 = 16;
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};
        //int threadsinrow=16;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_ve_slm_pm_fs", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtr); CHECKERROR;
	//errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devrandlist); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devvplist); CHECKERROR;
//	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int), &color); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devover); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(int), &rownum); CHECKERROR;


        /*
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devbitmap); CHECKERROR;


        errorCode = clSetKernelArg(csrKernel, 7, sizeof(cl_mem), &devprold); CHECKERROR;//added
        errorCode = clSetKernelArg(csrKernel, 8, sizeof(cl_mem), &devprnew); CHECKERROR;//added
        */


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

	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;
        cl_uint work_dimcpu=1;

        for (int groupnum = 240; groupnum <= 248; groupnum+= 24)
        //for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
        {
          //size_t globalsize[] = {totalsize, dim2};
          size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};

          errorCode = clEnqueueWriteBuffer(cmdQueue, devvplist, CL_TRUE, 0, sizeof(float)*rownum, vplist, 0, NULL, NULL); CHECKERROR;
          clFinish(cmdQueue);

          char stop=0;
          int curr=0;

          do{
            stop=1;
            errorCode = clEnqueueWriteBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernelcpu, 4, sizeof(int), &curr); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 4, sizeof(int), &curr); CHECKERROR;

            errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

            errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;

//            clFinish(cmdQueue);
            clFinish(cmdQueue_cpu);
            errorCode = clEnqueueReadBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);


            curr++;
            //printf("curr=%d, stop=%d\n",curr,stop);
          }while(!stop);




          int* tmpresult = (int*)malloc(sizeof(int)*rownum);
          errorCode = clEnqueueReadBuffer(cmdQueue, devvplist, CL_TRUE, 0, sizeof(int)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
          clFinish(cmdQueue);


//because I want to save time, so I remove it.//zf
          //two_vec_compare(vplistres, tmpresult, rownum);

          free(tmpresult);
          

          for (int k = 0; k < 0; k++)
          //for (int k = 0; k < 2; k++)
          {


            errorCode =clEnqueueWriteBuffer(cmdQueue, devvplist, CL_TRUE, 0, sizeof(float)*rownum, vplist, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);
            char stop=0;
            int curr=0;
            do{
              stop=1;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;
              errorCode = clSetKernelArg(csrKernelcpu, 4, sizeof(int), &curr); CHECKERROR;
              errorCode = clSetKernelArg(csrKernel, 4, sizeof(int), &curr); CHECKERROR;

              errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;

              //           clFinish(cmdQueue);
              clFinish(cmdQueue_cpu);
              errorCode = clEnqueueReadBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;
              clFinish(cmdQueue);

              curr++;
              //printf("curr=%d, stop=%d\n",curr,stop);
            }while(!stop);



          }

          double time_in_sec=0;
          for (int i = 0; i < ntimes; i++)
          {

            errorCode =clEnqueueWriteBuffer(cmdQueue, devvplist, CL_TRUE, 0, sizeof(float)*rownum, vplist, 0, NULL, NULL); CHECKERROR;
            clFinish(cmdQueue);
            char stop=0;
            int curr=0;
          double teststart = timestamp();
            do{
              stop=1;
              errorCode = clEnqueueWriteBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;
              errorCode = clSetKernelArg(csrKernelcpu, 4, sizeof(int), &curr); CHECKERROR;
              errorCode = clSetKernelArg(csrKernel, 4, sizeof(int), &curr); CHECKERROR;

              errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

              errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;

              //           clFinish(cmdQueue);
              clFinish(cmdQueue_cpu);
              errorCode = clEnqueueReadBuffer(cmdQueue, devover, CL_TRUE, 0, sizeof(char), &stop, 0, NULL, NULL); CHECKERROR;
              clFinish(cmdQueue);

              curr++;
              //printf("curr=%d, stop=%d\n",curr,stop);
            }while(!stop);
          double testend = timestamp();
          time_in_sec += (testend - teststart);

          }


          //double time_in_sec = (testend - teststart)/(double)dim2;
          double gflops = (double)nnz*2*curr/(time_in_sec/(double)ntimes)/(double)1e9;
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
        if (csrKernel)
          clReleaseKernel(csrKernel);
        //free(rowptrpad);


        //free(bitmap);free(rowforcpu);//zf
        //	clReleaseMemObject(devbitmap);//zf
        }



        if (devRowPtr)
          clReleaseMemObject(devRowPtr);
        if (devColId)
          clReleaseMemObject(devColId);
        /*    if (devData)
              clReleaseMemObject(devData);
              if (devVec)
              clReleaseMemObject(devVec);
              if (devRes)
              clReleaseMemObject(devRes);
              */

        freeObjects(devices, &context, &cmdQueue, &program);


        return;
    }


int main(int argc, char* argv[])
{
    if (argc < 3){
	printf("\nUsage: ./main *.mtx percentage(such as 40)\n");
        return 1;
    }

    char* filename = argv[1];
    cpuoffset = atoi(argv[2]);
    printf("cpuoffset=%d\n",cpuoffset);
    int ntimes = 1;
    //int ntimes = 20;

    coo_matrix<int, float> mat;
    init_coo_matrix(mat);
    ReadMMF(filename, &mat);


    //char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    
   {
	sprintf(clfilename, "%s%s", "/home/pacman/zf/wubo/apu_corun/graphcoloring_percent", "/spmv_csr_vector.cl");
    printMatInfo(&mat);


    if(mat.matinfo.width != mat.matinfo.height){
      printf("width != height\n");
      exit(-1);
    }



    csr_matrix<int, float> csrmat;
    coo2csr<int, float>(&mat, &csrmat);


////////////////////////////////////////////////////////////////////////////////////////////////////////
    int vertex_cnt=mat.matinfo.height;
    int* vplist=(int*)malloc(sizeof(int)*vertex_cnt);
    int* vplistori=(int*)malloc(sizeof(int)*vertex_cnt);
    int* randlist=(int*)malloc(sizeof(int)*vertex_cnt);
    printf("vertexcnt=%d\n",vertex_cnt);
//    char* h_over=(char*)malloc(sizeof(bool));
    for(int i=0; i<vertex_cnt; i++){
      vplist[i]=MY_INFINITY;
      vplistori[i]=MY_INFINITY;
      randlist[i]=rand();
    }
    int curr=0, color;
    char stop=0;
    do{
      stop=1;
      color=curr;
      for(int tid=0; tid<vertex_cnt; tid++){
        if(vplist[tid]!=MY_INFINITY)
          continue;
        int vid=tid;
        int start=csrmat.csr_row_ptr[vid];
        int end=csrmat.csr_row_ptr[vid+1];
//        if(vid==4)
 //         printf("start=%d, end=%d\n",start,end);

        int local_rand=randlist[vid];
        char found_larger=0;
        for(int i=start; i<end; i++){
          int dest = csrmat.csr_col_id[i];
          if((vplist[dest]<color)&&(vplist[dest]>=0))
            continue;
          if( (randlist[dest]>local_rand)  ||
              ( (randlist[dest]==local_rand) && (dest<vid))){
            found_larger = 1;
            //if(vid==0)
              //printf("f=%d\n",found_larger );
            break;
          }
        }
        if(found_larger==0)
          vplist[vid]=color;
        else
          stop=0;
      }
      curr++;
    }while(!stop);
    printf("cpu color=%d\n",curr);
 //   for(int i=0; i<10; i++)
  //    printf("vplist[%d]=%d\trand[%d]=%d\n",i,vplist[i],i,randlist[i]);
//    exit(0);









////////////////////////////////////////////////////////////////////////////////////////////////////////


    double opttime1 = 100000.0f;
    int optmethod1 = 0;

    spmv_csr_vector_ocl(&csrmat, vplistori, randlist, 0,  opttime1, optmethod1, clfilename, vplist, ntimes);//zf method1
    //spmv_csr_vector_ocl(&csrmat, vec, res, 0,  opttime1, optmethod1, clfilename, coores, ntimes);//zf method1

	double opttime2 = 100000.0f;
	int optmethod2 = 0;

	csr_matrix<int, float> padcsr;
	pad_csr(&csrmat, &padcsr, WARPSIZE / 2);
	printf("\nNNZ Before %d After %d\n", csrmat.matinfo.nnz, padcsr.matinfo.nnz);
        //I remove it because of saving time
        //spmv_csr_vector_ocl(&padcsr, vplistori, randlist, 0,  opttime2, optmethod2, clfilename, vplist, ntimes);//zf method1
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
    free(vplist);
    free(vplistori);
    free(randlist);


    }

    free_coo_matrix(mat);

    return 0;
}

