#include "spmv_util.h"


void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, int* resin, int* resout, int padNum, double& opttime, int& optmethod, char* oclfilename, int* res0, int ntimes)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_command_queue cmdQueue_cpu = NULL;
    cl_program program = NULL;

    assert(initialization2(devices, &context, &cmdQueue, &program, oclfilename, &cmdQueue_cpu) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Initialize values
    int nnz = mat->matinfo.nnz;
    int vecsize = mat->matinfo.width;
    int rownum = mat->matinfo.height;
    int rowptrsize = rownum + 1;

    //Create device memory objects
    cl_mem devRowPtr;
    cl_mem devColId;
    cl_mem devresin;
    cl_mem devresout;

    ALLOCATE_GPU_READ(devRowPtr, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ(devColId, mat->csr_col_id, sizeof(int)*nnz);

    devresin = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*rownum, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devresin, CL_TRUE, 0, sizeof(int)*rownum, res0, 0, NULL, NULL); CHECKERROR;
    devresout = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*rownum, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devresout, CL_TRUE, 0, sizeof(int)*rownum, res0, 0, NULL, NULL); CHECKERROR;






      
//////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////



	cl_uint work_dim = 1;
	//cl_uint work_dim = 2;
	//int dim2 = 16;
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};
//        int threadsinrow=16;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "dckernel", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devresin); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devresout); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int), &rownum); CHECKERROR;




	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;
        cl_uint work_dimcpu=1;
	
	for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
	{
	    //size_t globalsize[] = {totalsize, dim2};
	    size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, 1};

	    errorCode = clEnqueueWriteBuffer(cmdQueue, devresin, CL_TRUE, 0, sizeof(int)*rownum, res0, 0, NULL, NULL); CHECKERROR;
	    errorCode = clEnqueueWriteBuffer(cmdQueue, devresout, CL_TRUE, 0, sizeof(int)*rownum, res0, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);

	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;

	    clFinish(cmdQueue);
//	    clFinish(cmdQueue_cpu);
	    int* tmpresultin = (int*)malloc(sizeof(int)*rownum);
	    int* tmpresultout = (int*)malloc(sizeof(int)*rownum);
	    errorCode = clEnqueueReadBuffer(cmdQueue, devresin, CL_TRUE, 0, sizeof(int)*rownum, tmpresultin, 0, NULL, NULL); CHECKERROR;
	    errorCode = clEnqueueReadBuffer(cmdQueue, devresout, CL_TRUE, 0, sizeof(int)*rownum, tmpresultout, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);

            printf("in check:\n");
	    two_vec_compare(resin, tmpresultin, rownum);
            printf("out check:\n");
	    two_vec_compare(resout, tmpresultout, rownum);
	    free(tmpresultin);
	    free(tmpresultout);

	    for (int k = 0; k < 3; k++)
	    {
		errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);
	    }

	    double teststart = timestamp();
	    for (int i = 0; i < ntimes; i++)
	    {
		errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);
	    }
	    double testend = timestamp();
	    double time_in_sec = (testend - teststart);
	    double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
            int methodid=0;
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

//	if (devRowPtrPad)
//	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	//free(rowptrpad);



    
    
   
    



    //Clean up
    
    if (devColId)
	clReleaseMemObject(devColId);
    if (devresin)
	clReleaseMemObject(devresin);
    if (devresout)
	clReleaseMemObject(devresout);

    freeObjects(devices, &context, &cmdQueue, &program);


        return;
}


int main(int argc, char* argv[])
{
    if (argc < 2){
	printf("\nUsage: ./main *.mtx\n");
        return 1;
    }

    char* filename = argv[1];
    int ntimes = 20;

    coo_matrix<int, float> mat;
    init_coo_matrix(mat);
    ReadMMF(filename, &mat);

    //char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    
   {
	sprintf(clfilename, "%s%s", "/home/pacman/zf/wubo/apu_corun/dc_ori/", "/spmv_csr_vector.cl");
    printMatInfo(&mat);
    csr_matrix<int, float> csrmat;
    coo2csr<int, float>(&mat, &csrmat);

    if(mat.matinfo.width != mat.matinfo.height){
      printf("width != height\n");
      exit(-1);
    }

 
    int* resin = (int *)malloc(sizeof(int)*mat.matinfo.width);
    int* resout = (int *)malloc(sizeof(int)*mat.matinfo.width);
    int* res0 = (int *)malloc(sizeof(int)*mat.matinfo.width);
    initVectorZero<int, int>(resin, mat.matinfo.width);	
    initVectorZero<int, int>(resout, mat.matinfo.width);	
    initVectorZero<int, int>(res0, mat.matinfo.width);	

    for(int row=0; row<mat.matinfo.height; row++){
        int start = csrmat.csr_row_ptr[row];
        int end = csrmat.csr_row_ptr[row+1];
        resout[row]=end-start;
        for(int j=start; j<end; j++){
          int col=csrmat.csr_col_id[j];
          resin[col]++;
        }
    }

    int optmethod1 = 0;
    double opttime1 = 10000.0f;
    spmv_csr_vector_ocl(&csrmat, resin, resout, 0,  opttime1, optmethod1, clfilename, res0, ntimes);//zf method1




    /*
    float* vec = (float*)malloc(sizeof(float)*mat.matinfo.width);
    float* res = (float*)malloc(sizeof(float)*mat.matinfo.height);
    initVectorOne<int, float>(vec, mat.matinfo.width);	
    initVectorZero<int, float>(res, mat.matinfo.height);
    float* coores = (float*)malloc(sizeof(float)*mat.matinfo.height);
    spmv_only(&mat, vec, coores);
    double opttime1 = 10000.0f;
    int optmethod1 = 0;

    spmv_csr_vector_ocl(&csrmat, vec, res, 0,  opttime1, optmethod1, clfilename, coores, ntimes);//zf method1

	double opttime2 = 10000.0f;
	int optmethod2 = 0;

	csr_matrix<int, float> padcsr;
	pad_csr(&csrmat, &padcsr, WARPSIZE / 2);
	printf("\nNNZ Before %d After %d\n", csrmat.matinfo.nnz, padcsr.matinfo.nnz);
	spmv_csr_vector_ocl(&padcsr, vec, res, 16, opttime2, optmethod2, clfilename, coores, ntimes);
	free_csr_matrix(padcsr);
        */

	int nnz = mat.matinfo.nnz;
	double gflops = (double)nnz*2/opttime1/(double)1e9;
	printf("\n------------------------------------------------------------------------\n");
	printf("CSR VEC without padding best time %f ms best method %d gflops %f", opttime1*1000.0, optmethod1, gflops);
	printf("\n------------------------------------------------------------------------\n");
        /*
	gflops = (double)nnz*2/opttime2/(double)1e9;
	printf("CSR VEC with padding best time %f ms best method %d gflops %f", opttime2*1000.0, optmethod2, gflops);
	printf("\n------------------------------------------------------------------------\n");
  

    free(vec);
    free(res);
        */
    free_csr_matrix(csrmat);
    //free(coores);


    }

    free_coo_matrix(mat);

    return 0;
}

