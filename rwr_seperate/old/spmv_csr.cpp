#include "spmv_util.h"


void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, float* vec, float* result, int padNum, double& opttime, int& optmethod, char* oclfilename, float* coores, int ntimes)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devRowPtr;
    cl_mem devColId;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devTexVec;
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
    int paddedres = findPaddedSize(rownum, 16);
    devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;
    //errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;

	const cl_image_format floatFormat =
	{
	    CL_R,
	    CL_FLOAT,
	};
		
	int width = VEC2DWIDTH;
	int height = (vecsize + VEC2DWIDTH - 1)/VEC2DWIDTH;
	float* image2dVec = (float*)malloc(sizeof(float)*width*height);
	memset(image2dVec, 0, sizeof(float)*width*height);
	for (int i = 0; i < vecsize; i++)
	{
	    image2dVec[i] = vec[i];
	}
	size_t origin[] = {0, 0, 0};
	size_t vectorSize[] = {width, height, 1};
	devTexVec = clCreateImage2D(context, CL_MEM_READ_ONLY, &floatFormat, width, height, 0, NULL, &errorCode); CHECKERROR;
	errorCode = clEnqueueWriteImage(cmdQueue, devTexVec, CL_TRUE, origin, vectorSize, 0, 0, image2dVec, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);


    opttime = 10000.0f;
    optmethod = 0;
    int dim2 = 1;


    {
	int methodid = 7;
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, 8);
	int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	for (int i = 0; i <= mat->matinfo.height; i++)
	    rowptrpad[i] = mat->csr_row_ptr[i];


	ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));
	clFinish(cmdQueue);
       
//////////////////////////////////////////////////////////////////////////////////////

	int* rowptrpad_new = (int*)malloc(sizeof(int)*(padrowsize+1));
        int *ColId_new= (int*)malloc(sizeof(int)*(nnz));
        float *Data_new= (float*)malloc(sizeof(float)*(nnz));

	int* rownumzf = (int*)malloc(sizeof(int)*(padrowsize));
	int* rowno = (int*)malloc(sizeof(int)*(padrowsize));

        int count2[9];
        memset(count2,0,9*sizeof(int));

        for(int i=0 ; i<padrowsize; i++){
          rownumzf[i]=rowptrpad[i+1]-rowptrpad[i];
          rowno[i]=i;
          if(rownumzf[i]<=1)
            count2[0]+=1;
          else if(rownumzf[i]<=2)
            count2[1]+=1;
          else if(rownumzf[i]<=4)
            count2[2]+=1;
          else if(rownumzf[i]<=8)
            count2[3]+=1;
          else if(rownumzf[i]<=16)
            count2[4]+=1;
          else if(rownumzf[i]<=32)
            count2[5]+=1;
          else if(rownumzf[i]<=64)
            count2[6]+=1;
          else if(rownumzf[i]<=128)
            count2[7]+=1;
          else
            count2[8]+=1;
        }
        printf("[0,1]: %d (%f %)\n",count2[0],(float)(count2[0])/(float)(padrowsize)*100);
        printf("(1,2]: %d (%f %)\n",count2[1],(float)(count2[1])/(float)(padrowsize)*100);
        printf("(2,4]: %d (%f %)\n",count2[2],(float)(count2[2])/(float)(padrowsize)*100);
        printf("(4,8]: %d (%f %)\n",count2[3],(float)(count2[3])/(float)(padrowsize)*100);
        printf("(8,16]: %d (%f %)\n",count2[4],(float)(count2[4])/(float)(padrowsize)*100);
        printf("(16,32]: %d (%f %)\n",count2[5],(float)(count2[5])/(float)(padrowsize)*100);
        printf("(32,64]: %d (%f %)\n",count2[6],(float)(count2[6])/(float)(padrowsize)*100);
        printf("(64,128]: %d (%f %)\n",count2[7],(float)(count2[7])/(float)(padrowsize)*100);
        printf("(128,MAX]: %d (%f %)\n",count2[8],(float)(count2[8])/(float)(padrowsize)*100);
 //       exit(-1);
        for(int i=0; i<padrowsize-1; i++){
          for(int j=i+1; j<padrowsize; j++){
            if(rownumzf[j]>rownumzf[i]){

              int tmp1=rownumzf[j];
              rownumzf[j]=rownumzf[i];
              rownumzf[i]=tmp1;

              tmp1=rowno[i];
              rowno[i]=rowno[j];
              rowno[j]=tmp1;
            }
          }
        }
        rowptrpad_new[0]=0;
        //for(int i=0; i<height; i++){
        for(int i=0; i<padrowsize; i++){
          rowptrpad_new[i+1]=rowptrpad_new[i]+rownumzf[i];
        }
        int numbernew=0;
        //for(int i=0; i<padrowsize; i++){
        for(int i=0; i<mat->matinfo.height; i++){
          int startzf=mat->csr_row_ptr[rowno[i]];
          int endzf=mat->csr_row_ptr[rowno[i]+1];
          for(int j=startzf; j<endzf; j++){
            ColId_new[numbernew]=mat->csr_col_id[j];
            Data_new[numbernew]= mat->csr_data[j];
            numbernew++;
          }

        }
        printf("number = %d, height=%d\n",numbernew, mat->matinfo.height);
     //   for(int i=0; i<padrowsize; i++)
   //     printf("row: %d num: %d\n",rowno[i],rownumzf[i]);

    clEnqueueWriteBuffer(cmdQueue, devRowPtrPad, CL_TRUE, 0, sizeof(int)*rowptrsize, rowptrpad_new, 0, NULL, NULL); 
    if( errorCode != CL_SUCCESS ) { printf("Error: clCreateBuffer returned %d\n", errorCode); } 
 
    clEnqueueWriteBuffer(cmdQueue, devColId, CL_TRUE, 0, sizeof(int)*nnz, ColId_new, 0, NULL, NULL); 
    if( errorCode != CL_SUCCESS ) { printf("Error: clCreateBuffer returned %d\n", errorCode); } 
    clEnqueueWriteBuffer(cmdQueue, devData, CL_TRUE, 0, sizeof(int)*nnz, Data_new, 0, NULL, NULL); 
    if( errorCode != CL_SUCCESS ) { printf("Error: clCreateBuffer returned %d\n", errorCode); } 

       
	cl_mem devrowno;
	ALLOCATE_GPU_READ(devrowno, rowno, sizeof(int)*(padrowsize));
//////////////////////////////////////////////////////////////////////////////////////



	printf("\nRow Num %d padded size %d\n", rownum, padrowsize);
	cl_uint work_dim = 1;
	//cl_uint work_dim = 2;
	//int dim2 = 16;
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};
        int threadsinrow=16;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_ve_slm_pm_fs", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devrowno); CHECKERROR;
	//errorCode = clSetKernelArg(csrKernel, 7, sizeof(int), &threadsinrow); CHECKERROR;


	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;
	
	for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
	{
	    //size_t globalsize[] = {totalsize, dim2};
	    size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};

	    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);



	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;



	    clFinish(cmdQueue);
	    float* tmpresult = (float*)malloc(sizeof(float)*rownum);
	    errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);
	    two_vec_compare(coores, tmpresult, rownum);
	    free(tmpresult);

	    for (int k = 0; k < 3; k++)
	    {
		errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	    }
	    clFinish(cmdQueue);

	    double teststart = timestamp();
	    for (int i = 0; i < ntimes; i++)
	    {
		errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	    }
	    clFinish(cmdQueue);
	    double testend = timestamp();
	    double time_in_sec = (testend - teststart)/(double)dim2;
	    double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
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


    }

        //exit(-1);
        return;
    
    
   
    



    //Clean up
    if (image2dVec)
	free(image2dVec);

    if (devRowPtr)
	clReleaseMemObject(devRowPtr);
    if (devColId)
	clReleaseMemObject(devColId);
    if (devData)
	clReleaseMemObject(devData);
    if (devVec)
	clReleaseMemObject(devVec);
    if (devTexVec)
	clReleaseMemObject(devTexVec);
    if (devRes)
	clReleaseMemObject(devRes);

    freeObjects(devices, &context, &cmdQueue, &program);

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
	sprintf(clfilename, "%s%s", "/home/pacman/zf/wubo/apu_corun/zfcsr", "/spmv_csr_vector.cl");
    printMatInfo(&mat);
    csr_matrix<int, float> csrmat;
    coo2csr<int, float>(&mat, &csrmat);
    float* vec = (float*)malloc(sizeof(float)*mat.matinfo.width);
    float* res = (float*)malloc(sizeof(float)*mat.matinfo.height);
    initVectorOne<int, float>(vec, mat.matinfo.width);	
    initVectorZero<int, float>(res, mat.matinfo.height);
    float* coores = (float*)malloc(sizeof(float)*mat.matinfo.height);
    spmv_only(&mat, vec, coores);
    double opttime1 = 10000.0f;
    int optmethod1 = 0;

    spmv_csr_vector_ocl(&csrmat, vec, res, 0,  opttime1, optmethod1, clfilename, coores, ntimes);
    //spmv_csr_vector_ocl(&csrmat, vec, res, 0, dim2Size, opttime1, optmethod1, clfilename, deviceType, coores, ntimes);

	double opttime2 = 10000.0f;
	int optmethod2 = 0;

	csr_matrix<int, float> padcsr;
	pad_csr(&csrmat, &padcsr, WARPSIZE / 2);
	printf("\nNNZ Before %d After %d\n", csrmat.matinfo.nnz, padcsr.matinfo.nnz);
	spmv_csr_vector_ocl(&padcsr, vec, res, 16, opttime2, optmethod2, clfilename, coores, ntimes);
	free_csr_matrix(padcsr);

	int nnz = mat.matinfo.nnz;
	double gflops = (double)nnz*2/opttime1/(double)1e9;
	printf("\n------------------------------------------------------------------------\n");
	printf("CSR VEC without padding best time %f ms best method %d gflops %f", opttime1*1000.0, optmethod1, gflops);
	printf("\n------------------------------------------------------------------------\n");
	gflops = (double)nnz*2/opttime2/(double)1e9;
	printf("CSR VEC with padding best time %f ms best method %d gflops %f", opttime2*1000.0, optmethod2, gflops);
	printf("\n------------------------------------------------------------------------\n");
  

    free(vec);
    free(res);
    free_csr_matrix(csrmat);
    free(coores);


    }

    free_coo_matrix(mat);

    return 0;
}

