#include "spmv_util.h"
int cpuoffset;


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
	float* image2dVec = (float*)malloc(sizeof(float)*width*height);
	memset(image2dVec, 0, sizeof(float)*width*height);
	for (int i = 0; i < vecsize; i++)
	{
	    image2dVec[i] = vec[i];
	}
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
       
//////////////////////////////////////////////////////////////////////////////////////

        printf("sizeofulong=%d bytes",sizeof(unsigned long));
        unsigned long *bitmap=(unsigned long*)malloc(sizeof(unsigned long)*(((padrowsize)>>6) + 1));
	int* rowforcpu = (int*)malloc(sizeof(int)*(padrowsize));
        int rowforcpusum=0;
        memset(bitmap,0,(sizeof(unsigned long)*(((padrowsize)>>6) + 1)));

        for(int i=0 ; i<padrowsize; i++){
          int numi=rowptrpad[i+1]-rowptrpad[i];
          //if(numi>=16){
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
/*

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
        exit(-1);
 
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
        */
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

        int rowsetzf=(double)cpuoffset/100*rownum;//zf
        printf("rowsetzf=%d\n",rowsetzf);
        errorCode = clSetKernelArg(csrKernel, 7, sizeof(int), &rowsetzf); CHECKERROR;




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
        
	errorCode = clSetKernelArg(csrKernelcpu, 5, sizeof(int), &rowforcpu); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 6, sizeof(cl_mem), &devrowforcpu); CHECKERROR;
	
	errorCode = clSetKernelArg(csrKernelcpu, 7, sizeof(int), &rowforcpusum); CHECKERROR;//newadded

        errorCode = clSetKernelArg(csrKernelcpu, 8, sizeof(int), &rowsetzf); CHECKERROR;
	errorCode = clSetKernelArg(csrKernelcpu, 9, sizeof(int), &rownum); CHECKERROR;


	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;
        cl_uint work_dimcpu=1;
	
	for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
	{
	    //size_t globalsize[] = {totalsize, dim2};
	    size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};

	    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);



	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	    errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;



	    clFinish(cmdQueue);
	    clFinish(cmdQueue_cpu);
	    float* tmpresult = (float*)malloc(sizeof(float)*rownum);
	    errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
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
		errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	    errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);
	    clFinish(cmdQueue_cpu);

	    }
	    double teststart = timestamp();
	    for (int i = 0; i < ntimes; i++)
	    {
		errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	    errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);
	    clFinish(cmdQueue_cpu);
	    }
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


        free(bitmap);free(rowforcpu);//zf
	clReleaseMemObject(devbitmap);//zf
    }

        //exit(-1);
    
    
   
    



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
    if (devRes)
	clReleaseMemObject(devRes);

    freeObjects(devices, &context, &cmdQueue, &program);


        return;
}


int main(int argc, char* argv[])
{
    if (argc < 3){
	printf("\nUsage: ./main *.mtx percent\n");
        return 1;
    }

    char* filename = argv[1];

        cpuoffset = atoi(argv[2]);
            printf("cpuoffset=%d\n",cpuoffset);

    int ntimes = 5;

    coo_matrix<int, float> mat;
    init_coo_matrix(mat);
    ReadMMF(filename, &mat);

    //char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    
   {
	sprintf(clfilename, "%s%s", "/home/pacman/zf/wubo/apu_corun/zfcsr_percent_v2", "/spmv_csr_vector.cl");
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

    spmv_csr_vector_ocl(&csrmat, vec, res, 0,  opttime1, optmethod1, clfilename, coores, ntimes);//zf method1

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
  
        double optfinal=opttime1;
        if(opttime2<optfinal)
          optfinal=opttime2;
        printf("CAUTTION: kernel time(ms): %f\n",optfinal*1000.0);
 
    free(vec);
    free(res);
    free_csr_matrix(csrmat);
    free(coores);


    }

    free_coo_matrix(mat);

    return 0;
}

