#include<stdio.h>
#include"mmio.h"
#include"common.h"

int cpuoffset;

////////////////////////////////////////////////////////
static const int PAD_FACTOR = 16;

template <typename floatType>
void convertToColMajor(floatType *A, int *cols, int dim, int *rowDelimiters,
                       floatType *newA, int *newcols, int *rl, int maxrl,
                       bool padded)
{
    int pad = 0;
    if (padded && dim % PAD_FACTOR != 0)
    {   
        pad = PAD_FACTOR - dim % PAD_FACTOR;
    }   

    int newIndex = 0;
    for (int j=0; j<maxrl; j++)
    {   
        for (int i=0; i<dim; i++)
        {
            if (rowDelimiters[i] + j < rowDelimiters[i+1])
            {
                newA[newIndex] = A[rowDelimiters[i]+j];
                newcols[newIndex] = cols[rowDelimiters[i]+j];
            }
            else
            {
                newA[newIndex] = 0;
            }
            newIndex++;
        }
        if (padded)
        {
            for (int p=0; p<pad; p++)
            {
                newA[newIndex] = 0;
                newIndex++;
            }
        }
    }   
}

#define MAX_SOURCE_SIZE (0x100000)
#define CL_CHECK_ERROR(err) \
      {                       \
                if (err != CL_SUCCESS)                  \
                { \
                              std::cerr << __FILE__ << ':' << __LINE__ << " "<<err << std::endl; \
                              exit(1); \
                          } \
            }

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;
static const double MAX_RELATIVE_ERROR = .02;


char str_temp[1024];

cl_platform_id platform_id;
cl_device_id device_id[2];   
//cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel;
cl_command_queue clCommandQue[2];
//cl_command_queue clCommandQue;
cl_program clProgram;
static const int BLOCK_SIZE = 64;

//cl_mem a_mem_obj;
//cl_mem b_mem_obj;
//cl_mem c_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}



void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("spmv.cl", "r");
	if (!fp) {
		fprintf(stdout, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void cl_initialization_fusion()
{
	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	errcode = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id[0], &num_devices);
	if(errcode == CL_SUCCESS) printf("number of GPU is %d\n", num_devices);
	errcode |= clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id[1], &num_devices);
	if(errcode == CL_SUCCESS) printf("number of CPU is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	errcode = clGetDeviceInfo(device_id[0],CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("GPU device name is %s\n",str_temp);
	else printf("GPU Error getting device name\n");
	errcode = clGetDeviceInfo(device_id[1],CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("CPU device name is %s\n",str_temp);
	else printf("CPU Error getting device name\n");
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 2, device_id, NULL, NULL, &errcode);
	//clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue[0] = clCreateCommandQueue(clGPUContext, device_id[0], 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
	clCommandQue[1] = clCreateCommandQueue(clGPUContext, device_id[1], 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");

}


void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 0, 0, NULL, NULL, NULL);
	//errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
//	clFinish(clCommandQue);
}



void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue[0]);
	errcode = clFlush(clCommandQue[1]);
	errcode = clFinish(clCommandQue[0]);
	errcode = clFinish(clCommandQue[1]);

 
	errcode = clReleaseCommandQueue(clCommandQue[0]);
	errcode = clReleaseCommandQueue(clCommandQue[1]);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");

}


/////////////////////////////////////////////////////////
template <typename floatType, typename clFloatType>
int call_bhsparse(const char *datasetpath)
{
    int err = 0;

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    value_type *csrValA;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if ((f = fopen(datasetpath, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        cout << "Could not process Matrix Market banner.\n";
        return -2;
    }

    if ( mm_is_complex( matcode ) )
    {
        cout <<"Sorry, data type 'COMPLEX' is not supported.\n";
        return -3;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*cout << "type = Pattern\n";*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*cout << "type = real\n";*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*cout << "type = integer\n";*/ }

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report)) != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        cout << "symmetric = true" << endl;
    }
    else
    {
        cout << "symmetric = false" << endl;
    }

    int *csrRowPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    value_type *csrValA_tmp    = (value_type *)malloc(nnzA_mtx_report * sizeof(value_type));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;

        if (isReal)
            int count = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            int count = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            int count = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    printf("nnzA=%d, m=%d, csrRowPtrA_counter[m]=%d\n",nnzA,m,csrRowPtrA_counter[m]);
    csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrValA    = (value_type *)malloc(nnzA * sizeof(value_type));

    double gb = (double)((m + 1 + nnzA) * sizeof(int) + (2 * nnzA + m) * sizeof(value_type));
    double gflop = (double)(2 * nnzA);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);

    cout << " ( " << m << ", " << n << " ) nnz = " << nnzA << endl;
    
    for (int i = 0; i < nnzA; i++)
    {
        csrValA[i] = 1; //rand() % 10;
    }

    value_type *x = (value_type *)malloc(n * sizeof(value_type));
    
    for (int i = 0; i < n; i++)
        x[i] = 1; //rand() % 10;

    value_type *y = (value_type *)malloc(m * sizeof(value_type));
    value_type *y_ref = (value_type *)malloc(m * sizeof(value_type));

    // compute cpu results
    bhsparse_timer ref_timer;
    ref_timer.start();

    for (int iter = 0; iter < NUM_RUN; iter++)
    {
        for (int i = 0; i < m; i++)
        {
            value_type sum = 0;
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j];
            y_ref[i] = sum;
        }
    }
    
    double ref_time = ref_timer.stop() / (double)NUM_RUN;

    cout << "\ncpu sequential time = " << ref_time
         << " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  << " GFlops." << endl << endl;

    memset((void *)y, 0, m * sizeof(value_type));


//testruntest
    {
	read_cl_file();
	cl_initialization_fusion();
	cl_load_prog();



    int nItems=nnzA ;            // number of non-zero elements in the matrix
    int numRows=m;           // number of rows in the matrix

    int paddedSize = numRows + (PAD_FACTOR - numRows % PAD_FACTOR);
    float *h_out = new floatType[paddedSize];

    int *h_rowLengths = new int[paddedSize];
    int maxrl = 0;
    for (int k=0; k<numRows; k++)
    {
        h_rowLengths[k] = csrRowPtrA[k+1] - csrRowPtrA[k];
        if (h_rowLengths[k] > maxrl)
        {
            maxrl = h_rowLengths[k];
        }
    }
    for (int p=numRows; p < paddedSize; p++)
    {
        h_rowLengths[p] = 0;
    }

    // Column major format host data structures
    int cmSize = numRows;
    printf("maxrl= %d  cmSize= %d\n",maxrl, cmSize);
    bool padded=false;
    //int cmSize = padded ? paddedSize : numRows;
    floatType *h_valcm = new floatType[maxrl * cmSize];
    int *h_colscm = new int[maxrl * cmSize];
    convertToColMajor<float>(csrValA, csrColIdxA, numRows, csrRowPtrA, h_valcm,
                              h_colscm, h_rowLengths, maxrl, padded);

//intothe ellpackfunction
    int err;



//////////////////////////////////////////////////////////////////////////////////////


        unsigned long *bitmap=(unsigned long*)malloc(sizeof(unsigned long)*(((cmSize)>>6) + 1));
	int* rowforcpu = (int*)malloc(sizeof(int)*(cmSize));
        int rowforcpusum=0;
        memset(bitmap,0,(sizeof(unsigned long)*(((cmSize)>>6) + 1)));

        for(int i=0 ; i<cmSize; i++){
          int numi=h_rowLengths[i];
          //if(numi>=16){
          if(numi<128){
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
          if(rowforcpusum==0){
 //           printf("Sorry, do not need CPU\n");
//            exit(-1);
rowforcpusum=1;
          }

 	cl_mem devbitmap;
    devbitmap= clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned long)*(((cmSize)>>6)+1), NULL, &err);
    CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(clCommandQue[0], devbitmap, true, 0, sizeof(unsigned long)*(((cmSize)>>6)+1), bitmap, 0, NULL, NULL);
    CL_CHECK_ERROR(err);

	cl_mem devrowforcpu;
    devrowforcpu= clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(int)*(rowforcpusum), NULL, &err);
    CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(clCommandQue[0], devrowforcpu, true, 0, sizeof(int)*(rowforcpusum), rowforcpu, 0, NULL, NULL);
    CL_CHECK_ERROR(err);






//////////////////////////////////////////////////////////////////////////////////////





    // Device data structures
    cl_mem d_val, d_vec, d_out; // floating point
    cl_mem d_cols, d_rowLengths; // integer


    cl_mem d_val_cpu, d_vec_cpu ; // floating point
    cl_mem d_cols_cpu, d_rowLengths_cpu; // integer


    // Device data structures
    //cl_mem d_val, d_vec, d_out;
    //cl_mem d_cols, d_rowDelimiters;
    //int err;

    // Allocate device memory
    d_val = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, maxrl * cmSize *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_cols = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, maxrl * cmSize *
        sizeof(int), NULL, &err);
    CL_CHECK_ERROR(err);
    d_vec = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, numRows *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_out = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, paddedSize *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_rowLengths = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, cmSize *
        sizeof(int), NULL, &err);
    CL_CHECK_ERROR(err);


    // Transfer data to device
    err = clEnqueueWriteBuffer(clCommandQue[0], d_val, true, 0, maxrl * cmSize *
        sizeof(clFloatType), h_valcm, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(clCommandQue[0], d_cols, true, 0, maxrl * cmSize *
        sizeof(cl_int), h_colscm, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clCommandQue[0], d_vec, true, 0, numRows *
            sizeof(clFloatType), x, 0, NULL, NULL);
        CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(clCommandQue[0], d_rowLengths, true, 0, cmSize *
        sizeof(int), h_rowLengths, 0, NULL, NULL);
    CL_CHECK_ERROR(err);


/////////////////////////////////////CPU Side////////////////////////

    floatType *h_valcm_trans = new floatType[maxrl * cmSize];
    int *h_colscm_trans = new int[maxrl * cmSize];
    for(int i=0; i<cmSize; i++){
      for(int j=0; j<maxrl; j++){
        h_valcm_trans[i*maxrl+j]=h_valcm[i+j*cmSize];
        h_colscm_trans[i*maxrl+j]=h_colscm[i+j*cmSize];
      }   
    }   



    // Allocate device memory
    d_val_cpu = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, maxrl * cmSize *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_cols_cpu = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, maxrl * cmSize *
        sizeof(int), NULL, &err);
    CL_CHECK_ERROR(err);
    d_vec_cpu = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, numRows *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_out = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, paddedSize *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_rowLengths_cpu = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, cmSize *
        sizeof(int), NULL, &err);
    CL_CHECK_ERROR(err);


    // Transfer data to device
    err = clEnqueueWriteBuffer(clCommandQue[0], d_val_cpu, true, 0, maxrl * cmSize *
        sizeof(clFloatType), h_valcm_trans, 0, NULL, NULL);//zfadded
    CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(clCommandQue[0], d_cols_cpu, true, 0, maxrl * cmSize *
        sizeof(cl_int), h_colscm_trans, 0, NULL, NULL);//zfadded
    CL_CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(clCommandQue[0], d_vec_cpu, true, 0, numRows *
            sizeof(clFloatType), x, 0, NULL, NULL);
        CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(clCommandQue[0], d_rowLengths_cpu, true, 0, cmSize *
        sizeof(int), h_rowLengths, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
/////////////////////////////////////CPU Side////////////////////////





    err = clFinish(clCommandQue[0]);
    CL_CHECK_ERROR(err);
    cl_kernel ellpackr = clCreateKernel(clProgram, "spmv_ellpackr_kernel", &err);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 0, sizeof(cl_mem), (void*) &d_val);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 1, sizeof(cl_mem), (void*) &d_vec);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 2, sizeof(cl_mem), (void*) &d_cols);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 3, sizeof(cl_mem), (void*) &d_rowLengths);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 4, sizeof(cl_int), (void*) &cmSize);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(ellpackr, 5, sizeof(cl_mem), (void*) &d_out);
    CL_CHECK_ERROR(err);

	err= clSetKernelArg(ellpackr, 6, sizeof(cl_mem), &devbitmap); 
    CL_CHECK_ERROR(err);//zfadded

    int rowsetzf=(double)cpuoffset/100*cmSize;//zf
	err= clSetKernelArg(ellpackr, 7, sizeof(cl_int), &rowsetzf); 
    CL_CHECK_ERROR(err);//zfadded


    const size_t globalWorkSize = rowsetzf;
    //const size_t globalWorkSize = cmSize;
    const size_t localWorkSize = BLOCK_SIZE;

    //////////////////////////////////////////
	cl_kernel csrKernelcpu = NULL;
        cl_uint work_dimcpu=1;
        size_t globalsizecpu=3;
        size_t localsizecpu=1;
	csrKernelcpu = clCreateKernel(clProgram, "cpu_csr", &err); 
    CL_CHECK_ERROR(err);

	    err = clSetKernelArg(csrKernelcpu, 0, sizeof(cl_mem), (void*) &d_val_cpu);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(csrKernelcpu, 1, sizeof(cl_mem), (void*) &d_vec_cpu);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(csrKernelcpu, 2, sizeof(cl_mem), (void*) &d_cols_cpu);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(csrKernelcpu, 3, sizeof(cl_mem), (void*) &d_rowLengths_cpu);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(csrKernelcpu, 4, sizeof(cl_int), (void*) &cmSize);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(csrKernelcpu, 5, sizeof(cl_mem), (void*) &d_out);
    CL_CHECK_ERROR(err);

	err= clSetKernelArg(csrKernelcpu, 6, sizeof(cl_mem), &devrowforcpu);
    CL_CHECK_ERROR(err);
	err= clSetKernelArg(csrKernelcpu, 7, sizeof(int), &rowforcpusum); //newadded
    CL_CHECK_ERROR(err);
	err= clSetKernelArg(csrKernelcpu, 8, sizeof(int), &maxrl); //newadded
    CL_CHECK_ERROR(err);

	err= clSetKernelArg(csrKernelcpu, 9, sizeof(cl_int), &rowsetzf); 
    CL_CHECK_ERROR(err);

    //////////////////////////////////////////

    
    
        int iters = 100;
        double t_start, t_end;
	t_start = rtclock();
    for (int j = 0; j < iters; j++)
    {
            err = clEnqueueNDRangeKernel(clCommandQue[0], ellpackr, 1, NULL,
                &globalWorkSize, &localWorkSize, 0, NULL, NULL);
      CL_CHECK_ERROR(err);
	    err= clEnqueueNDRangeKernel(clCommandQue[1], csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); 
      CL_CHECK_ERROR(err);

      err = clFinish(clCommandQue[0]);
      err = clFinish(clCommandQue[1]);
    }
      CL_CHECK_ERROR(err);
	t_end = rtclock();
	fprintf(stdout, "OpenCL Runtime: %0.6lfs\n", (t_end - t_start)/iters);   
    printf("nRows= %d nItems= %d GFlops= %lf timeins: %lf timeinms: %lf\n", numRows, nItems,(double) nItems*2/((t_end - t_start)/iters)/(double)1e9, (t_end - t_start)/iters, (t_end - t_start)/iters*1000);

        printf("CAUTTION: kernel time(ms): %f\n", (t_end - t_start)/iters*1000);

    err = clEnqueueReadBuffer(clCommandQue[0], d_out, true, 0, numRows *
        sizeof(floatType), y, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clFinish(clCommandQue[0]);




    err = clReleaseKernel(ellpackr);
    CL_CHECK_ERROR(err);

    // Free device memory
    err = clReleaseMemObject(d_rowLengths);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_vec);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_out);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_val);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_cols);
    CL_CHECK_ERROR(err);

    // Free host memory
    delete[] h_rowLengths;
    delete[] h_valcm;
    delete[] h_colscm;




	cl_clean_up();


    }











    // compare ref and our results
    cout << endl << "Checking SpMV Correctness ... ";
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (y_ref[i] != y[i])
        {
            error_count++;
            if (i < 10) printf("rowid = %d, ref = %f, y = %f \n", i, y_ref[i], y[i]);
        }
    if (error_count)
    {
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    }
    else
    {
        cout << "PASS!";
    }

    cout << "\n";
    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);
    free(x);
    free(y);
    free(y_ref);

    return err;
}

int main(int argc, char ** argv)
{
    int err = 0;

    int argi = 1;

    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }



    if (argc < 3){ 
      printf("\nUsage: ./main *.mtx percentage(such as 40)\n");
      return 1;
    }   

    cpuoffset = atoi(argv[2]);
    printf("\ncpuoffset=%d\n",cpuoffset);


    cout << "------------------------------------------------------" << endl;

    cout << "----------" << filename << "----------" << endl;

    // report precision of floating-point
    if (sizeof(value_type) == 4)
    {
        cout << "PRECISION = " << "32-bit Single Precision" << endl;
    }
    else if (sizeof(value_type) == 8)
    {
        cout << "PRECISION = " << "64-bit Double Precision" << endl;
    }
    else
    {
        cout << "Wrong precision. Program exit!" << endl;
        return 0;
    }

    cout << "RUN SpMV " << NUM_RUN << " times" << endl;

    err = call_bhsparse<float,float>(filename);

    cout << "------------------------------------------------------" << endl;

    return 0;
}

