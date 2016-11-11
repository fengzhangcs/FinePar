/**
 * 2DConvolution.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include<cassert>
#include<iostream>
using namespace std;

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//#include "polybenchUtilFuncts.h"
//#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
//#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)
#define CL_CHECK_ERROR(err) \
      {                       \
                if (err != CL_SUCCESS)                  \
                { \
                              std::cerr << __FILE__ << ':' << __LINE__ << " " << std::endl; \
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
static const int PAD_FACTOR = 16;



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



template <typename floatType>
void fill(floatType *A, const int n, const float maxi)
{
    for (int j = 0; j < n; j++)
    {   
        A[j] = ((floatType) maxi * (rand() / (RAND_MAX + 1.0f)));
    }   
}

void initRandomMatrix(int *cols, int *rowDelimiters, const int n, const int dim)
{
    int nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    double prob = (double)n / ((double)dim * (double)dim);

    // Seed random number generator
    srand48(8675309L);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    bool fillRemaining = false;
    for (int i = 0; i < dim; i++)
    {   
        rowDelimiters[i] = nnzAssigned;
        for (int j = 0; j < dim; j++)
        {
            int numEntriesLeft = (dim * dim) - ((i * dim) + j); 
            int needToAssign   = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if ((nnzAssigned < n && drand48() <= prob) || fillRemaining)
            {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }   
    // Observe the convention to put the number of non zeroes at the end of the
    // row delimiters array
    rowDelimiters[dim] = n;
    assert(nnzAssigned == n); 
}

template <typename floatType>
void spmvCpu(const floatType *val, const int *cols, const int *rowDelimiters,
             const floatType *vec, int dim, floatType *out)
{

    for (int i=0; i<dim; i++) 
    {    
        floatType t = 0; 
        for (int j=rowDelimiters[i]; j<rowDelimiters[i+1]; j++) 
        {
            int col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t; 
    }    

}

template <typename floatType>
bool verifyResults(const floatType *cpuResults, const floatType *gpuResults,
                   const int size)
{

    bool passed = true;
    for (int i=0; i<size; i++)
    {
        if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i]
            > MAX_RELATIVE_ERROR)
        {
//#ifdef VERBOSE_OUTPUT
           cout << "Mismatch at i: "<< i << " ref: " << cpuResults[i] <<
                " dev: " << gpuResults[i] << endl;
//#endif
            passed = false;
        }
    }

    if (passed)
    {
        cout << "Passed" << endl;
    }
    else
    {
        cout << "Failed" << endl;
    }
    return passed;
}

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




template <typename floatType, typename clFloatType>
void RunTest(int nRows){

    int nItems;            // number of non-zero elements in the matrix
//    int nItemsPadded;
    int numRows;           // number of rows in the matrix
    float *h_val;
    int *h_cols, *h_rowDelimiters;
    float maxval=10;

    numRows = nRows;
    nItems = numRows * numRows / 100; 
    h_val = new floatType[nItems];
    h_cols = new int[nItems];
    h_rowDelimiters = new int[nRows+1];

    fill(h_val, nItems, maxval);
    initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows);

    float *h_vec = new floatType[numRows];
    float *refOut = new floatType[numRows];
    //float *h_out = new floatType[numRows];
    int paddedSize = numRows + (PAD_FACTOR - numRows % PAD_FACTOR);
    float *h_out = new floatType[paddedSize];

    int *h_rowDelimitersPad = new int[numRows+1];
	double t_start, t_end;
    fill<float>(h_vec, numRows, maxval);
	t_start = rtclock();
    spmvCpu<float>(h_val, h_cols, h_rowDelimiters, h_vec, numRows, refOut);
	t_end = rtclock();
	fprintf(stdout, "CPU Sequential Runtime: %0.6lfs\n", t_end - t_start);   

    int *h_rowLengths = new int[paddedSize];
    int maxrl = 0;
    for (int k=0; k<numRows; k++)
    {
        h_rowLengths[k] = h_rowDelimiters[k+1] - h_rowDelimiters[k];
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
    bool padded=false;
    //int cmSize = padded ? paddedSize : numRows;
    floatType *h_valcm = new floatType[maxrl * cmSize];
    int *h_colscm = new int[maxrl * cmSize];
    convertToColMajor<float>(h_val, h_cols, numRows, h_rowDelimiters, h_valcm,
                              h_colscm, h_rowLengths, maxrl, padded);



//intothe ellpackfunction
    int err;

    // Device data structures
    cl_mem d_val, d_vec, d_out; // floating point
    cl_mem d_cols, d_rowLengths; // integer

    // Allocate device memory
    d_val = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, maxrl * cmSize *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_cols = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, maxrl * cmSize *
        sizeof(int), NULL, &err);
    CL_CHECK_ERROR(err);
    d_vec = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, numRows *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_out = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, paddedSize *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_rowLengths = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, cmSize *
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
            sizeof(clFloatType), h_vec, 0, NULL, NULL);
        CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(clCommandQue[0], d_rowLengths, true, 0, cmSize *
        sizeof(int), h_rowLengths, 0, NULL, NULL);
    CL_CHECK_ERROR(err);

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

    const size_t globalWorkSize = cmSize;
    const size_t localWorkSize = BLOCK_SIZE;

        int iters = 100;
	t_start = rtclock();
    for (int j = 0; j < iters; j++)
    {
            err = clEnqueueNDRangeKernel(clCommandQue[0], ellpackr, 1, NULL,
                &globalWorkSize, &localWorkSize, 0, NULL, NULL);

      err = clFinish(clCommandQue[0]);
    }
      CL_CHECK_ERROR(err);
	t_end = rtclock();
	fprintf(stdout, "OpenCL Runtime: %0.6lfs\n", (t_end - t_start)/iters);   
    printf("nRows= %d nItems= %d GFlops= %lf \n", nRows, nItems,(double) nItems*2/((t_end - t_start)/iters)/(double)1e9);

         err = clEnqueueReadBuffer(clCommandQue[0], d_out, true, 0, numRows *
             sizeof(clFloatType), h_out, 0, NULL, NULL);
         CL_CHECK_ERROR(err);
         err = clFinish(clCommandQue[0]);

    if (! verifyResults(refOut, h_out, numRows))
    {
      return;  // If results don't match, don't report performance
    }


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




















/*
    // Device data structures
    cl_mem d_val, d_vec, d_out;
    cl_mem d_cols, d_rowDelimiters;
    int err;

    // Allocate device memory
    d_val = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, nItems*
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_cols = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, nItems*
        sizeof(cl_int), NULL, &err);
    d_vec = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, numRows *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_out = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, numRows *
        sizeof(clFloatType), NULL, &err);
    CL_CHECK_ERROR(err);
    d_rowDelimiters = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, (numRows+1) *
        sizeof(cl_int), NULL, &err);
    CL_CHECK_ERROR(err);

    // Transfer data to device
    err = clEnqueueWriteBuffer(clCommandQue[0], d_val, true, 0, nItems*
        sizeof(floatType), h_val, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(clCommandQue[0], d_cols, true, 0, nItems*
        sizeof(int), h_cols, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(clCommandQue[0], d_vec, true, 0, numRows *
        sizeof(floatType), h_vec, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(clCommandQue[0], d_rowDelimiters, true, 0, (numRows+1) *
        sizeof(int), h_rowDelimiters, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clFinish(clCommandQue[0]);
    CL_CHECK_ERROR(err);


    clKernel = clCreateKernel(clProgram, "spmv_csr_scalar_kernel", &err);
    if(err!= CL_SUCCESS) printf("Error in creating kernel\n");
    err = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void*) &d_val);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void*) &d_vec);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void*) &d_cols);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(clKernel, 3, sizeof(cl_mem), (void*) &d_rowDelimiters);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(clKernel, 4, sizeof(cl_int), (void*) &numRows);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(clKernel, 5, sizeof(cl_mem), (void*) &d_out);
    CL_CHECK_ERROR(err);

    const size_t scalarGlobalWSize = numRows;
    size_t localWorkSize = BLOCK_SIZE;
        int iters = 100;
	t_start = rtclock();
    for (int j = 0; j < iters; j++)
    {
      err = clEnqueueNDRangeKernel(clCommandQue[0], clKernel, 1, NULL,
          &scalarGlobalWSize, &localWorkSize, 0, NULL, NULL);
      err = clFinish(clCommandQue[0]);
    }
      CL_CHECK_ERROR(err);
	t_end = rtclock();
	fprintf(stdout, "OpenCL Runtime: %0.6lfs\n", (t_end - t_start)/iters);   


    err = clEnqueueReadBuffer(clCommandQue[0], d_out, true, 0, numRows *
        sizeof(floatType), h_out, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clFinish(clCommandQue[0]);

    if (! verifyResults(refOut, h_out, numRows))
    {
      return;  // If results don't match, don't report performance
    }


    err = clReleaseMemObject(d_rowDelimiters);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_vec);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_out);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_val);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_cols);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(clKernel);
    CL_CHECK_ERROR(err);
*/

}


int main(int argc, char *argv[])
{
	int i;


	read_cl_file();
	cl_initialization_fusion();
	cl_load_prog();

	

//	t_start = rtclock();
        
        RunTest<float,float>(32768);
        //RunTest<float,float>(12288);

//	t_end = rtclock(); 

//	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	cl_clean_up();
    	return 0;
}

