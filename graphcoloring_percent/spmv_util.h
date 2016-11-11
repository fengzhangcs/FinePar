
#include "CL/cl.h"
#include<assert.h>
#include<math.h>
#include<sys/time.h>
#include<stdio.h>
//#include "matrix_storage.h"
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "CL/cl.h"
#include<iostream>
using namespace std;


#define VEC2DWIDTH 1024
#define CSR_VEC_GROUP_SIZE 64
//#define WARPSIZE 64
#define WARPSIZE 32



#define CSR_VEC_MIN_TH_NUM 5760
#define MAX_LEVELS  1000

template <class dimType>
struct matrixInfo
{
    /** Matrix width*/
    dimType width;
     /** Matrix height*/
    dimType height;
    /** Number of non zeros*/
    dimType nnz;

};


template <class dimType, class dataType>
struct coo_matrix
{
    matrixInfo<dimType> matinfo;

    /** Row index, size nnz*/
    dimType* coo_row_id;
    /** Column index, size nnz*/
    dimType* coo_col_id;
    /** Data, size nnz */
    dataType* coo_data;
};

template <class dimType, class dataType>
struct csr_matrix
{
    matrixInfo<dimType> matinfo;

    /** Row pointer, size height + 1*/
    dimType* csr_row_ptr;
    /** Column index, size nnz*/
    dimType* csr_col_id;
    /** Data, size nnz */
    dataType* csr_data;
};

extern  void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, float* vec, float* result, int padNum,  double& opttime, int& optmethod, char* oclfilename, float* coores, int ntimes);


extern bool LoadSourceFromFile(
    const char* filename,
    char* & sourceCode );


extern int initialization2(cl_device_id* devices, cl_context* context, cl_command_queue* cmdQueue, cl_program* program, char* clFileName, cl_command_queue* cmdQueue2);

extern int initialization(cl_device_id* devices, cl_context* context, cl_command_queue* cmdQueue, cl_program* program, char* clFileName);

extern char *print_cl_errstring(cl_int err) ;

#define ALLOCATE_GPU_READ(deviceBuf, HostBuf, mem_size) \
    if(errorCode == CL_SUCCESS) { \
    deviceBuf = clCreateBuffer(context,  CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, mem_size, HostBuf, &errorCode); \
    clEnqueueWriteBuffer(cmdQueue, deviceBuf, CL_TRUE, 0, mem_size, HostBuf, 0, NULL, NULL); \
    if( errorCode != CL_SUCCESS ) { printf("Error: clCreateBuffer returned %d\n", errorCode); } \
    }
#define ALLOCATE_GPU_READ_cpu(deviceBuf, HostBuf, mem_size) \
    if(errorCode == CL_SUCCESS) { \
    deviceBuf = clCreateBuffer(context,  CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, mem_size, HostBuf, &errorCode); \
    clEnqueueWriteBuffer(cmdQueue, deviceBuf, CL_TRUE, 0, mem_size, HostBuf, 0, NULL, NULL); \
    if( errorCode != CL_SUCCESS ) { printf("Error: clCreateBuffer returned %d\n", errorCode); } \
    }


#define CHECKERROR {if (errorCode != CL_SUCCESS) {fprintf(stderr, "Error at line %d code %d message %s\n", __LINE__, errorCode, print_cl_errstring(errorCode)); exit(1);}}



 extern int findPaddedSize(int realSize, int alignment);

 extern double distance(float* vec1, float* vec2, int size);

 extern void two_vec_compare(int* coovec, int* newvec, int size);

 extern double timestamp ();


 extern void freeObjects(cl_device_id* devices, cl_context* context, cl_command_queue* cmdQueue, cl_program* program);


template <class dimType>
 extern void init_mat_info(matrixInfo<dimType>& info);


template <class dimType, class dataType>
 extern bool if_sorted_coo(coo_matrix<dimType, dataType>* mat);


template <class dimType, class dataType>
 extern bool sort_coo(coo_matrix<dimType, dataType>* mat);

//Read from matrix market format to a coo mat
 extern void ReadMMF(char* filename, coo_matrix<int, float>* mat);


template <class dimType, class dataType>
extern void init_coo_matrix(coo_matrix<dimType, dataType>& mat);


template <class dimType, class dataType>
 extern void free_coo_matrix(coo_matrix<dimType, dataType>& mat);


void printMatInfo(coo_matrix<int, float>* mat);


template <class dimType, class dataType>
void coo2csr(coo_matrix<dimType, dataType>* source, csr_matrix<dimType, dataType>* dest);


template<class dimType, class dataType>
void initVectorZero(dataType* vec, dimType vec_size);

template<class dimType, class dataType>
void initVectorOne(dataType* vec, dimType vec_size);


template <class dimType, class dataType>
void coo_spmv(coo_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size);


void spmv_only(coo_matrix<int, float>* mat, float* vec, float* coores);


template <class dimType, class dataType>
void init_coo_matrix(coo_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.coo_row_id = NULL;
    mat.coo_col_id = NULL;
    mat.coo_data = NULL;
}

template <class dimType>
void init_mat_info(matrixInfo<dimType>& info)
{
    info.width = (dimType)0;
    info.height = (dimType)0;
    info.nnz = (dimType)0;
}


template <class dimType, class dataType>
void coo2csr(coo_matrix<dimType, dataType>* source, csr_matrix<dimType, dataType>* dest)
{
    if (!if_sorted_coo(source))
    {
	assert(sort_coo(source) == true);
    }

    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dimType nnz = source->matinfo.nnz;
    dest->csr_row_ptr = (dimType*)malloc(sizeof(dimType)*(source->matinfo.height + 1));
    dest->csr_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->csr_data = (dataType*)malloc(sizeof(dataType)*nnz);

    memcpy(dest->csr_data, source->coo_data, sizeof(dataType)*nnz);
    memcpy(dest->csr_col_id, source->coo_col_id, sizeof(dimType)*nnz);

    dest->csr_row_ptr[0] = 0;
    dimType row = (dimType) 0;
    dimType curRow = (dimType) 0;
    while (row < nnz)
    {
	while (source->coo_row_id[row] == curRow && row < nnz)
	    row++;
	curRow++;
	dest->csr_row_ptr[curRow] = row;
    }
    if (curRow < source->matinfo.height)
    {
	curRow++;
	while (curRow <= source->matinfo.height)
	{
	    dest->csr_row_ptr[curRow] = dest->csr_row_ptr[curRow - 1];
	    curRow++;
	}
    }
}

template<class dimType, class dataType>
void initVectorZero(dataType* vec, dimType vec_size)
{
    for (dimType i = 0; i < vec_size; i++)
    {
	vec[i] = (dataType)0;
    }
}

template<class dimType, class dataType>
void initVectorOne(dataType* vec, dimType vec_size)
{
    for (dimType i = 0; i < vec_size; i++)
    {
	vec[i] = (dataType)1;
    }
}

template <class dimType, class dataType>
void free_coo_matrix(coo_matrix<dimType, dataType>& mat)
{
    if (mat.coo_row_id != NULL)
	free(mat.coo_row_id);
    if (mat.coo_col_id != NULL)
	free(mat.coo_col_id);
    if (mat.coo_data != NULL)
	free(mat.coo_data);
}

template <class dimType, class dataType>
bool if_sorted_coo(coo_matrix<dimType, dataType>* mat)
{
    dimType nnz = mat->matinfo.nnz;
    for (int i = 0; i < nnz - 1; i++)
    {
        if ((mat->coo_row_id[i] > mat->coo_row_id[i+1]) || (mat->coo_row_id[i] == mat->coo_row_id[i+1] && mat->coo_col_id[i] > mat->coo_col_id[i+1]))
            return false;
    }
    return true;
}



template <class dimType, class dataType>
bool sort_coo(coo_matrix<dimType, dataType>* mat)
{

    int i = 0;
    dimType  beg[MAX_LEVELS], end[MAX_LEVELS], L, R ;
    dimType pivrow, pivcol;
    dataType pivdata;

    beg[0]=0; 
    end[0]=mat->matinfo.nnz;
    while (i>=0) 
    {
	L=beg[i];
	if (end[i] - 1 > end[i])
	    R = end[i];
	else
	    R = end[i] - 1;
	if (L<R) 
	{
	    dimType middle = (L+R)/2;
	    pivrow=mat->coo_row_id[middle]; 
	    pivcol=mat->coo_col_id[middle];
	    pivdata=mat->coo_data[middle];
	    mat->coo_row_id[middle] = mat->coo_row_id[L];
	    mat->coo_col_id[middle] = mat->coo_col_id[L];
	    mat->coo_data[middle] = mat->coo_data[L];
	    mat->coo_row_id[L] = pivrow;
	    mat->coo_col_id[L] = pivcol;
	    mat->coo_data[L] = pivdata;
	    if (i==MAX_LEVELS-1) 
		return false;
	    while (L<R) 
	    {
		while (((mat->coo_row_id[R] > pivrow) || 
			    (mat->coo_row_id[R] == pivrow && mat->coo_col_id[R] > pivcol)) 
			&& L<R) 
		    R--; 
		if (L<R) 
		{
		    mat->coo_row_id[L] = mat->coo_row_id[R];
		    mat->coo_col_id[L] = mat->coo_col_id[R];
		    mat->coo_data[L] = mat->coo_data[R];
		    L++;
		}
		while (((mat->coo_row_id[L] < pivrow) || 
			    (mat->coo_row_id[L] == pivrow && mat->coo_col_id[L] < pivcol)) 
			&& L<R) 
		    L++; 
		if (L<R) 
		{
		    mat->coo_row_id[R] = mat->coo_row_id[L];
		    mat->coo_col_id[R] = mat->coo_col_id[L];
		    mat->coo_data[R] = mat->coo_data[L];
		    R--;
		}
	    }
	    mat->coo_row_id[L] = pivrow;
	    mat->coo_col_id[L] = pivcol;
	    mat->coo_data[L] = pivdata;
	    beg[i+1]=L+1; 
	    end[i+1]=end[i]; 
	    end[i++]=L; 
	}
	else 
	{
	    i--; 
	}
    }

    return true;
}

extern void pad_csr(csr_matrix<int, float>* source, csr_matrix<int, float>* dest, int alignment);

template <class dimType, class dataType>
void free_csr_matrix(csr_matrix<dimType, dataType>& mat)
{
    if (mat.csr_row_ptr != NULL)
	free(mat.csr_row_ptr);
    if (mat.csr_col_id != NULL)
	free(mat.csr_col_id);
    if (mat.csr_data != NULL)
	free(mat.csr_data);
}


