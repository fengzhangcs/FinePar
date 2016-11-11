#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "spmv_util.h"
//#include "spmv_csr_scalar.h"

int main(int argc, char* argv[])
{
    

    char* filename = argv[1];
    int ntimes = 20;

    coo_matrix<int, float> mat;
    init_coo_matrix(mat);
    ReadMMF(filename, &mat);

    //char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    
   {
	sprintf(clfilename, "%s%s", "/home/pacman/zf/spmv/zfcsr", "/spmv_csr_vector.cl");
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

