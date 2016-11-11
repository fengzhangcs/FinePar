
#include"spmv_util.h"


bool LoadSourceFromFile(
    const char* filename,
    char* & sourceCode )
{
    bool error = false;
    FILE* fp = NULL;
    int nsize = 0;

    // Open the shader file

    fp = fopen(filename, "rb");
    if( !fp )
    {
        error = true;
    }
    else
    {
        // Allocate a buffer for the file contents
        fseek( fp, 0, SEEK_END );
        nsize = ftell( fp );
        fseek( fp, 0, SEEK_SET );

        sourceCode = new char [ nsize + 1 ];
        if( sourceCode )
        {
            fread( sourceCode, 1, nsize, fp );
            sourceCode[ nsize ] = 0; // Don't forget the NULL terminator
        }
        else
        {
            error = true;
        }

        fclose( fp );
    }

    return error;
}


/*
 *zhangfeng added for cpu
 */
int initialization2(cl_device_id* devices, cl_context* context, cl_command_queue* cmdQueue, cl_program* program, char* clFileName,   cl_command_queue* cpu_cmdQueue)
{
    size_t devicesSize = 0;
    size_t numDevices = 0;

    cl_int errorCode = CL_SUCCESS;

    if (devices != NULL)
    {
	printf("Device id not NULL\n");	
	return -1;
    }
    if ((*context) != NULL)
    {
	printf("Context not NULL\n");	
	return -1;
    }
    if ((*cmdQueue) != NULL)
    {
	printf("Command Queue not NULL\n");	
	return -1;
    }
    if ((*program) != NULL)
    {
	printf("Program not NULL\n");	
	return -1;
    }

    char* programSource = NULL;

    if( errorCode == CL_SUCCESS )
    {
	printf("Program File Name: %s\n", clFileName );
	printf("---\n");
    }

    if( errorCode == CL_SUCCESS )
    {
	// Load the kernel source from the passed in file.
	if( LoadSourceFromFile( clFileName, programSource ) )
	{
	    printf("Error: Couldn't load kernel source from file '%s'.\n", clFileName );
	    errorCode = CL_INVALID_OPERATION;
	}
    }


    if( errorCode == CL_SUCCESS )
    {
	cl_uint size_ret = 0;
	errorCode = clGetPlatformIDs(0, NULL, &size_ret);
	if ( (errorCode != CL_SUCCESS) || (size_ret == 0) )
	{
	    return -1;
	}

	cl_platform_id * platforms = new cl_platform_id[size_ret];

	errorCode = clGetPlatformIDs(size_ret, platforms, NULL);

	if ( errorCode != CL_SUCCESS )
	{
	    delete[] platforms;
	    return -1;
	}
        devices = new cl_device_id [2];
        cl_uint num_devices;
        errorCode|= clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &devices[0], &num_devices);
        errorCode|= clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &devices[1], &num_devices);
        if (errorCode != CL_SUCCESS)
        {
          printf("Error: Failed to create a device group!\n");
          return EXIT_FAILURE;
        }
        (*context) = clCreateContext(0, 2, devices, NULL, NULL, &errorCode);
        if (!(*context))
        {
          printf("Error: Failed to create a compute context!\n");
          return EXIT_FAILURE;
        }
        if( errorCode == CL_SUCCESS )
        {
          // Create a command queue.
          *cmdQueue= clCreateCommandQueue(*context, devices[ 0], CL_QUEUE_PROFILING_ENABLE, &errorCode );//zhangfeng
          *cpu_cmdQueue = clCreateCommandQueue(*context, devices[ 1], CL_QUEUE_PROFILING_ENABLE, &errorCode );//zhangfeng
          if( errorCode != CL_SUCCESS )
          {
            printf("Error: clCreateCommandQueue() returned %d.\n", errorCode);
          }
        }
    }

    if( errorCode == CL_SUCCESS )
    {
	// Create program.
	(*program) = clCreateProgramWithSource(*context, 1, ( const char** )&programSource, NULL, &errorCode );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clCreateProgramWithSource() returned %d.\n", errorCode );
	}

	// No longer need the programSource char array
	delete [] programSource;
	programSource = NULL;
    }


    if( errorCode == CL_SUCCESS )
    {
	// Build the program.
	errorCode = clBuildProgram(*program, 0,  0 , "", NULL, NULL );
	//errorCode = clBuildProgram(*program, 1, &devices[ 0 ], "", NULL, NULL );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clBuildProgram() returned %d.\n", errorCode );
	}
	//if( errorCode != CL_SUCCESS )
	{

	    size_t  buildLogSize = 0;
	    clGetProgramBuildInfo(*program,  0 , CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize );
	    //clGetProgramBuildInfo(*program, devices[ 0 ], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize );
	    cl_char*    buildLog = new cl_char[ buildLogSize ];
	    if( buildLog )
	    {
		clGetProgramBuildInfo(*program,  0 , CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL );
		//clGetProgramBuildInfo(*program, devices[ 0 ], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL );

		printf(">>> Build Log:\n");
		printf("%s\n", buildLog );
		printf("<<< End of Build Log\n");
	    }
	}
    }

    if (programSource)
    {
	delete[] programSource;
	programSource = NULL;
    }

    return 1;
}


int initialization(cl_device_id* devices, cl_context* context, cl_command_queue* cmdQueue, cl_program* program, char* clFileName)
{
    size_t devicesSize = 0;
    size_t numDevices = 0;

    cl_int errorCode = CL_SUCCESS;

    if (devices != NULL)
    {
	printf("Device id not NULL\n");	
	return -1;
    }
    if ((*context) != NULL)
    {
	printf("Context not NULL\n");	
	return -1;
    }
    if ((*cmdQueue) != NULL)
    {
	printf("Command Queue not NULL\n");	
	return -1;
    }
    if ((*program) != NULL)
    {
	printf("Program not NULL\n");	
	return -1;
    }

    char* programSource = NULL;

    if( errorCode == CL_SUCCESS )
    {
	printf("Program File Name: %s\n", clFileName );
	printf("---\n");
    }

    if( errorCode == CL_SUCCESS )
    {
	// Load the kernel source from the passed in file.
	if( LoadSourceFromFile( clFileName, programSource ) )
	{
	    printf("Error: Couldn't load kernel source from file '%s'.\n", clFileName );
	    errorCode = CL_INVALID_OPERATION;
	}
    }


    if( errorCode == CL_SUCCESS )
    {
	cl_uint size_ret = 0;
	errorCode = clGetPlatformIDs(0, NULL, &size_ret);
	if ( (errorCode != CL_SUCCESS) || (size_ret == 0) )
	{
	    return -1;
	}

	cl_platform_id * platforms = new cl_platform_id[size_ret];

	errorCode = clGetPlatformIDs(size_ret, platforms, NULL);

	if ( errorCode != CL_SUCCESS )
	{
	    delete[] platforms;
	    return -1;
	}

	cl_context_properties  properties[3] =  {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], NULL };

	// Create OpenCL device and context.
	(*context) = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errorCode);
	//(*context) = clCreateContextFromType(properties, "CL_DEVICE_TYPE_GPU", NULL, NULL, &errorCode);
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clCreateContextFromType() returned %d.\n", errorCode);
	}
	delete[] platforms;
    }

    if( errorCode == CL_SUCCESS )
    {
	// Get the number of devices associated with the context.
	errorCode = clGetContextInfo(*context, CL_CONTEXT_DEVICES, 0, NULL, &devicesSize );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clGetContextInfo() to get num devices returned %d.\n", errorCode);
	}
    }

    if( errorCode == CL_SUCCESS )
    {
	// Get the list of devices associated with the context.
	numDevices = devicesSize / sizeof( cl_device_id );
	devices = new cl_device_id [ numDevices ];

	errorCode = clGetContextInfo(*context, CL_CONTEXT_DEVICES, devicesSize, devices, NULL );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clGetContextInfo() to get list of devices returned %d.\n", errorCode);
	}
    }

    if( errorCode == CL_SUCCESS )
    {
	// Print the device info.
	//errorCode = PrintDeviceInfo(devices, numDevices);
    }

    if( errorCode == CL_SUCCESS )
    {
	// Create a command queue.
	(*cmdQueue) = clCreateCommandQueue(*context, devices[ 0 ], CL_QUEUE_PROFILING_ENABLE, &errorCode );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clCreateCommandQueue() returned %d.\n", errorCode);
	}
    }

    if( errorCode == CL_SUCCESS )
    {
	// Create program.
	(*program) = clCreateProgramWithSource(*context, 1, ( const char** )&programSource, NULL, &errorCode );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clCreateProgramWithSource() returned %d.\n", errorCode );
	}

	// No longer need the programSource char array
	delete [] programSource;
	programSource = NULL;
    }


    if( errorCode == CL_SUCCESS )
    {
	// Build the program.
	errorCode = clBuildProgram(*program, 1, &devices[ 0 ], "", NULL, NULL );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clBuildProgram() returned %d.\n", errorCode );
	}
	//if( errorCode != CL_SUCCESS )
	{

	    size_t  buildLogSize = 0;
	    clGetProgramBuildInfo(*program, devices[ 0 ], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize );
	    cl_char*    buildLog = new cl_char[ buildLogSize ];
	    if( buildLog )
	    {
		clGetProgramBuildInfo(*program, devices[ 0 ], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL );

		printf(">>> Build Log:\n");
		printf("%s\n", buildLog );
		printf("<<< End of Build Log\n");
	    }
	}
    }

    if (programSource)
    {
	delete[] programSource;
	programSource = NULL;
    }

    return 1;
}

char *print_cl_errstring(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return strdup("Success!");
        case CL_DEVICE_NOT_FOUND:                 return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE:             return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE:           return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES:                 return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY:               return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP:                 return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH:            return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE:            return strdup("Program build failure");
        case CL_MAP_FAILURE:                      return strdup("Map failure");
        case CL_INVALID_VALUE:                    return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE:              return strdup("Invalid device type");
        case CL_INVALID_PLATFORM:                 return strdup("Invalid platform");
        case CL_INVALID_DEVICE:                   return strdup("Invalid device");
        case CL_INVALID_CONTEXT:                  return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES:         return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE:            return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR:                 return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT:               return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE:               return strdup("Invalid image size");
        case CL_INVALID_SAMPLER:                  return strdup("Invalid sampler");
        case CL_INVALID_BINARY:                   return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS:            return strdup("Invalid build options");
        case CL_INVALID_PROGRAM:                  return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE:       return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME:              return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION:        return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL:                   return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX:                return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE:                return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE:                 return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS:              return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION:           return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE:          return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE:           return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET:            return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST:          return strdup("Invalid event wait list");
        case CL_INVALID_EVENT:                    return strdup("Invalid event");
        case CL_INVALID_OPERATION:                return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT:                return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE:              return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL:                return strdup("Invalid mip-map level");
        default:                                  return strdup("Unknown");
    }
} 


#define ALLOCATE_GPU_READ(deviceBuf, HostBuf, mem_size) \
    if(errorCode == CL_SUCCESS) { \
    deviceBuf = clCreateBuffer(context,  CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, mem_size, HostBuf, &errorCode); \
    clEnqueueWriteBuffer(cmdQueue, deviceBuf, CL_TRUE, 0, mem_size, HostBuf, 0, NULL, NULL); \
    if( errorCode != CL_SUCCESS ) { printf("Error: clCreateBuffer returned %d\n", errorCode); } \
    }
#define CHECKERROR {if (errorCode != CL_SUCCESS) {fprintf(stderr, "Error at line %d code %d message %s\n", __LINE__, errorCode, print_cl_errstring(errorCode)); exit(1);}}



int findPaddedSize(int realSize, int alignment)
{
    if (realSize % alignment == 0)
	return realSize;
    return realSize + alignment - realSize % alignment;
}

double distance(float* vec1, float* vec2, int size)
{
	double sum = 0.0f;
	for (int i = 0; i < size; i++)
	{
		double tmp = vec1[i] - vec2[i];
		sum += tmp * tmp;
	}
	return sqrt(sum);
}



void two_vec_compare(int* coovec, int* newvec, int size)
{
//    double dist = distance(coovec, newvec, size);

    double maxdiff = 0.0f;
    int maxdiffid = 0;
    double maxratiodiff = 0.0f;
    int count = 0;
    for (int i = 0; i < size; i++)
    {
	int tmpa = coovec[i];
	if (tmpa < 0)
	    tmpa *= (-1);
	int tmpb = newvec[i];
	if (tmpb < 0)
	    tmpb *= (-1);
	double diff = tmpa - tmpb;
	if (diff < 0)
	    diff *= (-1);
	float maxab = (tmpa > tmpb)?tmpa:tmpb;
	double ratio = 0.0f;
	if (maxab > 0)
	    ratio = diff / maxab;
	if (diff > maxdiff)
	{
	    maxdiff = diff;
	    maxdiffid = i;
	}
	if (ratio > maxratiodiff)
	    maxratiodiff = ratio;
	if (coovec[i] != newvec[i] && count < 10)
	{
	    printf("Error i %d coo res %d res %d \n", i, coovec[i], newvec[i]);
	    count++;
	}
    }
    printf("Max diff id %d coo res %d res %d \n", maxdiffid, coovec[maxdiffid], newvec[maxdiffid]);
    printf("\nCorrectness Check: Distance N max diff %e max diff ratio %e vec size %d\n",  maxdiff, maxratiodiff, size);
}


double timestamp ()
{
    struct timeval tv;
    gettimeofday (&tv, 0);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}


void freeObjects(cl_device_id* devices, cl_context* context, cl_command_queue* cmdQueue, cl_program* program)
{
    if (*program)
	clReleaseProgram(*program);
    if (*cmdQueue)
	clReleaseCommandQueue(*cmdQueue);
    if (devices)
    {
	delete[] devices;
	devices = NULL;
    }
    if (*context)
	clReleaseContext(*context);
}



//Read from matrix market format to a coo mat
void ReadMMF(char* filename, coo_matrix<int, float>* mat)
{
    FILE* infile = fopen(filename, "r");
    char tmpstr[100];
    char tmpline[1030];
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    bool ifreal = false;
    if (strcmp(tmpstr, "real") == 0)
        ifreal = true;
    bool ifsym = false;
    fscanf(infile, "%s", tmpstr);
    if (strcmp(tmpstr, "symmetric") == 0)
        ifsym = true;
    int height = 0;
    int width = 0;
    int nnz = 0;
    while (true)
    {
	fscanf(infile, "%s", tmpstr);
	if (tmpstr[0] != '%')
	{
	    height = atoi(tmpstr);
	    break;
	}
	fgets(tmpline, 1025, infile);
    }

    fscanf(infile, "%d %d", &width, &nnz);
    mat->matinfo.height = height;
    mat->matinfo.width = width;
    int* rows = (int*)malloc(sizeof(int)*nnz);
    int* cols = (int*)malloc(sizeof(int)*nnz);
    float* data = (float*)malloc(sizeof(float)*nnz);
    int diaCount = 0;
    for (int i = 0; i < nnz; i++)
    {
        int rowid = 0;
        int colid = 0;
        fscanf(infile, "%d %d", &rowid, &colid);
        rows[i] = rowid - 1;
        cols[i] = colid - 1;
        data[i] = 1.0f;
        if (ifreal)
        {
            double dbldata = 0.0f;
            fscanf(infile, "%lf", &dbldata);
            data[i] = (float)dbldata;
        }
        if (rows[i] == cols[i])
            diaCount++;
    }
    
    if (ifsym)
    {
        int newnnz = nnz * 2 - diaCount;
        mat->matinfo.nnz = newnnz;
        mat->coo_row_id = (int*)malloc(sizeof(int)*newnnz);
        mat->coo_col_id = (int*)malloc(sizeof(int)*newnnz);
        mat->coo_data = (float*)malloc(sizeof(float)*newnnz);
        int matid = 0;
        for (int i = 0; i < nnz; i++)
        {
            mat->coo_row_id[matid] = rows[i];
            mat->coo_col_id[matid] = cols[i];
            mat->coo_data[matid] = data[i];
            matid++;
            if (rows[i] != cols[i])
            {
                mat->coo_row_id[matid] = cols[i];
                mat->coo_col_id[matid] = rows[i];
                mat->coo_data[matid] = data[i];
                matid++;
            }
        }
        assert(matid == newnnz);
        bool tmp = sort_coo<int, float>(mat);
        assert(tmp == true);
    }
    else
    {
        mat->matinfo.nnz = nnz;
        mat->coo_row_id = (int*)malloc(sizeof(int)*nnz);
        mat->coo_col_id = (int*)malloc(sizeof(int)*nnz);
        mat->coo_data = (float*)malloc(sizeof(float)*nnz);
        memcpy(mat->coo_row_id, rows, sizeof(int)*nnz);
        memcpy(mat->coo_col_id, cols, sizeof(int)*nnz);
        memcpy(mat->coo_data, data, sizeof(float)*nnz);
        if (!if_sorted_coo<int, float>(mat))
            sort_coo<int, float>(mat);
        //assert(if_sorted_coo(mat) == true);
    }
    
    fclose(infile);
    free(rows);
    free(cols);
    free(data);
}



void printMatInfo(coo_matrix<int, float>* mat)
{
    printf("\nMatInfo: Width %d Height %d NNZ %d\n", mat->matinfo.width, mat->matinfo.height, mat->matinfo.nnz);
    int minoffset = mat->matinfo.width;
    int maxoffset = -minoffset;
    int nnz = mat->matinfo.nnz;
    int lessn16 = 0;
    int inn16 = 0;
    int less16 = 0;
    int large16 = 0;
    for (int i = 0; i < nnz; i++)
    {
	int rowid = mat->coo_row_id[i];
	int colid = mat->coo_col_id[i];
	int diff = rowid - colid;
	if (diff < minoffset)
	    minoffset = diff;
	if (diff > maxoffset)
	    maxoffset = diff;
	if (diff < -15)
	    lessn16++;
	else if (diff < 0)
	    inn16++;
	else if (diff < 16)
	    less16++;
	else
	    large16++;
    }
    printf("Max Offset %d Min Offset %d\n", maxoffset, minoffset);
    printf("Histogram: <-15: %d -15~-1 %d < 0-15 %d > 16 %d\n", lessn16, inn16, less16, large16);

    if (!if_sorted_coo(mat))
    {
	assert(sort_coo(mat) == true);
    }

    int* cacheperrow = (int*)malloc(sizeof(int)*mat->matinfo.height);
    int* elemperrow = (int*)malloc(sizeof(int)*mat->matinfo.height);
    memset(cacheperrow, 0, sizeof(int)*mat->matinfo.height);
    memset(elemperrow, 0, sizeof(int)*mat->matinfo.height);
    int index = 0;
    for (int i = 0; i < mat->matinfo.height; i++)
    {
	if (i < mat->coo_row_id[index])
	    continue;
	int firstline = mat->coo_col_id[index]/16;
	cacheperrow[i] = 1;
	elemperrow[i] = 1;
	index++;
	while (mat->coo_row_id[index] == i)
	{
	    int nextline = mat->coo_col_id[index]/16;
	    if (nextline != firstline)
	    {
		firstline = nextline;
		cacheperrow[i]++;
	    }
	    elemperrow[i]++;
	    index++;
	}
    }
    int maxcacheline = 0;
    int mincacheline = 100000000;
    int sum = 0;
    for (int i = 0; i < mat->matinfo.height; i++)
    {
	if (cacheperrow[i] < mincacheline)
	    mincacheline = cacheperrow[i];
	if (cacheperrow[i] > maxcacheline)
	    maxcacheline = cacheperrow[i];
	sum += cacheperrow[i];
    }
    printf("Cacheline usage per row: max %d min %d avg %f\n", maxcacheline, mincacheline, (double)sum/(double)mat->matinfo.height);
}






template <class dimType, class dataType>
void coo_spmv(coo_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    //for (dimType i = (dimType)0; i < mat->matinfo.height; i++)
	//result[i] = (dataType)0;
    dimType nnz = mat->matinfo.nnz;
    for (dimType i = (dimType)0; i < nnz; i++)
    {
	dimType row = mat->coo_row_id[i];
	dimType col = mat->coo_col_id[i];
	dataType data = mat->coo_data[i];
	result[row] += data * vec[col];
    }
}


void spmv_only(coo_matrix<int, float>* mat, float* vec, float* coores)
{
    int ressize = mat->matinfo.height;
    for (int i = 0; i < ressize; i++)
	coores[i] = (float)0;
    coo_spmv<int, float>(mat, vec, coores, mat->matinfo.width);
}



void pad_csr(csr_matrix<int, float>* source, csr_matrix<int, float>* dest, int alignment)
{
	using namespace std;	
	dest->matinfo.height = source->matinfo.height;
	dest->matinfo.width = source->matinfo.width;
	dest->csr_row_ptr = (int*)malloc(sizeof(int)*(source->matinfo.height+1));
	vector<int> padcol;
	vector<float> paddata;
	padcol.reserve(source->matinfo.nnz*2);
	paddata.reserve(source->matinfo.nnz*2);
	
	dest->csr_row_ptr[0] = 0;
	
	for (int row = 0; row < source->matinfo.height; row++)
	{
		int start = source->csr_row_ptr[row];
		int end = source->csr_row_ptr[row+1];
		int size = end - start;
		int paddedsize = findPaddedSize(size, alignment);
		dest->csr_row_ptr[row+1] = dest->csr_row_ptr[row] + paddedsize;
		int i = 0;
		for (; i < size; i++)
		{
			padcol.push_back(source->csr_col_id[start + i]);
			paddata.push_back(source->csr_data[start + i]);
		}
		int lastcol = padcol[padcol.size() - 1];
		for (; i < paddedsize; i++)
		{
			padcol.push_back(lastcol);
			paddata.push_back(0.0f);
		}
	}
	dest->csr_col_id = (int*)malloc(sizeof(int)*padcol.size());
	dest->csr_data = (float*)malloc(sizeof(float)*paddata.size());
	dest->matinfo.nnz = padcol.size();
	for (unsigned int i = 0; i < padcol.size(); i++)
	{
		dest->csr_col_id[i] = padcol[i];
		dest->csr_data[i] = paddata[i];
	}
}
