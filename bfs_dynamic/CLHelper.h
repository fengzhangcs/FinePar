//------------------------------------------
//--cambine:helper function for OpenCL
//--programmer:	Jianbin Fang
//--date:	27/12/2010
//------------------------------------------
#ifndef _CL_HELPER_
#define _CL_HELPER_

#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
int cpuoffset ;



#include <pthread.h>
int rowInfo[11];
#define MAX 10

//void _clSetArgs(int kernel_id, int arg_idx, void * d_mem, int size = 0);
pthread_t thread[2];
pthread_mutex_t mut;
int number1=0;
int number2=0;
int work_group_size_global;





using std::string;
using std::ifstream;
using std::cerr;
using std::endl;
using std::cout;
//#pragma OPENCL EXTENSION cl_nv_compiler_options:enable
#define WORK_DIM 2	//work-items dimensions

struct oclHandleStruct
{
    cl_context              context;
    cl_device_id            *devices;
    cl_command_queue        *queue;
    //cl_command_queue        queue;
    cl_program              program;
    cl_int		cl_status;
    std::string error_str;
    std::vector<cl_kernel>  kernel;
};

struct oclHandleStruct oclHandles;

char kernel_file[100]  = "Kernels.cl";
int total_kernels = 3;
//int total_kernels = 2;
string kernel_names[3] = {"BFS_1", "BFS_2", "BFS_1_cpu"};
int work_group_size = 512;
int device_id_inused = 0; //deviced id used (default : 0)



 
void *thread1(void *)
{
  int i;
  //printf ("thread1 : I'm thread 1\n");
  int kernel_id=0;
	cl_uint work_dim = WORK_DIM;
  	size_t local_work_size[] = {work_group_size_global, 1};

  for (i = 0; i < MAX; i++)
  {
    //printf("thread1 : number1 = %d\n",number1);
    pthread_mutex_lock(&mut);
    if(number2>number1+1){
     number1++;

        int t1=rowInfo[number1+1]-rowInfo[number1];
        int  t2=work_group_size_global ;
        size_t global_work_size[]= {(t1/t2+(int)(t1%t2!=0))*t2, 1}; 
      //          _clSetArgs(0, 8, &number1, sizeof(int));
      oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[kernel_id], 8, sizeof(int), &number1);

        oclHandles.cl_status = clEnqueueNDRangeKernel(oclHandles.queue[0], oclHandles.kernel[kernel_id], work_dim, 0, \
                        global_work_size, local_work_size, 0 , 0, 0 );    
                        //global_work_size, local_work_size, 0 , 0, &(e[0]) );    

        //oclHandles.cl_status = clEnqueueNDRangeKernel(oclHandles.queue[1], \
         //   oclHandles.kernel[kernel_id2], work_dim, 0, \
          //  globalcpu, globallocal, 0 , 0, &(e[0]) ); 




      //errorCode = clSetKernelArg(csrKernel, 9, sizeof(int), &number1); CHECKERROR;
      //errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
    pthread_mutex_unlock(&mut);
      clFinish(oclHandles.queue[0]);
    }
    else{
      pthread_mutex_unlock(&mut);
      pthread_exit(NULL);
    }


 //   sleep(1);
  }


  //printf("thread1 :主函数在等我完成任务吗？\n");
  pthread_exit(NULL);
//return NULL;
}
void *thread2(void *)
{
  int i;
	cl_uint work_dim = WORK_DIM;
        size_t globalcpu[]={3,1},globallocal[]={1,1};
//  printf("thread2 : I'm thread 2\n");

  for (i = 0; i < MAX; i++)
  {
 //   printf("thread2 : number2 = %d\n",number2);
    pthread_mutex_lock(&mut);
    if(number2>number1+1){
      number2--;


      //          _clSetArgs(2, 8, &number2, sizeof(int));
      oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[2], 8, sizeof(int), &number2);
        oclHandles.cl_status = clEnqueueNDRangeKernel(oclHandles.queue[1], \
            oclHandles.kernel[2], work_dim, 0, \
            //oclHandles.kernel[2], work_dim, global_offsetstart, 
            globalcpu, globallocal, 0 , 0, 0 );   

        //errorCode = clSetKernelArg(csrKernelcpu, 11, sizeof(int), &number2); CHECKERROR;
      //errorCode = clEnqueueNDRangeKernel(cmdQueue_cpu, csrKernelcpu, work_dimcpu, NULL, &globalsizecpu, &localsizecpu, 0, NULL, NULL); CHECKERROR;
    pthread_mutex_unlock(&mut);
      clFinish(oclHandles.queue[1]);
    }
    else{
      pthread_mutex_unlock(&mut);
      pthread_exit(NULL);
    }
//    sleep(1);
  }


  //printf("thread2 :主函数在等我完成任务吗？\n");
  pthread_exit(NULL);
//return NULL;
}
void thread_create(void)
{
  int temp;
    number1=0; number2=9;
    number1--;
    number2++;
  memset(&thread, 0, sizeof(thread));          //comment1
  /*创建线程*/
  if((temp = pthread_create(&thread[0], NULL, thread1, NULL)) != 0)  //comment2     
    printf("线程1创建失败!\n");
  //else
   // printf("线程1被创建\n");

  if((temp = pthread_create(&thread[1], NULL, thread2, NULL)) != 0)  //comment3
    printf("线程2创建失败");
//  else
 //   printf("线程2被创建\n");
}
void thread_wait(void)
{
        /*等待线程结束*/
        if(thread[0] !=0)
           {             //comment4    
                pthread_join(thread[0],NULL);
  //              printf("线程1已经结束\n");
          }   
        if(thread[1] !=0) 
           {   
                //comment5
               pthread_join(thread[1],NULL);
   //             printf("线程2已经结束\n");
         }   
}






/*
 * Converts the contents of a file into a string
 */
string FileToString(const string fileName)
{
    ifstream f(fileName.c_str(), ifstream::in | ifstream::binary);

    try
    {
        size_t size;
        char*  str;
        string s;

        if(f.is_open())
        {
            size_t fileSize;
            f.seekg(0, ifstream::end);
            size = fileSize = f.tellg();
            f.seekg(0, ifstream::beg);

            str = new char[size+1];
            if (!str) throw(string("Could not allocate memory"));

            f.read(str, fileSize);
            f.close();
            str[size] = '\0';
        
            s = str;
            delete [] str;
            return s;
        }
    }
    catch(std::string msg)
    {
        cerr << "Exception caught in FileToString(): " << msg << endl;
        if(f.is_open())
            f.close();
    }
    catch(...)
    {
        cerr << "Exception caught in FileToString()" << endl;
        if(f.is_open())
            f.close();
    }
    string errorMsg = "FileToString()::Error: Unable to open file "
                            + fileName;
    throw(errorMsg);
}
//---------------------------------------
//Read command line parameters
//
void _clCmdParams(int argc, char* argv[]){
	for (int i =0; i < argc; ++i)
	{
		switch (argv[i][1])
		{
		case 'g':	//--g stands for size of work group
			if (++i < argc)
			{
				sscanf(argv[i], "%u", &work_group_size);
			}
			else
			{
				std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
				throw;
			}
			break;
		  case 'd':	 //--d stands for device id used in computaion
			if (++i < argc)
			{
				sscanf(argv[i], "%u", &device_id_inused);
			}
			else
			{
				std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
				throw;
			}
			break;
		default:
			;
		}
	}
	
}

//---------------------------------------
//Initlize CL objects
//--description: there are 5 steps to initialize all the OpenCL objects needed
//--revised on 04/01/2011: get the number of devices  and 
//  devices have no relationship with context
int _clInit()
{
    int DEVICE_ID_INUSED = device_id_inused;
    cl_int resultCL;
    
    oclHandles.context = NULL;
    oclHandles.devices = NULL;
    oclHandles.queue = NULL;
    oclHandles.program = NULL;

    cl_uint deviceListSize;

    //-----------------------------------------------
    //--cambine-1: find the available platforms and select one

    cl_uint numPlatforms;
    cl_platform_id targetPlatform = NULL;

    resultCL = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (resultCL != CL_SUCCESS)
        throw (string("InitCL()::Error: Getting number of platforms (clGetPlatformIDs)"));
    //printf("number of platforms:%d\n",numPlatforms);	//by cambine

    if (!(numPlatforms > 0))
        throw (string("InitCL()::Error: No platforms found (clGetPlatformIDs)"));

    cl_platform_id* allPlatforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));

    resultCL = clGetPlatformIDs(numPlatforms, allPlatforms, NULL);
    if (resultCL != CL_SUCCESS)
        throw (string("InitCL()::Error: Getting platform ids (clGetPlatformIDs)"));

    /* Select the target platform. Default: first platform */
    targetPlatform = allPlatforms[0];
    for (int i = 0; i < numPlatforms; i++)
    {
        char pbuff[128];
        resultCL = clGetPlatformInfo( allPlatforms[i],
                                        CL_PLATFORM_VENDOR,
                                        sizeof(pbuff),
                                        pbuff,
                                        NULL);
        if (resultCL != CL_SUCCESS)
            throw (string("InitCL()::Error: Getting platform info (clGetPlatformInfo)"));

		//printf("vedor is %s\n",pbuff);

    }
    free(allPlatforms);

    //////////////////////////////////////////////////
    cl_platform_id platform=targetPlatform ;
   cl_int  error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,NULL,&deviceListSize);
	if (error != CL_SUCCESS)
		return error;
	printf("deviceListSize :%d \n", deviceListSize);
	oclHandles.devices =(cl_device_id*)malloc(sizeof(cl_device_id)*deviceListSize);
    cl_device_id alldevice[2];
	error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,alldevice,NULL);
    if (error != CL_SUCCESS)
		return error;
	oclHandles.devices[0]=alldevice[0];
    error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_CPU,1,alldevice,NULL);
	if (error != CL_SUCCESS)
		return error;
	oclHandles.devices[1]=alldevice[0];
    char pbuf[128];
    error = clGetDeviceInfo(oclHandles.devices[0],
					CL_DEVICE_NAME,
                                       sizeof(pbuf),
                                        pbuf,
                                        NULL);
    if (error != CL_SUCCESS)
	return error;
    printf("devices0: %s\n", pbuf);
		
	error = clGetDeviceInfo(oclHandles.devices[1],
                                                        CL_DEVICE_NAME,
                                                        sizeof(pbuf),
                                                        pbuf,
                                                        NULL);
    if (error != CL_SUCCESS)
	return error;	
    printf("devices1: %s\n", pbuf);
	//====================================================================================================100
	//	CREATE CONTEXT FOR ALL devices
	//====================================================================================================100

	oclHandles.context = clCreateContext(0,2,oclHandles.devices,NULL,NULL,&error);
	if (error != CL_SUCCESS)
		return error;
              
	//====================================================================================================100
	//	CREATE COMMAND QUEUE FOR THE DEVICE
	//====================================================================================================100
	
	// Create a command queue
	oclHandles.queue =(cl_command_queue*)malloc(sizeof(cl_command_queue)*2);
	int CPU_GPU;
	for(CPU_GPU=0;CPU_GPU<2;CPU_GPU++){
		oclHandles.queue[CPU_GPU] = clCreateCommandQueue(oclHandles.context, 
											oclHandles.devices[CPU_GPU], 
											CL_QUEUE_PROFILING_ENABLE, 
											&error);
		if (error != CL_SUCCESS) 
			return error;
	}
	

	
   //--cambine-5: Load CL file, build CL program object, create CL kernel object
    std::string  source_str = FileToString(kernel_file);
    const char * source    = source_str.c_str();
    size_t sourceSize[]    = { source_str.length() };

    oclHandles.program = clCreateProgramWithSource(oclHandles.context, 
                                                    1, 
                                                    &source,
                                                    sourceSize,
                                                    &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL))
        throw(string("InitCL()::Error: Loading Binary into cl_program. (clCreateProgramWithBinary)"));    
    //insert debug information
    //std::string options= "-cl-nv-verbose"; //Doesn't work on AMD machines
    //options += " -cl-nv-opt-level=3";
    resultCL = clBuildProgram(oclHandles.program, deviceListSize, oclHandles.devices, NULL, NULL,NULL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL))
    {
        cerr << "InitCL()::Error: In clBuildProgram" << endl;

		size_t length;
        resultCL = clGetProgramBuildInfo(oclHandles.program, 
                                        oclHandles.devices[1], 
                                        CL_PROGRAM_BUILD_LOG, 
                                        0, 
                                        NULL, 
                                        &length);
        if(resultCL != CL_SUCCESS) 
            throw(string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

		char* buffer = (char*)malloc(length);
        resultCL = clGetProgramBuildInfo(oclHandles.program, 
                                        oclHandles.devices[1], 
                                        CL_PROGRAM_BUILD_LOG, 
                                        length, 
                                        buffer, 
                                        NULL);
        if(resultCL != CL_SUCCESS) 
            throw(string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

		cerr << buffer << endl;
        free(buffer);

        throw(string("InitCL()::Error: Building Program (clBuildProgram)"));
    } 

    //get program information in intermediate representation
    #ifdef PTX_MSG    
    size_t binary_sizes[deviceListSize];
    char * binaries[deviceListSize];
    //figure out number of devices and the sizes of the binary for each device. 
    oclHandles.cl_status = clGetProgramInfo(oclHandles.program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*deviceListSize, &binary_sizes, NULL );
    if(oclHandles.cl_status!=CL_SUCCESS){
        throw(string("--cambine:exception in _InitCL -> clGetProgramInfo-2"));
    }

    std::cout<<"--cambine:"<<binary_sizes<<std::endl;
    //copy over all of the generated binaries. 
    for(int i=0;i<deviceListSize;i++)
	binaries[i] = (char *)malloc( sizeof(char)*(binary_sizes[i]+1));
    oclHandles.cl_status = clGetProgramInfo(oclHandles.program, CL_PROGRAM_BINARIES, sizeof(char *)*deviceListSize, binaries, NULL );
    if(oclHandles.cl_status!=CL_SUCCESS){
        throw(string("--cambine:exception in _InitCL -> clGetProgramInfo-3"));
    }
    for(int i=0;i<deviceListSize;i++)
      binaries[i][binary_sizes[i]] = '\0';
    std::cout<<"--cambine:writing ptd information..."<<std::endl;
    FILE * ptx_file = fopen("cl.ptx","w");
    if(ptx_file==NULL){
	throw(string("exceptions in allocate ptx file."));
    }
    fprintf(ptx_file,"%s",binaries[DEVICE_ID_INUSED]);
    fclose(ptx_file);
    std::cout<<"--cambine:writing ptd information done."<<std::endl;
    for(int i=0;i<deviceListSize;i++)
	free(binaries[i]);
    #endif

    for (int nKernel = 0; nKernel < total_kernels; nKernel++)
    {
        /* get a kernel object handle for a kernel with the given name */
        cl_kernel kernel = clCreateKernel(oclHandles.program,
                                            (kernel_names[nKernel]).c_str(),
                                            &resultCL);

        if ((resultCL != CL_SUCCESS) || (kernel == NULL))
        {
            string errorMsg = "InitCL()::Error: Creating Kernel (clCreateKernel) \"" + kernel_names[nKernel] + "\"";
            throw(errorMsg);
        }

        oclHandles.kernel.push_back(kernel);
    }
  //get resource alocation information
    #ifdef RES_MSG
    char * build_log;
    size_t ret_val_size;
    oclHandles.cl_status = clGetProgramBuildInfo(oclHandles.program, oclHandles.devices[DEVICE_ID_INUSED], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    if(oclHandles.cl_status!=CL_SUCCESS){
	throw(string("exceptions in _InitCL -> getting resource information"));
    }    

    build_log = (char *)malloc(ret_val_size+1);
    oclHandles.cl_status = clGetProgramBuildInfo(oclHandles.program, oclHandles.devices[DEVICE_ID_INUSED], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    if(oclHandles.cl_status!=CL_SUCCESS){
	throw(string("exceptions in _InitCL -> getting resources allocation information-2"));
    }
    build_log[ret_val_size] = '\0';
    std::cout<<"--cambine:"<<build_log<<std::endl;
    free(build_log);
    #endif
}

//---------------------------------------
//release CL objects
void _clRelease()
{
    char errorFlag = false;

    for (int nKernel = 0; nKernel < oclHandles.kernel.size(); nKernel++)
    {
        if (oclHandles.kernel[nKernel] != NULL)
        {
            cl_int resultCL = clReleaseKernel(oclHandles.kernel[nKernel]);
            if (resultCL != CL_SUCCESS)
            {
                cerr << "ReleaseCL()::Error: In clReleaseKernel" << endl;
                errorFlag = true;
            }
            oclHandles.kernel[nKernel] = NULL;
        }
        oclHandles.kernel.clear();
    }

    if (oclHandles.program != NULL)
    {
        cl_int resultCL = clReleaseProgram(oclHandles.program);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCL()::Error: In clReleaseProgram" << endl;
            errorFlag = true;
        }
        oclHandles.program = NULL;
    }

    if (oclHandles.queue != NULL)
    {
        cl_int resultCL = clReleaseCommandQueue(oclHandles.queue[0]);
       	resultCL |= clReleaseCommandQueue(oclHandles.queue[1]);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCL()::Error: In clReleaseCommandQueue" << endl;
            errorFlag = true;
        }
        oclHandles.queue = NULL;
    }

    free(oclHandles.devices);

    if (oclHandles.context != NULL)
    {
        cl_int resultCL = clReleaseContext(oclHandles.context);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCL()::Error: In clReleaseContext" << endl;
            errorFlag = true;
        }
        oclHandles.context = NULL;
    }

    if (errorFlag) throw(string("ReleaseCL()::Error encountered."));
}
//--------------------------------------------------------
//--cambine:create buffer and then copy data from host to device
cl_mem _clCreateAndCpyMem(int size, void * h_mem_source) throw(string){
	cl_mem d_mem;
	d_mem = clCreateBuffer(oclHandles.context,	CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,  \
									size, h_mem_source, &oclHandles.cl_status);
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateAndCpyMem()"));
	#endif
	return d_mem;
}
//-------------------------------------------------------
//--cambine:	create read only  buffer for devices
//--date:	17/01/2011	
cl_mem _clMallocRW(int size, void * h_mem_ptr) throw(string){
 	cl_mem d_mem;
	d_mem = clCreateBuffer(oclHandles.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR|CL_MEM_ALLOC_HOST_PTR, size, h_mem_ptr, &oclHandles.cl_status);
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clMallocRW"));
	#endif
	return d_mem;
}
//-------------------------------------------------------
//--cambine:	create read and write buffer for devices
//--date:	17/01/2011	
cl_mem _clMalloc(int size, void * h_mem_ptr) throw(string){
 	cl_mem d_mem;
	d_mem = clCreateBuffer(oclHandles.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_mem_ptr, &oclHandles.cl_status);
	//d_mem = clCreateBuffer(oclHandles.context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, size, h_mem_ptr, &oclHandles.cl_status);
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clMalloc"));
	#endif
	return d_mem;
}
cl_mem _clMalloc_cpu(int size, void * h_mem_ptr) throw(string){
 	cl_mem d_mem;
	d_mem = clCreateBuffer(oclHandles.context, CL_MEM_READ_ONLY |CL_MEM_ALLOC_HOST_PTR| CL_MEM_COPY_HOST_PTR, size, h_mem_ptr, &oclHandles.cl_status);
	//d_mem = clCreateBuffer(oclHandles.context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, size, h_mem_ptr, &oclHandles.cl_status);
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clMalloc"));
	#endif
	return d_mem;
}


//-------------------------------------------------------
//--cambine:	transfer data from host to device
//--date:	17/01/2011
void _clMemcpyH2D(cl_mem d_mem, int size, const void *h_mem_ptr) throw(string){
	oclHandles.cl_status = clEnqueueWriteBuffer(oclHandles.queue[0], d_mem, CL_TRUE, 0, size, h_mem_ptr, 0, NULL, NULL);
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clMemcpyH2D"));
	#endif
}
//--------------------------------------------------------
//--cambine:create buffer and then copy data from host to device with pinned 
// memory
cl_mem _clCreateAndCpyPinnedMem(int size, float* h_mem_source) throw(string){
	cl_mem d_mem, d_mem_pinned;
	float * h_mem_pinned = NULL;
	d_mem_pinned = clCreateBuffer(oclHandles.context,	CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,  \
									size, NULL, &oclHandles.cl_status);
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateAndCpyMem()->d_mem_pinned"));
	#endif
	//------------
	d_mem = clCreateBuffer(oclHandles.context,	CL_MEM_READ_ONLY,  \
									size, NULL, &oclHandles.cl_status);
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateAndCpyMem() -> d_mem "));
	#endif
	//----------
	h_mem_pinned = (cl_float *)clEnqueueMapBuffer(oclHandles.queue[0], d_mem_pinned, CL_TRUE,  \
										CL_MAP_WRITE, 0, size, 0, NULL,  \
										NULL,  &oclHandles.cl_status);
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateAndCpyMem() -> clEnqueueMapBuffer"));
	#endif
	int element_number = size/sizeof(float);
	#pragma omp parallel for
	for(int i=0;i<element_number;i++){
		h_mem_pinned[i] = h_mem_source[i];
	}
	//----------
	oclHandles.cl_status = clEnqueueWriteBuffer(oclHandles.queue[0], d_mem, 	\
									CL_TRUE, 0, size, h_mem_pinned,  \
									0, NULL, NULL);
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateAndCpyMem() -> clEnqueueWriteBuffer"));
	#endif
	
	return d_mem;
}


//--------------------------------------------------------
//--cambine:create write only buffer on device
cl_mem _clMallocWO(int size) throw(string){
	cl_mem d_mem;
	d_mem = clCreateBuffer(oclHandles.context, CL_MEM_WRITE_ONLY, size, 0, &oclHandles.cl_status);
	#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateMem()"));
	#endif
	return d_mem;
}

//--------------------------------------------------------
//transfer data from device to host
void _clMemcpyD2H(cl_mem d_mem, int size, void * h_mem) throw(string){
	oclHandles.cl_status = clEnqueueReadBuffer(oclHandles.queue[0], d_mem, CL_TRUE, 0, size, h_mem, 0,0,0);
	#ifdef ERRMSG
		oclHandles.error_str = "excpetion in _clCpyMemD2H -> ";
		switch(oclHandles.cl_status){
			case CL_INVALID_COMMAND_QUEUE:
				oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
				break;
			case CL_INVALID_CONTEXT:
				oclHandles.error_str += "CL_INVALID_CONTEXT";
				break;	
			case CL_INVALID_MEM_OBJECT:
				oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
				break;
			case CL_INVALID_VALUE:
				oclHandles.error_str += "CL_INVALID_VALUE";
				break;	
			case CL_INVALID_EVENT_WAIT_LIST:
				oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
				break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
				break;	
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;		
			default:
				oclHandles.error_str += "Unknown reason";
				break;
		}
		if(oclHandles.cl_status != CL_SUCCESS)
			throw(oclHandles.error_str);
	#endif
}

//--------------------------------------------------------
//set kernel arguments
void _clSetArgs(int kernel_id, int arg_idx, void * d_mem, int size = 0) throw(string){
	if(!size){
		oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[kernel_id], arg_idx, sizeof(d_mem), &d_mem);
		#ifdef ERRMSG
		oclHandles.error_str = "excpetion in _clSetKernelArg() ";
		switch(oclHandles.cl_status){
			case CL_INVALID_KERNEL:
				oclHandles.error_str += "CL_INVALID_KERNEL";
				break;
			case CL_INVALID_ARG_INDEX:
				oclHandles.error_str += "CL_INVALID_ARG_INDEX";
				break;	
			case CL_INVALID_ARG_VALUE:
				oclHandles.error_str += "CL_INVALID_ARG_VALUE";
				break;
			case CL_INVALID_MEM_OBJECT:
				oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
				break;	
			case CL_INVALID_SAMPLER:
				oclHandles.error_str += "CL_INVALID_SAMPLER";
				break;
			case CL_INVALID_ARG_SIZE:
				oclHandles.error_str += "CL_INVALID_ARG_SIZE";
				break;	
			case CL_OUT_OF_RESOURCES:
				oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				break;
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;
			default:
				oclHandles.error_str += "Unknown reason";
				break;
		}
		if(oclHandles.cl_status != CL_SUCCESS)
			throw(oclHandles.error_str);
		#endif
	}
	else{
		oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[kernel_id], arg_idx, size, d_mem);
		#ifdef ERRMSG
		oclHandles.error_str = "excpetion in _clSetKernelArg() ";
		switch(oclHandles.cl_status){
			case CL_INVALID_KERNEL:
				oclHandles.error_str += "CL_INVALID_KERNEL";
				break;
			case CL_INVALID_ARG_INDEX:
				oclHandles.error_str += "CL_INVALID_ARG_INDEX";
				break;	
			case CL_INVALID_ARG_VALUE:
				oclHandles.error_str += "CL_INVALID_ARG_VALUE";
				break;
			case CL_INVALID_MEM_OBJECT:
				oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
				break;	
			case CL_INVALID_SAMPLER:
				oclHandles.error_str += "CL_INVALID_SAMPLER";
				break;
			case CL_INVALID_ARG_SIZE:
				oclHandles.error_str += "CL_INVALID_ARG_SIZE";
				break;	
			case CL_OUT_OF_RESOURCES:
				oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				break;
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;
			default:
				oclHandles.error_str += "Unknown reason";
				break;
		}
		if(oclHandles.cl_status != CL_SUCCESS)
			throw(oclHandles.error_str);
		#endif
	}
}
void _clFinish() throw(string){
	oclHandles.cl_status = clFinish(oclHandles.queue[0]);	
	#ifdef ERRMSG
	oclHandles.error_str = "excpetion in _clFinish";
	switch(oclHandles.cl_status){
		case CL_INVALID_COMMAND_QUEUE:
			oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
			break;
		case CL_OUT_OF_RESOURCES:
			oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			break;		
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;
		default:
			oclHandles.error_str += "Unknown reasons";
			break;

	}
	if(oclHandles.cl_status!=CL_SUCCESS){
		throw(oclHandles.error_str);
	}
	#endif
}
//--------------------------------------------------------
//--cambine:enqueue kernel
void _clInvokeKernel_fusion(int kernel_id, int kernel_id2, int work_items, int work_group_size) throw(string){
        work_group_size_global=work_group_size;
	cl_uint work_dim = WORK_DIM;
	cl_event e[1];
	if(work_items%work_group_size != 0)	//process situations that work_items cannot be divided by work_group_size
	  work_items = work_items + (work_group_size-(work_items%work_group_size));
  	size_t local_work_size[] = {work_group_size, 1};

        size_t globalcpu[]={3,1},globallocal[]={1,1};

thread_create();
thread_wait();




/*
        int t1=rowInfo[number1+1]-rowInfo[number1];
        //int t1=work_items*((double)cpuoffset/100);
        int  t2=work_group_size;
        size_t global_work_size[]= {(t1/t2+(int)(t1%t2!=0))*t2, 1};
        //size_t global_work_sizecpu[]= {(t1/t2+(int)(t1%t2!=0))*t2, 1};
	//size_t global_work_size[] = {work_items-global_work_sizecpu[0], 1};
	//size_t global_work_size[] = {work_items, 1};
        //size_t global_offsetstart[] = {global_work_size[0],0};
        //size_t global_offsetstart[] = {global_work_size[0],0};


	oclHandles.cl_status = clEnqueueNDRangeKernel(oclHandles.queue[0], oclHandles.kernel[kernel_id], work_dim, 0, \
			global_work_size, local_work_size, 0 , 0, &(e[0]) );	

	oclHandles.cl_status = clEnqueueNDRangeKernel(oclHandles.queue[1], \
            oclHandles.kernel[kernel_id2], work_dim, global_offsetstart, \
            globalcpu, globallocal, 0 , 0, &(e[0]) );	
            //global_work_sizecpu, local_work_size, 0 , 0, &(e[0]) );	
            */

        clFinish(oclHandles.queue[0]);
        clFinish(oclHandles.queue[1]);



	#ifdef ERRMSG
	oclHandles.error_str = "excpetion in _clInvokeKernel() -> ";
	switch(oclHandles.cl_status)
	{
		case CL_INVALID_PROGRAM_EXECUTABLE:
			oclHandles.error_str += "CL_INVALID_PROGRAM_EXECUTABLE";
			break;
		case CL_INVALID_COMMAND_QUEUE:
			oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
			break;
		case CL_INVALID_KERNEL:
			oclHandles.error_str += "CL_INVALID_KERNEL";
			break;
		case CL_INVALID_CONTEXT:
			oclHandles.error_str += "CL_INVALID_CONTEXT";
			break;
		case CL_INVALID_KERNEL_ARGS:
			oclHandles.error_str += "CL_INVALID_KERNEL_ARGS";
			break;
		case CL_INVALID_WORK_DIMENSION:
			oclHandles.error_str += "CL_INVALID_WORK_DIMENSION";
			break;
		case CL_INVALID_GLOBAL_WORK_SIZE:
			oclHandles.error_str += "CL_INVALID_GLOBAL_WORK_SIZE";
			break;
		case CL_INVALID_WORK_GROUP_SIZE:
			oclHandles.error_str += "CL_INVALID_WORK_GROUP_SIZE";
			break;
		case CL_INVALID_WORK_ITEM_SIZE:
			oclHandles.error_str += "CL_INVALID_WORK_ITEM_SIZE";
			break;
		case CL_INVALID_GLOBAL_OFFSET:
			oclHandles.error_str += "CL_INVALID_GLOBAL_OFFSET";
			break;
		case CL_OUT_OF_RESOURCES:
			oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
			break;
		case CL_INVALID_EVENT_WAIT_LIST:
			oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;
		default: 
			oclHandles.error_str += "Unkown reseason";
			break;		
	}
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(oclHandles.error_str);	
	#endif
	//_clFinish();
	// oclHandles.cl_status = clWaitForEvents(1, &e[0]);
	// #ifdef ERRMSG
        // if (oclHandles.cl_status!= CL_SUCCESS)
        //     throw(string("excpetion in _clEnqueueNDRange() -> clWaitForEvents"));
	// #endif
}
//--------------------------------------------------------
//--cambine:enqueue kernel
void _clInvokeKernel(int kernel_id, int work_items, int work_group_size) throw(string){
	cl_uint work_dim = WORK_DIM;
	cl_event e[1];
	if(work_items%work_group_size != 0)	//process situations that work_items cannot be divided by work_group_size
	  work_items = work_items + (work_group_size-(work_items%work_group_size));
  	size_t local_work_size[] = {work_group_size, 1};
	size_t global_work_size[] = {work_items, 1};

        /*
        int t1=work_items*((double)cpuoffset/100), t2=work_group_size;
        size_t global_work_sizecpu[]= {(t1/t2+(int)(t1%t2!=0))*t2, 1};
	size_t global_work_size[] = {work_items-global_work_sizecpu[0], 1};
	//size_t global_work_size[] = {work_items, 1};
        size_t global_offsetstart[] = {global_work_size[0],0};
        */
	oclHandles.cl_status = clEnqueueNDRangeKernel(oclHandles.queue[0], oclHandles.kernel[kernel_id], work_dim, 0, \
											global_work_size, local_work_size, 0 , 0, &(e[0]) );	

        /*
	oclHandles.cl_status = clEnqueueNDRangeKernel(oclHandles.queue[1], \
            oclHandles.kernel[kernel_id], work_dim, global_offsetstart, \
            global_work_sizecpu, local_work_size, 0 , 0, &(e[0]) );	

        clFinish(oclHandles.queue[0]);
        clFinish(oclHandles.queue[1]);
        */



	#ifdef ERRMSG
	oclHandles.error_str = "excpetion in _clInvokeKernel() -> ";
	switch(oclHandles.cl_status)
	{
		case CL_INVALID_PROGRAM_EXECUTABLE:
			oclHandles.error_str += "CL_INVALID_PROGRAM_EXECUTABLE";
			break;
		case CL_INVALID_COMMAND_QUEUE:
			oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
			break;
		case CL_INVALID_KERNEL:
			oclHandles.error_str += "CL_INVALID_KERNEL";
			break;
		case CL_INVALID_CONTEXT:
			oclHandles.error_str += "CL_INVALID_CONTEXT";
			break;
		case CL_INVALID_KERNEL_ARGS:
			oclHandles.error_str += "CL_INVALID_KERNEL_ARGS";
			break;
		case CL_INVALID_WORK_DIMENSION:
			oclHandles.error_str += "CL_INVALID_WORK_DIMENSION";
			break;
		case CL_INVALID_GLOBAL_WORK_SIZE:
			oclHandles.error_str += "CL_INVALID_GLOBAL_WORK_SIZE";
			break;
		case CL_INVALID_WORK_GROUP_SIZE:
			oclHandles.error_str += "CL_INVALID_WORK_GROUP_SIZE";
			break;
		case CL_INVALID_WORK_ITEM_SIZE:
			oclHandles.error_str += "CL_INVALID_WORK_ITEM_SIZE";
			break;
		case CL_INVALID_GLOBAL_OFFSET:
			oclHandles.error_str += "CL_INVALID_GLOBAL_OFFSET";
			break;
		case CL_OUT_OF_RESOURCES:
			oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
			break;
		case CL_INVALID_EVENT_WAIT_LIST:
			oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;
		default: 
			oclHandles.error_str += "Unkown reseason";
			break;		
	}
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(oclHandles.error_str);	
	#endif
	//_clFinish();
	// oclHandles.cl_status = clWaitForEvents(1, &e[0]);
	// #ifdef ERRMSG
        // if (oclHandles.cl_status!= CL_SUCCESS)
        //     throw(string("excpetion in _clEnqueueNDRange() -> clWaitForEvents"));
	// #endif
}
void _clInvokeKernel2D(int kernel_id, int range_x, int range_y, int group_x, int group_y) throw(string){
	cl_uint work_dim = WORK_DIM;
	size_t local_work_size[] = {group_x, group_y};
	size_t global_work_size[] = {range_x, range_y};
	cl_event e[1];
	/*if(work_items%work_group_size != 0)	//process situations that work_items cannot be divided by work_group_size
	  work_items = work_items + (work_group_size-(work_items%work_group_size));*/
	oclHandles.cl_status = clEnqueueNDRangeKernel(oclHandles.queue[0], oclHandles.kernel[kernel_id], work_dim, 0, \
											global_work_size, local_work_size, 0 , 0, &(e[0]) );	
	#ifdef ERRMSG
	oclHandles.error_str = "excpetion in _clInvokeKernel() -> ";
	switch(oclHandles.cl_status)
	{
		case CL_INVALID_PROGRAM_EXECUTABLE:
			oclHandles.error_str += "CL_INVALID_PROGRAM_EXECUTABLE";
			break;
		case CL_INVALID_COMMAND_QUEUE:
			oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
			break;
		case CL_INVALID_KERNEL:
			oclHandles.error_str += "CL_INVALID_KERNEL";
			break;
		case CL_INVALID_CONTEXT:
			oclHandles.error_str += "CL_INVALID_CONTEXT";
			break;
		case CL_INVALID_KERNEL_ARGS:
			oclHandles.error_str += "CL_INVALID_KERNEL_ARGS";
			break;
		case CL_INVALID_WORK_DIMENSION:
			oclHandles.error_str += "CL_INVALID_WORK_DIMENSION";
			break;
		case CL_INVALID_GLOBAL_WORK_SIZE:
			oclHandles.error_str += "CL_INVALID_GLOBAL_WORK_SIZE";
			break;
		case CL_INVALID_WORK_GROUP_SIZE:
			oclHandles.error_str += "CL_INVALID_WORK_GROUP_SIZE";
			break;
		case CL_INVALID_WORK_ITEM_SIZE:
			oclHandles.error_str += "CL_INVALID_WORK_ITEM_SIZE";
			break;
		case CL_INVALID_GLOBAL_OFFSET:
			oclHandles.error_str += "CL_INVALID_GLOBAL_OFFSET";
			break;
		case CL_OUT_OF_RESOURCES:
			oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
			break;
		case CL_INVALID_EVENT_WAIT_LIST:
			oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;
		default: 
			oclHandles.error_str += "Unkown reseason";
			break;		
	}
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(oclHandles.error_str);	
	#endif
	//_clFinish();
	/*oclHandles.cl_status = clWaitForEvents(1, &e[0]);

	#ifdef ERRMSG

        if (oclHandles.cl_status!= CL_SUCCESS)

            throw(string("excpetion in _clEnqueueNDRange() -> clWaitForEvents"));

	#endif*/
}

//--------------------------------------------------------
//release OpenCL objects
void _clFree(cl_mem ob) throw(string){
	if(ob!=NULL)
		oclHandles.cl_status = clReleaseMemObject(ob);	
	#ifdef ERRMSG
	oclHandles.error_str = "excpetion in _clFree() ->";
	switch(oclHandles.cl_status)
	{
		case CL_INVALID_MEM_OBJECT:
			oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
			break;
		case CL_OUT_OF_RESOURCES:
			oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;			
		default: 
			oclHandles.error_str += "Unkown reseason";
			break;		
	}        
    if (oclHandles.cl_status!= CL_SUCCESS)
       throw(oclHandles.error_str);
	#endif
}
#endif //_CL_HELPER_
