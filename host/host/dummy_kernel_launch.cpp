#include "common.h"
#include "testFilter.h"

#include "testScan.h"

using namespace aocl_utils;


// OpenCL Vars---------0 for CPU, 1 for GPU
extern cl_context Context;        // OpenCL context
extern cl_program Program;           // OpenCL program
extern cl_command_queue CommandQueue[2];// OpenCL command que
extern cl_platform_id Platform[2];      // OpenCL platform
extern cl_device_id Device[2];          // OpenCL device
extern cl_ulong totalLocalMemory[2];      /**< Max local memory allowed */
extern cl_device_id allDevices[10];
extern void validateFilter( Record* h_Rin, int beginPos, int rLen, 
					Record* h_Rout, int outSize, int smallKey, int largeKey);

void dummy_kernel_call(int _CPU_GPU)
{
//	size_t numThreadsPerBlock_x=numThreadPB;
//	size_t globalWorkingSetSize=numThreadPB*numBlock;
	cl_kernel Kernel;

	cl_getKernel("dummy",&Kernel, _CPU_GPU);

    int err =  clEnqueueTask(CommandQueue[0], Kernel, 0, NULL, NULL);//&events[K_DATA_IN]
    checkError(err, "enqueue dummy kernel task failed ");

    clFinish(CommandQueue[0]);

}
