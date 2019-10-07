//
// wzk, Xtra, SCE, NTU, Singapore.
#include "common.h"
#include "testScan.h"
//#include "KernelScheduler.h"
//#include "Helper.h"
//#include "PrimitiveCommon.h"
//#include "KernelScheduler.h"

using namespace aocl_utils;


// OpenCL Vars---------0 for CPU, 1 for GPU
extern cl_context Context;        // OpenCL context
extern cl_program Program;           // OpenCL program
extern cl_command_queue CommandQueue[2];// OpenCL command que
extern cl_platform_id Platform[2];      // OpenCL platform
extern cl_device_id Device[2];          // OpenCL device
extern cl_ulong totalLocalMemory[2];      /**< Max local memory allowed */
extern cl_device_id allDevices[10];
static cl_ulong usedLocalMemory;       /**< Used local memory by kernel */
extern void validateScan( int* input, int rLen, int* output );

double time_slot[4];

void initScan(int rLen,ScanPara* SP){
	SP->gLength=rLen;
	SP->blockSize=256;
	usedLocalMemory=0;
	cl_int status;
	float t = log((float)SP->gLength) / log((float)SP->blockSize);
    SP->pass = (cl_uint)t;

    // If t is equal to SP->pass
    if(fabs(t - (float)SP->pass) < 1e-7)
    {
       SP->pass--;
    }

	/* Allocate output buffers */
    SP->outputBuffer = (cl_mem*)malloc(SP->pass * sizeof(cl_mem));

    for(int i = 0; i < (int)SP->pass; i++)
    {
        int size = (int)(SP->gLength / pow((float)SP->blockSize,(float)i));
		SP->outputBuffer[i] = clCreateBuffer(
            Context, 
            CL_MEM_READ_WRITE,
            sizeof(cl_int) * size,
            0, 
            &status);
		assert(status==CL_SUCCESS);

    }

    /* Allocate SP->blockSumBuffers */
    SP->blockSumBuffer = (cl_mem*)malloc(SP->pass * sizeof(cl_mem));

    for(int i = 0; i < (int)SP->pass; i++)
    {
        int size = (int)(SP->gLength / pow((float)SP->blockSize,(float)(i + 1)));
        SP->blockSumBuffer[i] = clCreateBuffer(
            Context, 
            CL_MEM_READ_WRITE,
            sizeof(cl_int) * size,
            0, 
            &status);

	assert(status==CL_SUCCESS);

    }

    /* Create a SP->tempBuffer on device */
    int tempLength = (int)(SP->gLength / pow((float)SP->blockSize, (float)SP->pass));

   SP->tempBuffer = clCreateBuffer(Context,
        CL_MEM_READ_WRITE,
        sizeof(cl_int) * tempLength,
        0,
        &status);
	assert(status==CL_SUCCESS);

}

void bScan_int(cl_uint len, 
                       cl_mem *inputBuffer, 
                       cl_mem *outputBuffer,
                       cl_mem *blockSumBuffer,
					   int *index,
					   cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,ScanPara* SP,int _CPU_GPU)
{
    //cl_event events[1];
    /* set the block size*/ 
    size_t globalThreads[1]= {len / 2};
    size_t localThreads[1] = {SP->blockSize / 2};
	cl_getKernel("ScanLargeArrays_kernel",Kernel, _CPU_GPU);
    /*assert(localThreads[0] > maxWorkItemSizes[0] ||
       localThreads[0] > maxWorkGroupSize)
    {
        std::cout<<"Unsupported: Device does not"
            "support requested number of work items.";

        return SDK_FAILURE;
    }*/

    /* Set appropriate arguments to the kernel */
    /* 1st argument to the kernel - SP->outputBuffer */
    cl_int status = clSetKernelArg( (*Kernel), 0, sizeof(cl_mem), (void *)outputBuffer);
    /* 2nd argument to the kernel - inputBuffer */
    status |= clSetKernelArg( (*Kernel), 1, sizeof(cl_mem),  (void *)inputBuffer);
    /* 3rd argument to the kernel - local memory */
 //   status |= clSetKernelArg( (*Kernel),  2,  SP->blockSize * sizeof(cl_int), NULL);
    /* 4th argument to the kernel - block_size  */
    status |= clSetKernelArg((*Kernel), 2, sizeof(cl_int),&SP->blockSize);
    /* 5th argument to the kernel - SP->gLength  */
    status |= clSetKernelArg((*Kernel), 3, sizeof(cl_int), &len);
    /* 6th argument to the kernel - sum of blocks  */
    status |= clSetKernelArg( (*Kernel), 4, sizeof(cl_mem), blockSumBuffer);

    checkError(status, "Error in clSetKernelArg");
/*
    status = clGetKernelWorkGroupInfo((*Kernel),
        Device[0],
        CL_KERNEL_LOCAL_MEM_SIZE,
        sizeof(cl_ulong),
        &usedLocalMemory,
        NULL);
	assert(status==CL_SUCCESS);
    if(usedLocalMemory > totalLocalMemory[0])
    {
        printf("Unsupported: Insufficient local memory on device.\n");
    }
*/
    /* Enqueue a kernel run call.*/
	kernel_enqueue(SP->blockSize,19,1, globalThreads, localThreads,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
}

void pScan_int(cl_uint len,
                       cl_mem *inputBuffer,
                       cl_mem *outputBuffer,
					   int *index,
					   cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,ScanPara* SP,int _CPU_GPU)
{
    cl_int status;
    cl_getKernel("prefixSum_kernel",Kernel, _CPU_GPU);
    size_t globalThreads[1]= {len/2};
    size_t localThreads[1] = {len/2};
/*
    status = clGetKernelWorkGroupInfo((*Kernel),
        Device[0],
        CL_KERNEL_LOCAL_MEM_SIZE,
        sizeof(cl_ulong),
        &usedLocalMemory,
        NULL);
	assert(status==CL_SUCCESS);
    if(usedLocalMemory > totalLocalMemory[0])
    {
        printf("Unsupported: Insufficient local memory on device.\n");
    }
*/
    /*if(localThreads[0] > maxWorkItemSizes[0] ||
        localThreads[0] > maxWorkGroupSize)
    {
        std::cout << "Unsupported: Device does not support"
            "requested number of work items." << std::endl;
        return SDK_SUCCESS;
    }*/


    /* Set appropriate arguments to the kernel */
    /* 1st argument to the kernel - SP->outputBuffer */
    status = clSetKernelArg((*Kernel), 0, sizeof(cl_mem), (void *)outputBuffer);
	assert(status==CL_SUCCESS);

    /* 2nd argument to the kernel - inputBuffer */
    status = clSetKernelArg((*Kernel), 1, sizeof(cl_mem), (void *)inputBuffer);
	assert(status==CL_SUCCESS);

    /* 3rd argument to the kernel - local memory */
  //  status = clSetKernelArg( (*Kernel), 2, len * sizeof(cl_int), NULL);
  //  assert(status==CL_SUCCESS);

    /* 4th argument to the kernel - SP->gLength */
    status = clSetKernelArg( (*Kernel),2, sizeof(cl_int),(void*)&len);
	assert(status==CL_SUCCESS);

    /* Enqueue a kernel run call.*/
	kernel_enqueue(SP->blockSize,18,1, globalThreads, localThreads,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
}
void bAddition_int(cl_uint len,
                           cl_mem *inputBuffer,
                           cl_mem *outputBuffer, 
						   int *index,
						  cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,ScanPara* SP,int _CPU_GPU)
{
    cl_int status;
    cl_getKernel("blockAddition_kernel",Kernel, _CPU_GPU);
    /* set the block size*/ 
    size_t globalThreads[1]= {len};
    size_t localThreads[1] = {SP->blockSize};

   
    /*** Set appropriate arguments to the kernel ***/
    /* 1st argument to the kernel - inputBuffer */
    status = clSetKernelArg((*Kernel), 0, sizeof(cl_mem), (void*)inputBuffer);
	assert(status==CL_SUCCESS);

    /* 2nd argument to the kernel - SP->outputBuffer */
    status = clSetKernelArg((*Kernel), 1, sizeof(cl_mem), (void *)outputBuffer);
	assert(status==CL_SUCCESS);

/*
    status = clGetKernelWorkGroupInfo((*Kernel),
        Device[0],
        CL_KERNEL_LOCAL_MEM_SIZE,
        sizeof(cl_ulong),
        &usedLocalMemory,
        NULL);
	assert(status==CL_SUCCESS);
    if(usedLocalMemory > totalLocalMemory[0])
    {
        printf("Unsupported: Insufficient local memory on device.\n");
    }
*/
	    /* Enqueue a kernel run call.*/
	kernel_enqueue(SP->blockSize,17,1, globalThreads, localThreads,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
}

void closeScan(ScanPara* SP){
    cl_int status;
	cl_uint refCount;
	status = clGetMemObjectInfo(SP->tempBuffer,
                		        CL_MEM_REFERENCE_COUNT,
							    sizeof(cl_uint),
						        &refCount,
								NULL);
	//printf("refCount: %d\n", refCount);
	while(refCount != 0)
	{
		//printf("release tempBuffer\n\n");
		CL_FREE(SP->tempBuffer); //reduce count by 1	
		//printf("success release tempBuffer\n");
		refCount--;
	}
    for(int i = 0; i < (int)SP->pass; i++)
    {
		status = clGetMemObjectInfo(SP->outputBuffer[i],
                		            CL_MEM_REFERENCE_COUNT,
									sizeof(cl_uint),
									&refCount,
									NULL);
		while(refCount != 0)
		{
			//printf("release outputBuffer\n\n");
			CL_FREE(SP->outputBuffer[i]); //reduce count by 1
			refCount--;
			//printf("success release outputBuffer\n");
		}

		status = clGetMemObjectInfo(SP->blockSumBuffer[i],
                		            CL_MEM_REFERENCE_COUNT,
									sizeof(cl_uint),
									&refCount,
									NULL);
		while(refCount != 0)
		{
			//printf("release blockSumBuffer\n\n");
			CL_FREE(SP->blockSumBuffer[i]); //reduce count by 1
			refCount--;
			//printf("success release blockSumBuffer\n");
		}
    }

	//HOST_FREE(SP);
}
#define REDUCE_MEM_COPY

void scanImpl(cl_mem d_Src, int rLen, cl_mem d_Dst,int *index,cl_event *eventList,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,ScanPara* SP,int _CPU_GPU)
{   //*Flag_CPU_GPU = 0;
//const double scan_time_0 = getCurrentTimestamp();	
    /* Do block-wise sum */
time_slot[0] = getCurrentTimestamp();

#ifdef REDUCE_MEM_COPY
    bScan_int(SP->gLength, &d_Src, &d_Dst, &SP->blockSumBuffer[0],index,eventList,kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
#else
    bScan_int(SP->gLength, &d_Src, &SP->outputBuffer[0], &SP->blockSumBuffer[0],index,eventList,kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
#endif
	//const double scan_time_1 = getCurrentTimestamp();	

    for(int i = 1; i < (int)SP->pass; i++)
    {
        bScan_int((cl_uint)(SP->gLength / pow((float)SP->blockSize, (float)i)), 
            &SP->blockSumBuffer[i - 1],
            &SP->outputBuffer[i],
            &SP->blockSumBuffer[i],index,eventList,kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
    }
    int tempLength = (int)(SP->gLength / pow((float)SP->blockSize, (float)SP->pass));

time_slot[1] = getCurrentTimestamp();

    /* Do scan to SP->tempBuffer */
    pScan_int(tempLength, &SP->blockSumBuffer[SP->pass - 1], &SP->tempBuffer,index,eventList,kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
//const double scan_time_2 = getCurrentTimestamp();	
time_slot[2] = getCurrentTimestamp();

    /* Do block-addition on SP->outputBuffers */

	bAddition_int((cl_uint)(SP->gLength / pow((float)SP->blockSize, (float)(SP->pass - 1))),
        &SP->tempBuffer, &SP->outputBuffer[SP->pass - 1],index,eventList,kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
#ifdef REDUCE_MEM_COPY
	for(int i = SP->pass - 1; i > 1; i--)
    {
        bAddition_int((cl_uint)(SP->gLength / pow((float)SP->blockSize, (float)(i - 1))),
            &SP->outputBuffer[i], &SP->outputBuffer[i - 1],index,eventList,kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
    }
        bAddition_int((cl_uint)(SP->gLength ), //  / pow((float)SP->blockSize, (float)(1 - 1))
            &SP->outputBuffer[1], &d_Dst,index,eventList,kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
		//printf("rLen = 0x%x, SP->gLength = 0x%x\n", rLen, SP->gLength);
#else
	for(int i = SP->pass - 1; i > 0; i--)
    {
        bAddition_int((cl_uint)(SP->gLength / pow((float)SP->blockSize, (float)(i - 1))),
            &SP->outputBuffer[i], &SP->outputBuffer[i - 1],index,eventList,kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
    }

	cl_copyBuffer(d_Dst, SP->outputBuffer[0], rLen*sizeof(int),index,eventList,Flag_CPU_GPU,burden,_CPU_GPU);

#endif
time_slot[3] = getCurrentTimestamp();

//const double scan_time_4 = getCurrentTimestamp();

//  printf("\n SP->gLength = %d", SP->gLength );  printf("\n SP->blockSize = %d", SP->blockSize);  
//  printf("\nscan_large_array (0x%x),Time: %0.3f ms", rLen,    (scan_time_1 - scan_time_0) * 1e3);
//  printf("\nscan_offset_array (0x%x),Time: %0.3f ms", rLen, (scan_time_2 - scan_time_1) * 1e3);
//  printf("\nadd_large_array (0x%x),Time: %0.3f ms", rLen, (scan_time_3 - scan_time_2) * 1e3);
//  printf("\ncopy_to_dest_block (0x%x),Time: %0.3f ms", rLen, (scan_time_4 - scan_time_3) * 1e3);
  
}

void testScanImpl(int rLen)
{
	int _CPU_GPU=0;
	cl_event eventList[2];
	int index=0;
	cl_kernel Kernel; 
	int CPU_GPU;
	double burden;	
	int result=0;
	int memSize=sizeof(int)*rLen;
	int outSize=sizeof(int)*rLen;
	void *Rin;
	HOST_MALLOC(Rin, memSize);
	generateRandInt((int*)Rin, rLen,rLen,0);
	void *Rout;
	HOST_MALLOC(Rout, outSize);
	cl_mem d_Rin;
	CL_MALLOC(&d_Rin, memSize);
	cl_mem d_Rout;
	CL_MALLOC(&d_Rout, outSize);
	cl_writebuffer(d_Rin, Rin, memSize,_CPU_GPU);//cl_writebuffer(d_Rin, Rin, memSize,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);
	ScanPara *SP;
	SP=(ScanPara*)malloc(sizeof(ScanPara));
	initScan(rLen,SP);

//  const double start_time = getCurrentTimestamp();
	scanImpl(d_Rin,rLen,d_Rout,&index,eventList,&Kernel,&CPU_GPU,&burden,SP,_CPU_GPU);	
 // const double end_time = getCurrentTimestamp();
   // Wall-clock time taken.
//  printf("\nprescan (%d),Time: %0.3f ms\n", rLen, (end_time - start_time) * 1e3);
    printf("prescan (%d),part: %0.3f, %0.3f, %0.3f, sum: %0.3f ms\n", rLen, (time_slot[1]-time_slot[0])*1e3, 
		            (time_slot[2]-time_slot[1])*1e3, (time_slot[3]-time_slot[2])*1e3,
					(time_slot[3]-time_slot[0])*1e3);

  

	cl_readbuffer(Rout, d_Rout, outSize,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(index-1)%2]);
	closeScan(SP);
	//deschedule(CPU_GPU,burden);
	validateScan( (int*)Rin, rLen, (int*)Rout );
	HOST_FREE(Rin);
	HOST_FREE(Rout);
	CL_FREE(d_Rin);
	CL_FREE(d_Rout);
	clReleaseKernel(Kernel);  
	clReleaseEvent(eventList[0]);
	clReleaseEvent(eventList[1]);
}