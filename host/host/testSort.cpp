#include "common.h"
//#include "PrimitiveCommon.h"
#include "testSort.h"
//#include "Helper.h"
//#include "KernelScheduler.h"
//#include "OpenCL_DLL.h"
// OpenCL Vars---------0 for CPU, 1 for GPU
using namespace aocl_utils;
extern void CL_RadixSort(Record* h_Rin, int rLen,Record* h_Rout,int numThread, int numBlock, int _CPU_GPU);
extern void validateSort(Record * h_Rin, int rLen);


extern cl_context Context;        // OpenCL context
extern cl_program Program;           // OpenCL program
extern cl_command_queue CommandQueue[2];// OpenCL command que
extern cl_platform_id Platform[2];      // OpenCL platform
extern cl_device_id Device[2];          // OpenCL device
extern cl_ulong totalLocalMemory[2];      /**< Max local memory allowed */
extern cl_device_id allDevices[10];

double time_sort[3]; 

#if 1
  void radixSortImpl_sm(cl_mem d_R, int rLen, int meaningless, int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_int  sortAscending = 1; //1: ascending order, 0: descending order
	cl_uint temp;
	cl_int err;

    int dir = 1; //(dir != 0);
	int arrayLength = rLen;
    int batchSize   = rLen/arrayLength;

    cl_int  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    cl_int threadCount = SHARED_SIZE_LIMIT / 2;

	size_t global_work_size[1] = { rLen/2};      
	size_t local_work_size[1] = {threadCount};          

     int counter_debug = 0;
	 int size_small = 2;
	 int size_large = SHARED_SIZE_LIMIT;
	 int dir_shared = 0;
	 int sole       = 0;

//time_sort[0] = getCurrentTimestamp();
/*    if (arrayLength <= SHARED_SIZE_LIMIT)
    {
        assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
        bitonicSortShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
    }
    else
*/ 
    //    bitonicSortShared1<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal);
	cl_getKernel("bitonicSortShared",kernel);
	err  = clSetKernelArg((*kernel), 0, sizeof(cl_mem), (void *) &d_R);
	err |= clSetKernelArg((*kernel), 1, sizeof(cl_uint), (void *) &rLen);
	err |= clSetKernelArg((*kernel), 2, sizeof(cl_uint), (void *) &size_small);
	err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &size_large);
	err |= clSetKernelArg((*kernel), 4, sizeof(cl_uint), (void *) &dir_shared);
	err |= clSetKernelArg((*kernel), 5, sizeof(cl_uint), (void *) &sole);
	err |= clSetKernelArg((*kernel), 6, sizeof(cl_uint), (void *) &size_large);

	checkError(err, "ERROR: Failed to set input kernel arguments of bitonicSortShared");

	kernel_enqueue(rLen, 31, 1, global_work_size, local_work_size,eventList,index,kernel,Flag_CPU_GPU,burden,_CPU_GPU);

	int log_size;
	int log_stride;
	int log_rLen = -1;
	int n = rLen;
	// uint16_t logValue = -1;
     while (n) {//
         log_rLen++;
         n >>= 1;
     }
	 //printf("log_rLen = %d\n", log_rLen);
     //return logValue;
//time_sort[1] = getCurrentTimestamp();

        for (int size = 2 * SHARED_SIZE_LIMIT, log_size = 11; size <= arrayLength, log_size <= log_rLen; (size <<= 1), log_size++)
            for (unsigned stride = size / 2, log_stride = log_size-1; stride > 0; (stride >>= 3), log_stride-=3) // >= SHARED_SIZE_LIMIT/2
              {
				if (stride >= SHARED_SIZE_LIMIT)
				{
				  global_work_size[0] = arrayLength*batchSize/8;   //2   //  4; 
	              local_work_size[0] =256;      //  512; //  

                   // bitonicMergeGlobal<<<(batchSize * arrayLength) / 512, 256>>>(d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, stride, dir);
	               cl_getKernel("bitonicMergeGlobal",kernel);
	               err  = clSetKernelArg((*kernel), 0, sizeof(cl_mem), (void *) &d_R);
	               err |= clSetKernelArg((*kernel), 1, sizeof(cl_uint), (void *) &rLen);
	               err |= clSetKernelArg((*kernel), 2, sizeof(cl_uint), (void *) &log_size); //size
	               err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &log_stride); //stride
				   err |= clSetKernelArg((*kernel), 4, sizeof(cl_uint), (void *) &dir);
				   checkError(err, "ERROR: Failed to set input kernel arguments of bitonicMergeGlobal");                   
                   kernel_enqueue(rLen, 66, 1, global_work_size, local_work_size, eventList,index,kernel,Flag_CPU_GPU,burden,_CPU_GPU);

				}
                else
                {   counter_debug++;
				  global_work_size[0] = rLen/2;      
	              local_work_size[0] = threadCount;      

				  size_small = SHARED_SIZE_LIMIT;
				  size_large = SHARED_SIZE_LIMIT;
				  sole       = 1;
				// bitonicMergeShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, dir);
	               cl_getKernel("bitonicSortShared",kernel);
	               err  = clSetKernelArg((*kernel), 0, sizeof(cl_mem), (void *) &d_R);
	               err |= clSetKernelArg((*kernel), 1, sizeof(cl_uint), (void *) &rLen);
//	               err |= clSetKernelArg((*kernel), 2, sizeof(cl_uint), (void *) &size);
//	               //err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &stride);
//				   err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &dir);
	err |= clSetKernelArg((*kernel), 2, sizeof(cl_uint), (void *) &size_small);
	err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &size_large);
	err |= clSetKernelArg((*kernel), 4, sizeof(cl_uint), (void *) &dir);
	err |= clSetKernelArg((*kernel), 5, sizeof(cl_uint), (void *) &sole);
	err |= clSetKernelArg((*kernel), 6, sizeof(cl_uint), (void *) &size);

				   checkError(err, "ERROR: Failed to set input kernel arguments of bitonicMergeShared");                   
                   kernel_enqueue(rLen, 66, 1, global_work_size, local_work_size, eventList,index,kernel,Flag_CPU_GPU,burden,_CPU_GPU);

					break;
                }
             }
//time_sort[2] = getCurrentTimestamp();
// printf("\nsort(0x%x) Time: part: %0.3f, %0.3f, sum: %0.3fms \n", rLen, (time_sort[1]-time_sort[0])*1e3, 
//	                                                                    (time_sort[2]-time_sort[1])*1e3,
//																		(time_sort[2]-time_sort[0])*1e3 );

	//cl_int numStages = 0; //stages in total
	//for (temp = rLen; temp > 1; temp >>= 1)
	//	++numStages;
	//radixSort_int(d_R,rLen,numThreadPB,numBlock,sortAscending,numStages,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
}


#else
  void radixSortImpl_sm(cl_mem d_R, int rLen, int meaningless, int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_int  sortAscending = 1; //1: ascending order, 0: descending order
	cl_uint temp;
	cl_int err;

    int dir = 1; //(dir != 0);
	int arrayLength = rLen;
    int batchSize   = rLen/arrayLength;

    cl_int  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    cl_int threadCount = SHARED_SIZE_LIMIT / 2;

	size_t global_work_size[1] = { rLen/2};      
	size_t local_work_size[1] = {threadCount};          

     int counter_debug = 0;
	 int size_small = 2;
	 int size_large = SHARED_SIZE_LIMIT;
	 int dir_shared = 0;
	 int sole       = 0;
time_sort[0] = getCurrentTimestamp();
/*    if (arrayLength <= SHARED_SIZE_LIMIT)
    {
        assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
        bitonicSortShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
    }
    else
*/ 
    //    bitonicSortShared1<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal);
	cl_getKernel("bitonicSortShared",kernel);
	err  = clSetKernelArg((*kernel), 0, sizeof(cl_mem), (void *) &d_R);
	err |= clSetKernelArg((*kernel), 1, sizeof(cl_uint), (void *) &rLen);
	err |= clSetKernelArg((*kernel), 2, sizeof(cl_uint), (void *) &size_small);
	err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &size_large);
	err |= clSetKernelArg((*kernel), 4, sizeof(cl_uint), (void *) &dir_shared);
	err |= clSetKernelArg((*kernel), 5, sizeof(cl_uint), (void *) &sole);
	err |= clSetKernelArg((*kernel), 6, sizeof(cl_uint), (void *) &size_large);

	checkError(err, "ERROR: Failed to set input kernel arguments of bitonicSortShared");

	kernel_enqueue(rLen, 31, 1, global_work_size, local_work_size,eventList,index,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
time_sort[1] = getCurrentTimestamp();

        for (int size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1) // >= SHARED_SIZE_LIMIT/2
              {
				if (stride >= SHARED_SIZE_LIMIT)
				{
				  global_work_size[0] = arrayLength*batchSize/2;      //  4; //
	              local_work_size[0] =256;      //  512; //  

                   // bitonicMergeGlobal<<<(batchSize * arrayLength) / 512, 256>>>(d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, stride, dir);
	               cl_getKernel("bitonicMergeGlobal",kernel);
	               err  = clSetKernelArg((*kernel), 0, sizeof(cl_mem), (void *) &d_R);
	               err |= clSetKernelArg((*kernel), 1, sizeof(cl_uint), (void *) &rLen);
	               err |= clSetKernelArg((*kernel), 2, sizeof(cl_uint), (void *) &size);
	               err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &stride);
				   err |= clSetKernelArg((*kernel), 4, sizeof(cl_uint), (void *) &dir);
				   checkError(err, "ERROR: Failed to set input kernel arguments of bitonicMergeGlobal");                   
                   kernel_enqueue(rLen, 66, 1, global_work_size, local_work_size, eventList,index,kernel,Flag_CPU_GPU,burden,_CPU_GPU);

				}
                else
                {   counter_debug++;
				  global_work_size[0] = rLen/2;      
	              local_work_size[0] = threadCount;      

				  size_small = SHARED_SIZE_LIMIT;
				  size_large = SHARED_SIZE_LIMIT;
				  sole       = 1;
				// bitonicMergeShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, dir);
	               cl_getKernel("bitonicSortShared",kernel);
	               err  = clSetKernelArg((*kernel), 0, sizeof(cl_mem), (void *) &d_R);
	               err |= clSetKernelArg((*kernel), 1, sizeof(cl_uint), (void *) &rLen);
//	               err |= clSetKernelArg((*kernel), 2, sizeof(cl_uint), (void *) &size);
//	               //err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &stride);
//				   err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &dir);
	err |= clSetKernelArg((*kernel), 2, sizeof(cl_uint), (void *) &size_small);
	err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &size_large);
	err |= clSetKernelArg((*kernel), 4, sizeof(cl_uint), (void *) &dir);
	err |= clSetKernelArg((*kernel), 5, sizeof(cl_uint), (void *) &sole);
	err |= clSetKernelArg((*kernel), 6, sizeof(cl_uint), (void *) &size);

				   checkError(err, "ERROR: Failed to set input kernel arguments of bitonicMergeShared");                   
                   kernel_enqueue(rLen, 66, 1, global_work_size, local_work_size, eventList,index,kernel,Flag_CPU_GPU,burden,_CPU_GPU);

					break;
                }
             }
time_sort[2] = getCurrentTimestamp();
 printf("\nsort(0x%x) Time: part: %0.3f, %0.3f, sum: %0.3fms \n", rLen, (time_sort[1]-time_sort[0])*1e3, 
	                                                                    (time_sort[2]-time_sort[1])*1e3,
																		(time_sort[2]-time_sort[0])*1e3 );

	//cl_int numStages = 0; //stages in total
	//for (temp = rLen; temp > 1; temp >>= 1)
	//	++numStages;
	//radixSort_int(d_R,rLen,numThreadPB,numBlock,sortAscending,numStages,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
}
#endif

void radixSort_int(cl_mem d_R, int rLen, int numThreadPB, int numBlock, int sortAscending, int numStages,int *index,cl_event *eventList,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_int err;

	cl_int stage;
	cl_int passOfStage;

	size_t global_work_size[1] = { rLen / 2};         //number of global work-items, specific to this program
	size_t local_work_size[1] = {numThreadPB};             //work-group size

	cl_getKernel("BitonicSort_kernel",kernel);

	err  = clSetKernelArg((*kernel), 0, sizeof(cl_mem), (void *) &d_R);
	err |= clSetKernelArg((*kernel), 3, sizeof(cl_uint), (void *) &rLen);
	err |= clSetKernelArg((*kernel), 4, sizeof(cl_uint), (void *) &sortAscending);

    checkError(err, "ERROR: Failed to set input kernel arguments");


	for (stage = 0; stage < numStages; ++stage)
	{
		err = clSetKernelArg((*kernel),1,sizeof(cl_uint),(void*)&stage);
		for (passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
		{
			err = clSetKernelArg((*kernel),2,sizeof(cl_uint),(void*)&passOfStage);
			kernel_enqueue(rLen, 31, 1, global_work_size, local_work_size,eventList,index,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
			clWaitForEvents(1,&eventList[*index]);
		}
		clWaitForEvents(1,&eventList[*index]);
	}
}

void radixSortImpl(cl_mem d_R, int rLen, int meaningless, int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_int  sortAscending = 1; //1: ascending order, 0: descending order
	cl_uint temp;

	cl_int numStages = 0; //stages in total
	for (temp = rLen; temp > 1; temp >>= 1)
		++numStages;

	radixSort_int(d_R,rLen,numThreadPB,numBlock,sortAscending,numStages,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
}


void testSortImpl(int rLen, int numThreadPB, int numBlock)
{
	int _CPU_GPU=0;
	int memSize = sizeof(Record) * rLen;
	void* h_Rin;
	HOST_MALLOC(h_Rin,memSize);
	generateRand((Record*) h_Rin, 200, rLen, 0);
	void * h_Rout;
	HOST_MALLOC(h_Rout,memSize);

	CL_RadixSort((Record*) h_Rin,  rLen,(Record*) h_Rout, numThreadPB,  numBlock,_CPU_GPU);

//	validateSort((Record*) h_Rout, rLen);

	HOST_FREE(h_Rin);
	HOST_FREE(h_Rout);
	printf("Sort test finished\n");
}


