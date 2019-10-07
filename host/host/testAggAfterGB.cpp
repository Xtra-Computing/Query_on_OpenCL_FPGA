#include "common.h"
#include "testGroupBy.h"
//#include "PrimitiveCommon.h"
#include "testScatter.h"
//#include "Helper.h"
//#include "OpenCL_DLL.h"
//#include "KernelScheduler.h"
double time_agg[6]; 

extern void CL_agg_max_afterGroupBy(Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, int numThread,int _CPU_GPU );
extern void CL_agg_min_afterGroupBy(Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, int numThread,int _CPU_GPU );
extern void CL_agg_sum_afterGroupBy(Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, int numThread,int _CPU_GPU );
extern void CL_agg_avg_afterGroupBy(Record* h_Rin, int rLen, int* h_startPos, int numGroups, Record* h_Ragg, int* h_aggResults, int numThread,int _CPU_GPU );
extern  int CL_GroupBy(Record * h_Rin, int rLen, Record* h_Rout, int** h_startPos, int numThread, int numBlock , int _CPU_GPU); 
extern void validateGroupBy( Record* h_Rin, int rLen, Record* h_Rout, int* h_startPos, int numGroup );
extern void validateAggAfterGroupBy( Record * h_Rin, int rLen, int* startPos, int numGroups, Record * Ragg, int* aggResults, int OPERATOR );
extern void validateAggAfterGroupBy_easy( Record * h_Rin, int rLen, int* startPos, int numGroups, int* aggResults, int OPERATOR );

extern int range_selection(cl_mem d_Rin, int rLen, int rangeSmallKey, int rangeLargeKey, cl_mem* d_Rout,
					int numThreadPB , int numBlock , int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU);

using namespace aocl_utils;

// OpenCL Vars---------0 for CPU, 1 for GPU
extern cl_context Context;        // OpenCL context
extern cl_program Program;           // OpenCL program
extern cl_command_queue CommandQueue[2];// OpenCL command que
extern cl_platform_id Platform[2];      // OpenCL platform
extern cl_device_id Device[2];          // OpenCL device
extern cl_ulong totalLocalMemory[2];      /**< Max local memory allowed */
extern cl_device_id allDevices[10];

extern int kernel_index;

#ifndef REDUCE_AFTER_GROUPBY
#define REDUCE_AFTER_GROUPBY
#define MAX_NUM_BLOCK (512)
void mapBeforeGather_int(cl_mem d_Rin, int rLen,cl_mem d_loc,cl_mem d_temp,int numBlock, int numThread,int *index,cl_event *eventList,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	size_t numThreadsPerBlock_x=numThread;
	size_t globalWorkingSetSize=numThread*numBlock;
	cl_getKernel("mapBeforeGather_kernel",kernel);

    cl_int ciErr1 = clSetKernelArg((*kernel), 0, sizeof(cl_mem), (void*)&d_Rin);	
	ciErr1 |= clSetKernelArg((*kernel), 1, sizeof(cl_int), (void*)&rLen);
    ciErr1 |= clSetKernelArg((*kernel), 2, sizeof(cl_mem), (void*)&d_loc);
	ciErr1 |= clSetKernelArg((*kernel), 3, sizeof(cl_mem), (void*)&d_temp);
    //printf("clSetKernelArg 0 - 3...\n\n"); 

	    checkError(ciErr1, "ERROR: in arg_set of mapBeforeGather_int, Line in file !!!\n\n");

	kernel_enqueue(rLen,3, 
		1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,kernel,Flag_CPU_GPU,burden,_CPU_GPU);	
}
void gatherBeforeAgg(cl_mem d_Rin, int rLen, cl_mem d_Ragg, cl_mem d_S, int numThread, int numBlock ,int *index,cl_event *eventList,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_mem d_loc;
	CL_MALLOC(&d_loc, sizeof(int)*rLen ) ;
	clWaitForEvents(1,&eventList[(*index-1)%2]);
	cl_mem d_temp;
	CL_MALLOC(&d_temp, sizeof(int)*rLen ) ;

	mapBeforeGather_int( d_Rin, rLen, d_loc, d_temp,numBlock, numThread,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	gatherImpl( d_Ragg, rLen, d_loc, d_S,rLen, numThread, numBlock,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	CL_FREE(d_temp);
	clWaitForEvents(1,&eventList[(*index-1)%2]);
	CL_FREE(d_loc);
}
void parallelAggregate_init(cl_mem d_S,cl_mem d_startPos,cl_mem d_aggResults,
							int OPERATOR,int blockOffset, int numGroups, int numBlock, int numThread,int sharedMemSize, int rLen,int *index,cl_event *eventList,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{ 
	size_t numThreadsPerBlock_x=numThread;
	size_t globalWorkingSetSize=numThread*numBlock;
	cl_getKernel("parallelAggregate_kernel", kernel, _CPU_GPU);
    // Set the Argument values
    cl_int ciErr1 = clSetKernelArg((*kernel), 0, sizeof(cl_mem), (void*)&d_S);	
	ciErr1 |= clSetKernelArg((*kernel), 1, sizeof(cl_mem), (void*)&d_startPos);
    ciErr1 |= clSetKernelArg((*kernel), 2, sizeof(cl_mem), (void*)&d_aggResults);
	ciErr1 |= clSetKernelArg((*kernel), 3, sizeof(cl_int), (void*)&OPERATOR);
	ciErr1 |= clSetKernelArg((*kernel), 4, sizeof(cl_int), (void*)&blockOffset);
	ciErr1 |= clSetKernelArg((*kernel), 5, sizeof(cl_int), (void*)&numGroups);
	ciErr1 |= clSetKernelArg((*kernel), 6, sizeof(cl_int), (void*)&rLen);
	ciErr1 |= clSetKernelArg((*kernel), 7, sharedMemSize * sizeof(cl_int), NULL);
    //printf("clSetKernelArg 0 - 6...\n\n"); 

    checkError(ciErr1, "ERROR: in arg_set of parallelAggregate_init, Line in file !!!\n\n");

	kernel_enqueue(rLen,4, 
		1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,kernel,Flag_CPU_GPU,burden,_CPU_GPU);

}
void aggAfterGroupByImpl(cl_mem d_Rin, int rLen, cl_mem d_startPos, int numGroups, cl_mem d_Ragg, cl_mem d_aggResults, int OPERATOR,  int numThread,int *index,cl_event *eventList,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	//gather=============================================================
	cl_mem d_S;
	CL_MALLOC( &d_S, sizeof(Record)*rLen ) ;
	gatherBeforeAgg( d_Rin, rLen, d_Ragg, d_S, 512, 256,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[0]);
	//parallel aggregation after gather======================================
	//numThread = 1;
	int numChunk = ceil(((float)numGroups)/MAX_NUM_BLOCK);
	int numBlock;
	int blockOffset;

	int sharedMemSize = sizeof(int)*numThread;

	for( int chunkIdx = 0; chunkIdx < numChunk; chunkIdx++ )
	{
		blockOffset = chunkIdx*MAX_NUM_BLOCK;
		if( chunkIdx == ( numChunk - 1 ) )
		{
			numBlock = numGroups - chunkIdx*MAX_NUM_BLOCK;
		}
		else
		{
			numBlock = MAX_NUM_BLOCK;
		}
		parallelAggregate_init(d_S, d_startPos, d_aggResults, OPERATOR, blockOffset, numGroups,numBlock, numThread, sharedMemSize, rLen,index,eventList,kernel,Flag_CPU_GPU,burden,_CPU_GPU);
		clWaitForEvents(1,&eventList[(*index-1)%2]);
	}
	clWaitForEvents(1,&eventList[0]); 
	CL_FREE( d_S );
}
#endif

#if 1
void testAggAfterGroupByImpl( int rLen, int OPERATOR, int numThread, int numBlock)
{
	int CPU_GPU;
	int _CPU_GPU=0;
	int memSize = sizeof(Record)*rLen;
	void* h_Rin;
	void* h_Rout;
	void* h_Sin;
	void* h_startPos;
	HOST_MALLOC( h_Rin, memSize );
	HOST_MALLOC( h_Rout, memSize );
	HOST_MALLOC( h_Sin, memSize );
	generateRand((Record *) h_Rin, 50, rLen, 0 );
	generateRand((Record *) h_Sin, TEST_MAX, rLen, 0 );  

    void *h_aggResults;
    HOST_MALLOC( h_aggResults, memSize );

 //const double start_time = getCurrentTimestamp();
   	cl_mem d_Rin;
	cl_mem d_Rout;
	cl_mem d_startPos;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_event eventList[2];
	int index=0;
	cl_kernel Kernel; 

	double burden;

	CL_MALLOC( &d_Rin, memSize );
	CL_MALLOC(&d_Rout, memSize );
	cl_writebuffer( d_Rout, h_Rin, memSize,&index,eventList,&CPU_GPU,&burden,_CPU_GPU); //d_Rin
	int numGroups = 0;
	//d_Rin
	numGroups= groupByImpl(d_Rout, rLen, d_Rout, &d_startPos, numThread, numBlock,&index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);
	//int numGroup = 0;
	//numGroup = CL_GroupBy((Record *)h_Rin, rLen,(Record *) h_Rout, &h_startPos, numThread, numBlock,_CPU_GPU);
 // const double end_time = getCurrentTimestamp();

  //	printf("numGroups  = %d\n", numGroups);
   if (kernel_index == 30)
   {
     _CPU_GPU = 1;
   }

	cl_mem  d_aggResults;
    CL_MALLOC(&d_aggResults, sizeof(int)*numGroups );

	int numChunk = ceil(((float)numGroups)/MAX_NUM_BLOCK);

	int blockOffset;
	int sharedMemSize = sizeof(int)*numThread;

	for( int chunkIdx = 0; chunkIdx < numChunk; chunkIdx++ )
	{
		blockOffset = chunkIdx*MAX_NUM_BLOCK;
		if( chunkIdx == ( numChunk - 1 ) )
		{
			numBlock = numGroups - chunkIdx*MAX_NUM_BLOCK;
		}
		else
		{
			numBlock = MAX_NUM_BLOCK;
		}
		parallelAggregate_init(d_Rout, d_startPos, d_aggResults, OPERATOR, blockOffset, numGroups,numBlock, numThread, sharedMemSize, rLen,&index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);
		//clWaitForEvents(1,&eventList[0]);
	}
	clWaitForEvents(1,&eventList[0]); 
time_agg[3] = getCurrentTimestamp();

/*
	cl_mem  d_Ragg;
	CL_MALLOC(&d_Ragg, sizeof(int)*rLen );
	cl_writebuffer( d_Ragg, h_Ragg, sizeof(Record)*rLen,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );


	aggAfterGroupByImpl(d_Rin, rLen, d_startPos, numGroup, d_Ragg, d_aggResults, REDUCE_AVERAGE, numThread,&index,eventList,kernel,_CPU_GPU,burden,_CPU_GPU);

*/
/*
	validateGroupBy((Record*) h_Rin, rLen, (Record*)h_Rout,h_startPos, numGroup );

  const double start_time_1 = getCurrentTimestamp();
	void* h_aggResults;
	HOST_MALLOC(h_aggResults, sizeof(int)*numGroup );
	switch(OPERATOR){
	case REDUCE_MAX:
		{
			CL_agg_max_afterGroupBy((Record *)h_Rout,rLen,h_startPos,numGroup,(Record *)h_Sin,(int *)h_aggResults,numThread,_CPU_GPU);
			break;
		}
	case REDUCE_MIN:
		{
			CL_agg_min_afterGroupBy((Record *)h_Rout,rLen,h_startPos,numGroup,(Record *)h_Sin,(int *)h_aggResults,numThread,_CPU_GPU);
			break;
		}
	case REDUCE_SUM:
		{
			CL_agg_sum_afterGroupBy((Record *)h_Rout,rLen,h_startPos,numGroup,(Record *)h_Sin,(int *)h_aggResults,numThread,_CPU_GPU);
			break;
		}
	case REDUCE_AVERAGE:
		{
			CL_agg_avg_afterGroupBy((Record *)h_Rout,rLen,h_startPos,numGroup,(Record *)h_Sin,(int *)h_aggResults,numThread,_CPU_GPU);
			break;
		}
	}
*/
 // const double end_time_1 = getCurrentTimestamp();
 //   printf("1 Time: %0.3f ms, 2 Time: %0.3f ms\n",   (end_time - start_time)* 1e3,  (end_time_1 - start_time_1) * 1e3);
  	HOST_MALLOC( h_startPos, sizeof(int)*numGroups );
//  printf("aggregation (0x%x),Time: %0.3f ms\n", rLen,    (end_time - start_time + end_time_1 - start_time_1) * 1e3);
  	cl_readbuffer( h_aggResults, d_aggResults, sizeof(int)*numGroups,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );
  	cl_readbuffer( h_startPos, d_startPos, sizeof(int)*numGroups,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );
  	cl_readbuffer( h_Rout, d_Rout, sizeof(Record)*rLen,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );

	printf("aggregation (0x%x): %0.3f(sort), %0.3f(group), %0.3f(write), %0.3f(agg);sum: %0.3fms.\n", rLen,   
		                                                   (time_agg[4]-time_agg[0]) * 1e3,
														   (time_agg[5]-time_agg[4]) * 1e3,
		                                                   (time_agg[2]-time_agg[1]) * 1e3,
														   (time_agg[3]-time_agg[2]) * 1e3,
														   (time_agg[3]-time_agg[0]) * 1e3);

	//printf("numGroups  = %d\n", numGroups);
	//for (int ii = 0; ii < numGroups; ii++)
	//	printf("h_startPos[%d] = 0x%x,h_aggResults[%d] = 0x%x\n", ii, ((int *)h_startPos)[ii], ii, ((int *)h_aggResults)[ii]);

	validateAggAfterGroupBy_easy((Record*) h_Rout, rLen, (int*)h_startPos, numGroups, (int *)h_aggResults, OPERATOR); //h_Rin
}
#else
void testAggAfterGroupByImpl( int rLen, int OPERATOR, int numThread, int numBlock)
{
	int _CPU_GPU=0;
	int memSize = sizeof(Record)*rLen;
	void* h_Rin;
	void* h_Rout;
	void* h_Sin;
	int* h_startPos;
	HOST_MALLOC( h_Rin, memSize );
	HOST_MALLOC( h_Rout, memSize );
	HOST_MALLOC( h_Sin, memSize );
	generateRand((Record *) h_Rin, 50, rLen, 0 );
	generateRand((Record *) h_Sin, TEST_MAX, rLen, 0 );  

 const double start_time = getCurrentTimestamp();
	int numGroup = 0;
	numGroup = CL_GroupBy((Record *)h_Rin, rLen,(Record *) h_Rout, &h_startPos, numThread, numBlock,_CPU_GPU);
  const double end_time = getCurrentTimestamp();

	validateGroupBy((Record*) h_Rin, rLen, (Record*)h_Rout,h_startPos, numGroup );

  const double start_time_1 = getCurrentTimestamp();
	void* h_aggResults;
	HOST_MALLOC(h_aggResults, sizeof(int)*numGroup );
	switch(OPERATOR){
	case REDUCE_MAX:
		{
			CL_agg_max_afterGroupBy((Record *)h_Rout,rLen,h_startPos,numGroup,(Record *)h_Sin,(int *)h_aggResults,numThread,_CPU_GPU);
			break;
		}
	case REDUCE_MIN:
		{
			CL_agg_min_afterGroupBy((Record *)h_Rout,rLen,h_startPos,numGroup,(Record *)h_Sin,(int *)h_aggResults,numThread,_CPU_GPU);
			break;
		}
	case REDUCE_SUM:
		{
			CL_agg_sum_afterGroupBy((Record *)h_Rout,rLen,h_startPos,numGroup,(Record *)h_Sin,(int *)h_aggResults,numThread,_CPU_GPU);
			break;
		}
	case REDUCE_AVERAGE:
		{
			CL_agg_avg_afterGroupBy((Record *)h_Rout,rLen,h_startPos,numGroup,(Record *)h_Sin,(int *)h_aggResults,numThread,_CPU_GPU);
			break;
		}
	}
  const double end_time_1 = getCurrentTimestamp();
    printf("1 Time: %0.3f ms, 2 Time: %0.3f ms\n",   (end_time - start_time)* 1e3,  (end_time_1 - start_time_1) * 1e3);

  printf("aggregation (0x%x),Time: %0.3f ms\n", rLen,    (end_time - start_time + end_time_1 - start_time_1) * 1e3);

	validateAggAfterGroupBy((Record*) h_Rin, rLen, (int*)h_startPos, numGroup,(Record*) h_Sin, (int *)h_aggResults, OPERATOR);
}
#endif


void testFilterAggAfterGroupByImpl( int rLen, int OPERATOR, int numThread, int numBlock)
{
	int CPU_GPU;
	int _CPU_GPU=0;
	int memSize = sizeof(Record)*rLen;
	void* h_Rin;
	void* h_Rout;
	void* h_Sin;
	void* h_startPos;
	HOST_MALLOC( h_Rin, memSize );
	HOST_MALLOC( h_Rout, memSize );
	HOST_MALLOC( h_Sin, memSize );
	generateRand((Record *) h_Rin, 50, rLen, 0 );
	generateRand((Record *) h_Sin, TEST_MAX, rLen, 0 );  

    void *h_aggResults;
    HOST_MALLOC( h_aggResults, memSize );

 //const double start_time = getCurrentTimestamp();
   	cl_mem d_Rin;
	cl_mem d_Rout;
	cl_mem d_startPos;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	double burden;
	int rangeSmallKey = 0;
	int rangeLargeKey = 50;

	cl_event eventList[2];
	int index=0;
	cl_kernel Kernel; 
	int CPU_GPU;
	double burden;

	CL_MALLOC( &d_Rin, sizeof(Record)*rLen );
	cl_writebuffer( d_Rin, h_Rin, sizeof(Record)*rLen,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);

const double start_time = getCurrentTimestamp();
	int outSize = range_selection( d_Rin, rLen, rangeSmallKey, rangeLargeKey, &d_Rout, 
		numThread, numBlock,&index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);
  const double end_time = getCurrentTimestamp();
  printf("range_selection (0x%x),Time: %0.3f ms\n", rLen,    (end_time - start_time) * 1e3);

	(*h_Rout) = (Record*)malloc( sizeof(Record)*outSize );
	cl_readbuffer(*h_Rout, d_Rout, sizeof(Record)*outSize,&index,eventList,&CPU_GPU,&burden,_CPU_GPU);
	//clWaitForEvents(1,&eventList[(index-1)%2]);
	//deschedule(CPU_GPU,burden);
	CL_FREE(d_Rin);
	CL_FREE(d_Rout);





	CL_MALLOC( &d_Rin, memSize );
	CL_MALLOC(&d_Rout, memSize );
	cl_writebuffer( d_Rout, h_Rin, memSize,&index,eventList,&CPU_GPU,&burden,_CPU_GPU); //d_Rin
	int numGroups = 0;
	//d_Rin
	numGroups= groupByImpl(d_Rout, rLen, d_Rout, &d_startPos, numThread, numBlock,&index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);
	//int numGroup = 0;
	//numGroup = CL_GroupBy((Record *)h_Rin, rLen,(Record *) h_Rout, &h_startPos, numThread, numBlock,_CPU_GPU);
 // const double end_time = getCurrentTimestamp();

  //	printf("numGroups  = %d\n", numGroups);
   if (kernel_index == 30)
   {
     _CPU_GPU = 1;
   }

	cl_mem  d_aggResults;
    CL_MALLOC(&d_aggResults, sizeof(int)*numGroups );

	int numChunk = ceil(((float)numGroups)/MAX_NUM_BLOCK);

	int blockOffset;
	int sharedMemSize = sizeof(int)*numThread;

	for( int chunkIdx = 0; chunkIdx < numChunk; chunkIdx++ )
	{
		blockOffset = chunkIdx*MAX_NUM_BLOCK;
		if( chunkIdx == ( numChunk - 1 ) )
		{
			numBlock = numGroups - chunkIdx*MAX_NUM_BLOCK;
		}
		else
		{
			numBlock = MAX_NUM_BLOCK;
		}
		parallelAggregate_init(d_Rout, d_startPos, d_aggResults, OPERATOR, blockOffset, numGroups,numBlock, numThread, sharedMemSize, rLen,&index,eventList,&Kernel,&CPU_GPU,&burden,_CPU_GPU);
		//clWaitForEvents(1,&eventList[0]);
	}
	clWaitForEvents(1,&eventList[0]); 
time_agg[3] = getCurrentTimestamp();

 // const double end_time_1 = getCurrentTimestamp();
 //   printf("1 Time: %0.3f ms, 2 Time: %0.3f ms\n",   (end_time - start_time)* 1e3,  (end_time_1 - start_time_1) * 1e3);
  	HOST_MALLOC( h_startPos, sizeof(int)*numGroups );
//  printf("aggregation (0x%x),Time: %0.3f ms\n", rLen,    (end_time - start_time + end_time_1 - start_time_1) * 1e3);
  	cl_readbuffer( h_aggResults, d_aggResults, sizeof(int)*numGroups,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );
  	cl_readbuffer( h_startPos, d_startPos, sizeof(int)*numGroups,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );
  	cl_readbuffer( h_Rout, d_Rout, sizeof(Record)*rLen,&index,eventList,&CPU_GPU,&burden,_CPU_GPU );

	printf("aggregation (0x%x): %0.3f(sort), %0.3f(group), %0.3f(write), %0.3f(agg);sum: %0.3fms.\n", rLen,   
		                                                   (time_agg[4]-time_agg[0]) * 1e3,
														   (time_agg[5]-time_agg[4]) * 1e3,
		                                                   (time_agg[2]-time_agg[1]) * 1e3,
														   (time_agg[3]-time_agg[2]) * 1e3,
														   (time_agg[3]-time_agg[0]) * 1e3);

	//printf("numGroups  = %d\n", numGroups);
	//for (int ii = 0; ii < numGroups; ii++)
	//	printf("h_startPos[%d] = 0x%x,h_aggResults[%d] = 0x%x\n", ii, ((int *)h_startPos)[ii], ii, ((int *)h_aggResults)[ii]);

	validateAggAfterGroupBy_easy((Record*) h_Rout, rLen, (int*)h_startPos, numGroups, (int *)h_aggResults, OPERATOR); //h_Rin
}