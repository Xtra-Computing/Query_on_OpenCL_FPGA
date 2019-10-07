#include "common.h"
#include "testFilter.h"
//#include "KernelScheduler.h"
#include "testScan.h"
//#include "Helper.h"
//#include "PrimitiveCommon.h"
//#include "KernelScheduler.h"
//#include "OpenCL_DLL.h"
double time_filter[6];

using namespace aocl_utils;

 extern int CL_RangeSelection(Record* h_Rin, int rLen, int rangeSmallKey, int rangeLargeKey, Record** h_Rout, 
															  int numThreadPB, int numBlock,int _CPU_GPU );
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

void filterImpl_map_int(cl_mem d_Rin, int beginPos, int rLen, 
					cl_mem d_mark, int smallKey, int largeKey, cl_mem d_temp,
					int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	size_t numThreadsPerBlock_x=numThreadPB;
	size_t globalWorkingSetSize=numThreadPB*numBlock;

	cl_getKernel("filterImpl_map_kernel",Kernel);

    // Set the Argument values
    cl_int ciErr1 = clSetKernelArg((*Kernel), 0, sizeof(cl_mem), (void*)&d_Rin);	
	ciErr1 |= clSetKernelArg((*Kernel), 1, sizeof(cl_int), (void*)&beginPos);
	ciErr1 |= clSetKernelArg((*Kernel), 2, sizeof(cl_int), (void*)&rLen);
	ciErr1 |= clSetKernelArg((*Kernel), 3, sizeof(cl_mem), (void*)&d_mark);	
	ciErr1 |= clSetKernelArg((*Kernel), 4, sizeof(cl_int), (void*)&smallKey);
	ciErr1 |= clSetKernelArg((*Kernel), 5, sizeof(cl_int), (void*)&largeKey);
    ciErr1 |= clSetKernelArg((*Kernel), 6, sizeof(cl_mem), (void*)&d_temp); 

    checkError(ciErr1, "Error in filterImpl_map_kernel clSetKernelArg, Line");

	//printf("clSetKernelArg 0 - 6...\n\n"); 
	kernel_enqueue(rLen,20, 1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);

}
void filterImpl_outSize_int(cl_mem d_outSize,cl_mem d_mark,cl_mem d_markOutput,int rLen,int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU )
{
	size_t numThreadsPerBlock_x=numThreadPB;
	size_t globalWorkingSetSize=numThreadPB*numBlock;
	cl_getKernel("filterImpl_outSize_kernel",Kernel);
    // Set the Argument values
    cl_int ciErr1 = clSetKernelArg((*Kernel), 0, sizeof(cl_mem), (void*)&d_outSize);	
    checkError(ciErr1, "Error in filterImpl_outSize_kernel clSetKernelArg 0, Line");

	ciErr1 = clSetKernelArg((*Kernel), 1, sizeof(cl_mem), (void*)&d_mark);	
    checkError(ciErr1, "Error in filterImpl_outSize_kernel clSetKernelArg 1, Line");

	ciErr1 = clSetKernelArg((*Kernel), 2, sizeof(cl_mem), (void*)&d_markOutput);
    checkError(ciErr1, "Error in filterImpl_outSize_kernel clSetKernelArg 2, Line");

	ciErr1 = clSetKernelArg((*Kernel), 3, sizeof(cl_int), (void*)&rLen);
    checkError(ciErr1, "Error in filterImpl_outSize_kernel clSetKernelArg 3, Line");

	kernel_enqueue(rLen, 21, 1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);


}
void filterImpl_write_int( cl_mem d_Rout, cl_mem d_Rin, cl_mem d_mark, 
						cl_mem d_markOutput, int beginPos, int rLen,
						int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	size_t numThreadsPerBlock_x=numThreadPB;
	size_t globalWorkingSetSize=numThreadPB*numBlock;
	cl_getKernel("filterImpl_write_kernel",Kernel);
    // Set the Argument values
    cl_int ciErr1 = clSetKernelArg((*Kernel), 0, sizeof(cl_mem), (void*)&d_Rout);	
	ciErr1 = clSetKernelArg((*Kernel), 1, sizeof(cl_mem), (void*)&d_Rin);	
	ciErr1 = clSetKernelArg((*Kernel), 2, sizeof(cl_mem), (void*)&d_mark);
	ciErr1 = clSetKernelArg((*Kernel), 3, sizeof(cl_mem), (void*)&d_markOutput);
	ciErr1 |= clSetKernelArg((*Kernel), 4, sizeof(cl_int), (void*)&beginPos);
	ciErr1 |= clSetKernelArg((*Kernel), 5, sizeof(cl_int), (void*)&rLen);
    //printf("clSetKernelArg 0 - 5...\n\n"); 
	    checkError(ciErr1, "Error in filterImpl_write_kernel clSetKernelArg, Line");

	kernel_enqueue(rLen, 23,  1, &globalWorkingSetSize, &numThreadsPerBlock_x,eventList,index,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
}

void filterImpl( cl_mem d_Rin, int beginPos, int rLen, cl_mem* d_Rout, int* outSize, 
				int numThread, int numBlock, int smallKey, int largeKey,int *index,cl_event *eventList,cl_kernel *Kernel, int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_mem d_mark;
	CL_MALLOC( &d_mark, sizeof(int)*rLen ) ;

	cl_mem d_markOutput;
	CL_MALLOC( &d_markOutput, sizeof(int)*rLen ) ;

	cl_mem d_temp;
	CL_MALLOC( &d_temp, sizeof(int)*rLen ) ;

	ScanPara *SP;
	SP=(ScanPara*)malloc(sizeof(ScanPara));
	initScan(rLen,SP);
	//printf("rLen = 0x%x\n", rLen);

 // const double start_time = getCurrentTimestamp();
time_filter[0] =  getCurrentTimestamp();
	  filterImpl_map_int( d_Rin, beginPos, rLen, d_mark, smallKey, largeKey, d_temp, numThread, numBlock,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	 // clWaitForEvents(1,&eventList[(*index-1)%2]);
	  //prefex sum

	//    int  ciErr1=clFlush(CommandQueue[_CPU_GPU]); 
time_filter[1] =  getCurrentTimestamp();

 // const double end_time_1 = getCurrentTimestamp();

	   scanImpl( d_mark, rLen, d_markOutput,index,eventList,Kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
	   clWaitForEvents(1,&eventList[0]); 
	   closeScan(SP);
time_filter[2] =  getCurrentTimestamp();

 // const double end_time_2 = getCurrentTimestamp();
#if 1
	  //get the outSize
	  cl_mem d_outSize;
	  CL_MALLOC(&d_outSize, sizeof(int)) ;
	  filterImpl_outSize_int(d_outSize, d_mark, d_markOutput, rLen,1,1,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	  clWaitForEvents(1,&eventList[0]); 
	  //bufferchecking(d_outSize,sizeof(int));
	  cl_readbuffer(outSize,d_outSize, sizeof(int),index,eventList,Flag_CPU_GPU,burden,_CPU_GPU);
 // const double end_time_3 = getCurrentTimestamp();
#endif
	//write the reduced result
	  CL_MALLOC( d_Rout, sizeof(Record) * rLen);  //*outSize
	  filterImpl_write_int( *d_Rout, d_Rin, d_mark, d_markOutput, beginPos, rLen,numThread, numBlock,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU );
	  clWaitForEvents(1,&eventList[0]); 
 // const double end_time = getCurrentTimestamp();
time_filter[3] =  getCurrentTimestamp();

  
 // Wall-clock time taken.
 // printf("\nmap (0x%x),Time: %0.3f ms\n", rLen,    (end_time_1 - start_time) * 1e3);
 // printf("\nscan (0x%x),Time: %0.3f ms\n", rLen, (end_time_2 - end_time_1) * 1e3);
 // printf("\nsize (0x%x),Time: %0.3f ms\n", rLen, (end_time_3 - end_time_2) * 1e3);
 // printf("\nscatter (0x%x),Time: %0.3f ms\n", rLen, (end_time - end_time_3) * 1e3);


	CL_FREE(d_mark);
	CL_FREE(d_markOutput);
	CL_FREE( d_temp );
}

/*
void filterImpl( cl_mem d_Rin, int beginPos, int rLen, cl_mem* d_Rout, int* outSize, 
				int numThread, int numBlock, int smallKey, int largeKey,int *index,cl_event *eventList,cl_kernel *Kernel, int *Flag_CPU_GPU,double * burden,int _CPU_GPU)
{
	cl_mem d_mark;
	CL_MALLOC( &d_mark, sizeof(int)*rLen ) ;

	cl_mem d_markOutput;
	CL_MALLOC( &d_markOutput, sizeof(int)*rLen ) ;

	cl_mem d_temp;
	CL_MALLOC( &d_temp, sizeof(int)*rLen ) ;

	filterImpl_map_int( d_Rin, beginPos, rLen, d_mark, smallKey, largeKey, d_temp, numThread, numBlock,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(*index-1)%2]);
	//prefex sum
	ScanPara *SP;
	SP=(ScanPara*)malloc(sizeof(ScanPara));
	initScan(rLen,SP);
	scanImpl( d_mark, rLen, d_markOutput,index,eventList,Kernel,Flag_CPU_GPU,burden,SP,_CPU_GPU);
	clWaitForEvents(1,&eventList[(*index-1)%2]); 
	closeScan(SP);

	//get the outSize
	cl_mem d_outSize;
	CL_MALLOC(&d_outSize, sizeof(int)) ;
	filterImpl_outSize_int(d_outSize, d_mark, d_markOutput, rLen,1,1,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU);
	clWaitForEvents(1,&eventList[(*index-1)%2]); 
	//bufferchecking(d_outSize,sizeof(int));
	cl_readbuffer(outSize,d_outSize, sizeof(int),index,eventList,Flag_CPU_GPU,burden,_CPU_GPU);

	//write the reduced result
	CL_MALLOC( d_Rout, sizeof(Record)*(*outSize) );
	filterImpl_write_int( *d_Rout, d_Rin, d_mark, d_markOutput, beginPos, rLen,numThread, numBlock,index,eventList,Kernel,Flag_CPU_GPU,burden,_CPU_GPU );
	clWaitForEvents(1,&eventList[(*index-1)%2]); 
	CL_FREE(d_mark);
	CL_FREE(d_markOutput);
	CL_FREE( d_temp );
}
*/

void testFilterImpl( int rLen, int numThreadPB, int numBlock)//->corresponding to selection
{
	 int _CPU_GPU=0;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_event eventList[2];
	int index=0;
	cl_kernel Kernel; 
	int CPU_GPU = 0;
	double burden;

	int beginPos = 0;
	int memSize = sizeof(Record)*rLen;
	
	void *Rin;
	HOST_MALLOC(Rin, memSize);
	generateRand( (Record *)Rin, 100, rLen, 0 );
	Record* Rout;

	int smallKey = rand()%100;
	int largeKey = ((smallKey + 6) > 100)? 100:(smallKey + 6); //smallKey; //
	printf("smallkey = %d, largekey = %d\n", smallKey, largeKey);


	int* outSize = (int*)malloc( sizeof(int) );

 	*outSize = CL_RangeSelection((Record*) Rin,  rLen, smallKey, largeKey, &Rout, numThreadPB,numBlock , _CPU_GPU);
    printf("filter (%d),part: %0.3f, %0.3f, %0.3f, sum: %0.3f ms\n", rLen, (time_filter[1]-time_filter[0])*1e3, 
		                                                                   (time_filter[2]-time_filter[1])*1e3, 
																		   (time_filter[3]-time_filter[2])*1e3,
					                                                       (time_filter[3]-time_filter[0])*1e3);

	 validateFilter( (Record*) Rin, 0, rLen, Rout,  *outSize, smallKey, largeKey);

	 // CL_PointSelection((Record*) Rin,  rLen, smallKey, &Rout, numThreadPB,numBlock, _CPU_GPU);
	// printf("CL_PointSelectionFinish\n");
}

