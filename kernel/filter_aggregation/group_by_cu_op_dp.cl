typedef uint2 Record;
#define L_BLOCK_SIZE 256


#define SHARED_SIZE_LIMIT_LOG 10
#define SHARED_SIZE_LIMIT     (1<<SHARED_SIZE_LIMIT_LOG)//1024


#define REDUCE_SUM (0)
#define REDUCE_MAX (1)
#define REDUCE_MIN (2)
#define REDUCE_AVERAGE (3)

#define TEST_MAX (1<<30)
#define TEST_MIN (0)

#define LOG_NUM_BANKS 4

inline int CONFLICT_FREE_OFFSET_REDUCE( int index )
{
		//return ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS));

		return ((index) >> LOG_NUM_BANKS);
}	
/*
__attribute((num_compute_units(NUM_CU_SORT)))
__kernel //kid 31
	void BitonicSort_kernel(__global Record * theArray,
                          const uint stage, 
						  const uint passOfStage,
						  const uint width,
						  const uint direction)
{
   uint sortIncreasing = direction;
   uint threadId = get_global_id(0);
    
    uint pairDistance = 1 << (stage - passOfStage);
    uint blockWidth   = 2 * pairDistance;

    uint leftId = (threadId % pairDistance) 
                   + (threadId / pairDistance) * blockWidth;

    uint rightId = leftId + pairDistance;
    
    Record leftElement = theArray[leftId];
    Record rightElement = theArray[rightId];
    
    uint sameDirectionBlockWidth = 1 << stage;
    
    if((threadId/sameDirectionBlockWidth) % 2 == 1)
        sortIncreasing = 1 - sortIncreasing;

    Record greater;
    Record lesser;
    if(leftElement.y > rightElement.y)
    {
        greater = leftElement;
        lesser  = rightElement;
    }
    else
    {
        greater = rightElement;
        lesser  = leftElement;
    }
    
    if(sortIncreasing)
    {
        theArray[leftId]  = lesser;
        theArray[rightId] = greater;
    }
    else
    {
        theArray[leftId]  = greater;
        theArray[rightId] = lesser;
    }
}
*/
__kernel void dummy()
{
}
////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
//Bottom-level bitonic sort
//Almost the same as bitonicSortShared with the exception of
//even / odd subarrays being sorted in opposite directions
//Bitonic merge accepts both
//Ascending | descending or descending | ascending sorted pairs
__attribute((num_compute_units(NUM_CU_S)))
__attribute__((reqd_work_group_size(SHARED_SIZE_LIMIT/2,1,1)))
__kernel void bitonicSortShared( __global Record * d_array, uint arrayLength, uint size_small, uint size_large, uint dir, uint sole, uint size_merge)
{ 

    //local memory storage for current subarray
    __local Record s_array[SHARED_SIZE_LIMIT];

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint bid = get_group_id(0);

    d_array += bid * SHARED_SIZE_LIMIT + lid;

    s_array[lid  +                       0]  = d_array[0                          ];
    s_array[lid  + (SHARED_SIZE_LIMIT / 2)]  = d_array[0 + (SHARED_SIZE_LIMIT / 2)];

    for (uint size = size_small; size < (size_large<<1); size <<= 1) //2 SHARED_SIZE_LIMIT  //size=SHARED_SIZE_LIMIT
    {
        //Bitonic merge
        uint ddd;
       if (sole == 0)
	   {
		  if (size == SHARED_SIZE_LIMIT)
            ddd = bid & 1; // blockIdx.x
		  else
		    ddd = ( lid & (size / 2)) != 0; //threadIdx.x
	   }
	   else
	   {
           uint comparatorI = gid & ((arrayLength / 2) - 1);
           ddd = dir ^ ((comparatorI & (size_merge / 2)) != 0);
	      
	   }
		
        for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);   //__syncthreads();
            
			uint pos = 2 * lid - (lid & (stride - 1)); //threadIdx.x 

			Record aa = s_array[pos +      0];
			Record bb = s_array[pos + stride];
			if ( (aa.y > bb.y ) == ddd) //swap when condition satisfied
			{
			  s_array[pos +      0] = bb;
			  s_array[pos + stride] = aa;
			}
        }
    }
      barrier(CLK_LOCAL_MEM_FENCE); //  __syncthreads();

	  d_array[0  +                       0]  = s_array[lid                          ];
	  d_array[0  + (SHARED_SIZE_LIMIT / 2)]  = s_array[lid + (SHARED_SIZE_LIMIT / 2)];

}



//Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
//__attribute__((max_work_group_size(SHARED_SIZE_LIMIT/2,1,1))) 

//each work item handles 8 elements, only Size/8 work items are required to reduce the number of global memory accesses and kernel launching times.
__attribute((num_compute_units(NUM_CU_G)))
__kernel void bitonicMergeGlobal(
    __global Record * d_array,
    uint arrayLength,
    uint log_size, //size
    uint log_stride,
    uint dir
)
{
    uint global_comparatorI = get_global_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);
	uint size               = 1<<log_size;

	//uint stride           = 1<<log_stride;
	uint stride_1           = 1<<(log_stride-2);  
	uint stride_2           = 1<<(log_stride-1);  
	uint stride_4           = 1<<log_stride;  
	uint stride_3           = stride_1 + stride_2;  
	uint stride_5           = stride_1 + stride_4;  
	uint stride_6           = stride_2 + stride_4;  
	uint stride_7           = stride_2 + stride_5;  

    //Bitonic merge
    uint ddd     = dir ^ ((comparatorI & (size / 8)) != 0); //2
                     //        base                                             offset.... (1<<(log_stride-2))
    uint pos     = ( (comparatorI>>(log_stride-2))<<(log_stride+1) ) + ( comparatorI&(stride_1-1) ); //2 * global_comparatorI - (global_comparatorI & (stride - 1));//bug here....

    Record data_0 = d_array[pos +        0];
    Record data_1 = d_array[pos + stride_1];
    Record data_2 = d_array[pos + stride_2];
    Record data_3 = d_array[pos + stride_3];
    Record data_4 = d_array[pos + stride_4];
    Record data_5 = d_array[pos + stride_5];
    Record data_6 = d_array[pos + stride_6];
    Record data_7 = d_array[pos + stride_7];

//first stage
    if ( (data_0.y > data_4.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_0;
	  data_0  =  data_4;
	  data_4  =  tmp;
    }
    if ( (data_1.y > data_5.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_1;
	  data_1  =  data_5;
	  data_5  =  tmp;
    }
    if ( (data_2.y > data_6.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_2;
	  data_2  =  data_6;
	  data_6  =  tmp;
    }
    if ( (data_3.y > data_7.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_3;
	  data_3  =  data_7;
	  data_7  =  tmp;
    }
  
//second stage...
  if ( (log_stride - SHARED_SIZE_LIMIT_LOG) >= 1 )
  {
    if ( (data_0.y > data_2.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_0;
	  data_0  =  data_2;
	  data_2  =  tmp;
    }
    if ( (data_1.y > data_3.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_1;
	  data_1  =  data_3;
	  data_3  =  tmp;
    }
    if ( (data_4.y > data_6.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_4;
	  data_4  =  data_6;
	  data_6  =  tmp;
    }
    if ( (data_5.y > data_7.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_5;
	  data_5  =  data_7;
	  data_7  =  tmp;
    }
  }

//third stage...
  if ( (log_stride - SHARED_SIZE_LIMIT_LOG) >= 2 )
  {
	if ( (data_0.y > data_1.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_0;
	  data_0  =  data_1;
	  data_1  =  tmp;
    }
    if ( (data_2.y > data_3.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_2;
	  data_2  =  data_3;
	  data_3  =  tmp;
    }
    if ( (data_4.y > data_5.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_4;
	  data_4  =  data_5;
	  data_5  =  tmp;
    }
    if ( (data_6.y > data_7.y ) == ddd) //swap when condition satisfied
    {
      Record tmp;
	  tmp     =  data_6;
	  data_6  =  data_7;
	  data_7  =  tmp;
    }
   }
    d_array[pos +        0]  =  data_0;
    d_array[pos + stride_1]  =  data_1;
    d_array[pos + stride_2]  =  data_2;
    d_array[pos + stride_3]  =  data_3;
    d_array[pos + stride_4]  =  data_4;
    d_array[pos + stride_5]  =  data_5;
    d_array[pos + stride_6]  =  data_6;
    d_array[pos + stride_7]  =  data_7;
}

__attribute((num_compute_units(NUM_CU_SCAN)))
__kernel //kid=9
void scanGroupLabel_kernel(__global Record* d_Rin, int rLen,__global int* d_groupLabel )
	{
		int bx = get_group_id(0);
		int tx = get_local_id(0);
		const int blockDimX=get_local_size(0);
		const int gridDimX=get_num_groups(0);
		int gridSize = blockDimX*gridDimX;
		int currentValue;
		int nextValue;
        char flag = 0;
		for( int idx = bx*blockDimX + tx; idx < rLen - 1; idx += gridSize )
		{
			currentValue = d_Rin[idx].y;
			nextValue = d_Rin[idx + 1].y;

			if( currentValue != nextValue )
			{
			   flag = 1;
			}
			else
			 {
			   flag = 0;
			 }
			
			d_groupLabel[idx + 1] = flag;
		}

		//write the first position
		if( (bx == 0) && (tx == 0) )
		{
			d_groupLabel[0] = 1;
		}
	}



/*
 * ScanLargeArrays_kernel : Scan is done for each block and the sum of each
 * block is stored in separate array (sumBuffer). SumBuffer is scanned
 * and results are added to every value of next corresponding block to
 * compute the scan of a large array.(not limited to 2*MAX_GROUP_SIZE)
 * Scan uses a balanced tree algorithm. See Belloch, 1990 "Prefix Sums
 * and Their Applications"
 * @param output output data 
 * @param input  input data
 * @param block  local memory used in the kernel
 * @param sumBuffer  sum of blocks
 * @param length length of the input data
 */

__attribute((num_compute_units(NUM_CU_SCAN_LARGE)))
__kernel //kid=19
void ScanLargeArrays_kernel(__global int *output,
               		__global int *input,
              		// __local  int *block,	 // Size : block_size
					const uint block_size,	 // size of block				
              		const uint length,	 	 // no of elements 
					 __global int *sumBuffer)  // sum of blocks
			
{
	int tid = get_local_id(0);
	int gid = get_global_id(0);
	int bid = get_group_id(0);
	__local int block[L_BLOCK_SIZE];
	int offset = 1;

    /* Cache the computational window in shared memory */
	block[2*tid]     = input[2*gid];
	block[2*tid + 1] = input[2*gid + 1];	

    /* build the sum in place up the tree */
	for(int d = block_size>>1; d > 0; d >>=1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(tid<d)
		{
			int ai = offset*(2*tid + 1) - 1;
			int bi = offset*(2*tid + 2) - 1;
			
			block[bi] += block[ai];
		}
		offset *= 2;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int group_id = get_group_id(0);
	if (tid == 0)	
	{
      /* store the value in sum buffer before making it to 0 */ 	
	  sumBuffer[bid]        = block[block_size - 1];
      /* clear the last element */
	  block[block_size - 1] = 0;	
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
    /* scan back down the tree */


    /* traverse down the tree building the scan in the place */
	for(int d = 1; d < block_size ; d *= 2)
	{
		offset >>=1;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(tid < d)
		{
			int ai = offset*(2*tid + 1) - 1;
			int bi = offset*(2*tid + 2) - 1;
			
			int t = block[ai];
			block[ai] = block[bi];
			block[bi] += t;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);	

    /*write the results back to global memory */
    output[2*gid]     = block[2*tid];
    output[2*gid + 1] = block[2*tid + 1];

}


__kernel //kid=18
void prefixSum_kernel(__global int *output, __global int *input, 
                      //__local  int *block, 
					  const uint length)
{
	int tid = get_local_id(0);
        __local int block[L_BLOCK_SIZE];
	int offset = 1;

    /* Cache the computational window in shared memory */
	block[2*tid]     = input[2*tid];
	block[2*tid + 1] = input[2*tid + 1];	

    /* build the sum in place up the tree */
	for(int d = length>>1; d > 0; d >>=1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(tid<d)
		{
			int ai = offset*(2*tid + 1) - 1;
			int bi = offset*(2*tid + 2) - 1;
			
			block[bi] += block[ai];
		}
		offset *= 2;
	}

    /* scan back down the tree */

    /* clear the last element */
	if(tid == 0)
	{
		block[length - 1] = 0;
	}

    /* traverse down the tree building the scan in the place */
	for(int d = 1; d < length ; d *= 2)
	{
		offset >>=1;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(tid < d)
		{
			int ai = offset*(2*tid + 1) - 1;
			int bi = offset*(2*tid + 2) - 1;
			
			int t = block[ai];
			block[ai] = block[bi];
			block[bi] += t;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

    /*write the results back to global memory */
	output[2*tid]     = block[2*tid];
	output[2*tid + 1] = block[2*tid + 1];
}

__attribute((num_simd_work_items(NUM_SIMD_ADD))) //
__attribute((num_compute_units(NUM_CU_ADD)))
__attribute((reqd_work_group_size(256,1,1))) 
//__attribute((num_compute_units(NUM_CU_BLOCK_ADD)))
__kernel//kid=17
void blockAddition_kernel(__global int* input, __global int* output)
{	
	int globalId = get_global_id(0);
	int groupId = get_group_id(0);
	int localId = get_local_id(0);

	__local int value[1];

	/* Only 1 thread of a group will read from global buffer */
	if(localId == 0)
	{
		value[0] = input[groupId];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	output[globalId] += value[0];
}

__kernel void //kid=2
groupByImpl_outSize_kernel(__global  int* d_outSize,__global  int* d_mark,__global  int* d_markOutput, int rLen )
{
	*d_outSize = d_mark[rLen-1] + d_markOutput[rLen-1];
}

__attribute((num_compute_units(NUM_CU_WRITE)))
__kernel  //kid=8
void groupByImpl_write_kernel(__global int* d_startPos,__global int* d_groupLabel,__global int* d_writePos, int rLen )
	{
		int bx = get_group_id(0);
		int tx = get_local_id(0);
		const int blockDimX=get_local_size(0);
		const int gridDimX=get_num_groups(0);
		int gridSize = blockDimX*gridDimX;

		for( int idx = bx*blockDimX + tx; idx < rLen; idx += gridSize )
		{
			if( d_groupLabel[idx] == 1 )
			{
				d_startPos[d_writePos[idx]] = idx;
			}
		}	
	}
/*
__kernel//kid=3
void mapBeforeGather_kernel(__global  Record* d_Rin, int rLen,__global  int* d_loc,__global  int* d_temp )
{
	int bx = get_group_id(0);
	int tx = get_local_id(0);
	int numThread = get_local_size(0);
	const int gridDimX=get_num_groups(0);
	int gridSize = numThread*gridDimX;

	for( int idx = bx*numThread + tx; idx < rLen; idx += gridSize )
	{
		d_loc[idx] = d_Rin[idx].x;
		d_temp[idx] = d_Rin[idx].y;
	}
}
*/

__attribute((num_compute_units(NUM_CU_AGG)))
__kernel//kid=4
void parallelAggregate_kernel(__global  Record* d_S,__global  int* d_startPos,__global  int* d_aggResults, int OPERATOR, int blockOffset, int numGroups, int rLen ,__local int* s_data  )
{
	const int blockDimX=get_local_size(0);
	int bx = get_group_id(0);
	int tx = get_local_id(0);
	int numThread =blockDimX;
	int idx = blockOffset + get_group_id(0);
	int start = d_startPos[idx];
	int end = (idx == (numGroups - 1))?(rLen):(d_startPos[idx + 1]);
//	int totalTx = bx*numThread + tx;
	int currentVal;
			//initialization the shared memory
	int l_init = 0;
	
	if( OPERATOR == REDUCE_MIN )
		l_init = TEST_MAX;
	 
    s_data[tx] = l_init;

    for( int i = start + tx; i < end; i = i + numThread )
    {
      currentVal = d_S[i].x;
      if( ((currentVal > s_data[tx])&(OPERATOR == REDUCE_MAX)) || ((currentVal < s_data[tx])&(OPERATOR == REDUCE_MIN))  )
      {
         s_data[tx] = currentVal;
      }
	  else if ((OPERATOR == REDUCE_SUM) || (OPERATOR == REDUCE_AVERAGE))
	  {
         s_data[tx] += currentVal;
	  }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

		int delta = 2;
		while( delta <= numThread )
		{
			int offset = delta*tx;
			if( offset < numThread )
			{
				//s_data[offset] = (s_data[offset] > s_data[offset + delta/2])?(s_data[offset]):(s_data[offset + delta/2]);
				//s_data[offset] = (s_data[offset] < s_data[offset + delta/2])?(s_data[offset]):(s_data[offset + delta/2]);
				//s_data[offset] = (s_data[offset]) + (s_data[offset + delta/2]);
				int A_a = s_data[offset];
				int A_b = s_data[offset + delta/2];
				int A_c = 0;
				if      (OPERATOR == REDUCE_MAX)
				{
				   A_c = A_a > A_b? A_a:A_b;
				}
				else if (OPERATOR == REDUCE_MIN)
				{
				   A_c = A_a < A_b? A_a:A_b;
				}
				else if ( (OPERATOR == REDUCE_SUM) || (OPERATOR == REDUCE_AVERAGE) )
				{
				   A_c = A_a + A_b;
				}
				s_data[offset] = A_c;
			}

			delta = delta*2;
			 barrier(CLK_LOCAL_MEM_FENCE);
		}

/*
	if( OPERATOR == REDUCE_MAX )
	{
		//initialization the shared memory
		s_data[tx] = 0;

		for( int i = start + tx; i < end; i = i + numThread )
		{
			currentVal = d_S[i].y;
			if( currentVal > s_data[tx] )
			{
				s_data[tx] = currentVal;
			}
		}

		 barrier(CLK_LOCAL_MEM_FENCE);

		int delta = 2;
		while( delta <= numThread )
		{
			int offset = delta*tx;
			if( offset < numThread )
			{
				s_data[offset] = (s_data[offset] > s_data[offset + delta/2])?(s_data[offset]):(s_data[offset + delta/2]);
			}

			delta = delta*2;
			 barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	else if( OPERATOR == REDUCE_MIN )
	{
		//initialization the shared memory
		s_data[tx] = TEST_MAX;

		for( int i = start + tx; i < end; i = i + numThread )
		{
			currentVal = d_S[i].y;
			if( currentVal < s_data[tx] )
			{
				s_data[tx] = currentVal;
			}
		}

		 barrier(CLK_LOCAL_MEM_FENCE);

		int delta = 2;
		while( delta <= numThread )
		{
			int offset = delta*tx;
			if( offset < numThread )
			{
				s_data[offset] = (s_data[offset] < s_data[offset + delta/2])?(s_data[offset]):(s_data[offset + delta/2]);
			}

			delta = delta*2;
			 barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	else if( (OPERATOR == REDUCE_SUM) || (OPERATOR == REDUCE_AVERAGE) )
	{
		//initialization the shared memory
		s_data[tx] = 0;

		for( int i = start + tx; i < end; i = i + numThread )
		{
			s_data[tx] += d_S[i].y;
		}

		 barrier(CLK_LOCAL_MEM_FENCE);

		int delta = 2;
		while( delta <= numThread )
		{
			int offset = delta*tx;
			if( offset < numThread )
			{
				s_data[offset] = (s_data[offset]) + (s_data[offset + delta/2]);
			}

			delta = delta*2;
			 barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
*/

//	 barrier(CLK_LOCAL_MEM_FENCE);

	if( tx == 0 )
	{  
		int p_result = 0;
		if( OPERATOR ==	REDUCE_AVERAGE )
		{
			p_result = s_data[0]/(end - start); //d_aggResults[idx]
		}
		else
		{
			p_result = s_data[0];
		}
		d_aggResults[idx] = p_result;
	}

}
/*
__kernel//kid=5
void copyLastElement_kernel(__global  int* d_odata,__global Record* d_Rin, int base, int offset)
{
	d_odata[offset] = d_Rin[base].y;
}
*/
/*
//with shared memory, with coalesced
__kernel//kid=6
void perscanFirstPass_kernel(__global int* temp, __global int* d_temp,__global int* d_odata,__global Record* d_idata, int numElementsPerBlock,  int isFull, int base, int d_odataOffset, int OPERATOR,int sharedMemSize )
{
{		
		int thid = get_local_id(0);
		int offset = 1;
		int numThread = get_local_size(0);
		int baseIdx = get_group_id(0)*(numThread*2) + base; 

		int mem_ai = baseIdx + thid;
		int mem_bi = mem_ai + numThread;

		int ai = thid;
		int bi = thid + numThread;

		int bankOffsetA = CONFLICT_FREE_OFFSET_REDUCE( ai );
		int bankOffsetB = CONFLICT_FREE_OFFSET_REDUCE( bi );

		d_temp[mem_ai] = d_idata[mem_ai].x;
		temp[ai + bankOffsetA] = d_idata[mem_ai].y;
		
		if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
		{
			if( isFull )
			{
				d_temp[mem_bi] = d_idata[mem_bi].x;
				temp[bi + bankOffsetB] = d_idata[mem_bi].y;
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : (0);
			}
		}
		else if( OPERATOR == REDUCE_MAX )
		{
			if( isFull )
			{
				d_temp[mem_bi] = d_idata[mem_bi].x;
				temp[bi + bankOffsetB] = d_idata[mem_bi].y;
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : ( TEST_MIN );
			}
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			if( isFull )
			{
				d_temp[mem_bi] = d_idata[mem_bi].x;
				temp[bi + bankOffsetB] = d_idata[mem_bi].y;
			}
			else
			{
				temp[bi + bankOffsetB] = (bi < numElementsPerBlock) ? (d_idata[mem_bi].y) : ( TEST_MAX );
			}
		}


		barrier(CLK_LOCAL_MEM_FENCE);

		//build sum in place up the tree
		if( OPERATOR == REDUCE_SUM || OPERATOR == REDUCE_AVERAGE )
		{
			for( int d = (numThread*2)>>1; d > 0; d >>= 1 )
			{
				barrier(CLK_LOCAL_MEM_FENCE);

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] += temp[ai];
				}

				offset *= 2;
			}
		}
		else if( OPERATOR == REDUCE_MAX )
		{
			for( int d = (numThread*2)>>1; d > 0; d >>= 1 )
			{
				barrier(CLK_LOCAL_MEM_FENCE);

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] = (temp[bi] > temp[ai])?(temp[bi]):(temp[ai]);				
				}

				offset *= 2;
			}
		}
		else if( OPERATOR == REDUCE_MIN )
		{
			for( int d = (numThread*2)>>1; d > 0; d >>= 1 )
			{
				barrier(CLK_LOCAL_MEM_FENCE);

				if( thid < d )
				{
					int ai = offset*( 2*thid + 1 ) - 1;
					int bi = offset*( 2*thid + 2 ) - 1;
					ai += CONFLICT_FREE_OFFSET_REDUCE( ai );
					bi += CONFLICT_FREE_OFFSET_REDUCE( bi );

					temp[bi] = (temp[bi] > temp[ai])?(temp[ai]):(temp[bi]);				
				}

				offset *= 2;
			}
		}


		barrier(CLK_LOCAL_MEM_FENCE);

		//write out the reduced block sums to d_odata
		if( thid == (numThread - 1)  )
		{
			d_odata[get_group_id(0) + d_odataOffset] = temp[bi+bankOffsetB];
		}
	}
}
*/

/*
__kernel void//kid=25
optGather_kernel( __global Record *d_R, int rLen, __global int *loc, int from, int to, 
		  __global Record *d_S, int sLen)
{
	int iGID = get_global_id(0);	
	int delta=get_global_size(0);
	int targetLoc=0;
	for(int pos=iGID;pos<sLen;pos+=delta)
	{
		targetLoc=loc[pos];
		if(targetLoc>=from && targetLoc<to)
		d_S[pos]=d_R[targetLoc];
	}
}
*/
	
