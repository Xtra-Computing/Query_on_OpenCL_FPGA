/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

typedef uint2 Record;
#define L_BLOCK_SIZE 256

#define SHARED_SIZE_LIMIT_LOG 10
#define SHARED_SIZE_LIMIT     (1<<SHARED_SIZE_LIMIT_LOG)//1024


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


#if 0

//Combined bitonic merge steps for
//size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__attribute((num_compute_units(NUM_CU_S_1)))
__attribute__((reqd_work_group_size(SHARED_SIZE_LIMIT/2,1,1)))
__kernel void bitonicMergeShared(
    __global Record * d_array,
    uint arrayLength,
    uint size,
    uint dir
)
{
/*    //Shared memory storage for current subarray
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
*/
    //local memory storage for current subarray
    __local Record s_array[SHARED_SIZE_LIMIT];

      uint gid = get_global_id(0);
      uint lid = get_local_id(0);
      uint bid = get_group_id(0);

	  d_array += bid * SHARED_SIZE_LIMIT + lid;

	  s_array[lid  +                       0]  = d_array[0                          ];
	  s_array[lid  + (SHARED_SIZE_LIMIT / 2)]  = d_array[0 + (SHARED_SIZE_LIMIT / 2)];

    //Bitonic merge
    uint comparatorI = gid & ((arrayLength / 2) - 1);
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);

    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
    {
/*        __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(
            s_key[pos +      0], s_val[pos +      0],
            s_key[pos + stride], s_val[pos + stride],
            ddd
        );
*/
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
/*
    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
*/
    barrier(CLK_LOCAL_MEM_FENCE); //  __syncthreads();
    d_array[0  +                       0]  = s_array[lid                          ];
    d_array[0  + (SHARED_SIZE_LIMIT / 2)]  = s_array[lid + (SHARED_SIZE_LIMIT / 2)];
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L)
{
    if (!L)
    {
        *log2L = 0;
        return 0;
    }
    else
    {
        for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);

        return L;
    }
}

extern "C" uint bitonicSort(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint dir
)
{
    //Nothing to sort
    if (arrayLength < 2)
        return 0;

    //Only power-of-two array lengths are supported by this implementation
    uint log2L;
    uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);

    dir = (dir != 0);

    uint  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;
    uint counter_debug = 0;
    if (arrayLength <= SHARED_SIZE_LIMIT)
    {
        assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
        bitonicSortShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
    }
    else
    {
        bitonicSortShared1<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal);

        for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1) // >= SHARED_SIZE_LIMIT/2
                if (stride >= SHARED_SIZE_LIMIT)
                {
                    bitonicMergeGlobal<<<(batchSize * arrayLength) / 512, 256>>>(d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, stride, dir);
                }
                else
                {   counter_debug++;
                    bitonicMergeShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, dir);
                    break;
                }
    }
	printf("bitonicMergeShared runs %d times\n", counter_debug);

    return threadCount;
}
#endif