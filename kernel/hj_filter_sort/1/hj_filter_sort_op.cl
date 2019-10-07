#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

uint sim_hash(uint key, uint mod)
{
	return key % mod;
}

inline bool local_getlock(__local int *lock)
{
	return atomic_cmpxchg(lock, 0, 1) == 0;
}

inline void local_releaselock(__local int *lock)
{
	atomic_xchg(lock, 0);
	//*lock = 0;
}

#define LOCAL_MEMORY_NUM_BITS 15
#define LOCAL_MEMORY_NUM      (1<<LOCAL_MEMORY_NUM_BITS)  //32<<10)

__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))
__kernel void buildHashTable(__global uint * restrict rTableOnDevice, __global uint * restrict rHashTable,
	                        const uint size, const uint rHashTableBucketNum, const uint hashBucketSize, __global uint * restrict rHashCount)
{
	uint lid          = get_local_id(0);
	uint lsize        = get_local_size(0);	
	uint key, val, hash_lock, hash_index, count;
    __local int l_lock[LOCAL_MEMORY_NUM];
	for (int i = lid; i < LOCAL_MEMORY_NUM; i += lsize)
	  l_lock[i] = 0; 
    barrier(CLK_LOCAL_MEM_FENCE);	
	 
	for (uint i = lid; i < size; i += lsize)//while (tid < size) 		tid += lsize;
	{
		key = rTableOnDevice[i * 2 + 0];
		val = rTableOnDevice[i * 2 + 1];
		
		//1, get the local lock.... multiple counters (hash_index) share the local lock.
        hash_lock = key &(LOCAL_MEMORY_NUM-1);
        uint lock_status = 0;
        while (lock_status == 0)
        { 
            lock_status = local_getlock(&l_lock[hash_lock]);
        }
         
		//2, update the corresponding bucket with index: hash_index.
		hash_index = key & (rHashTableBucketNum-1); // real bucket to populate
		count = rHashCount[hash_index]++;            // compute the offset of the bucket..
		rHashTable[hash_index * hashBucketSize * 2 + count * 2 + 0] = key;	  //hash
        rHashTable[hash_index * hashBucketSize * 2 + count * 2 + 1] = val;      //hash

		//3, release the local lock.
		local_releaselock(&l_lock[hash_lock]);
	}
}

__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(PROBE_CU)))
__kernel void probeHashTable(__global uint * restrict rHashTable, __global uint * restrict sTableOnDevice, __global uint * restrict matchedTable, 
                            const uint size, const uint rHashTableBucketNum, const uint hashBucketSize, __global uint * restrict rHashCount)
{
	uint gsize        = get_global_size(0);
    uint gid          = get_global_id(0);
    uint lid          = get_local_id(0);

	int block_id      = get_group_id(0);
 //   int block_size    = get_num_groups(0);	
	uint unit_size    = (size<<1)/gsize;

	uint key, val, hash, count, matchedNum;
	uint hashBucketRealSize;

	__local uint local_counter[128];

	local_counter[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);	
		
		
    for (uint i = gid; i < size; i += gsize)//while (tid < size) 		tid += lsize;
	{
		key = sTableOnDevice[i * 2 + 0];
		val = sTableOnDevice[i * 2 + 1];

        hash = key &(rHashTableBucketNum-1);//sim_hash(key,rHashTableBucketNum);

		count = 0;
		hashBucketRealSize = rHashCount[hash];
			
        while(count < hashBucketRealSize) //before optimization:  hashBucketRealSize
        {
          uint key_prob   = rHashTable[hash * hashBucketSize * 2 + count * 2 + 0]; 
		  uint value_prob = rHashTable[hash * hashBucketSize * 2 + count * 2 + 1]; 

          if(key_prob == key)
          {
             uint result_offset = gid * unit_size + local_counter[lid]; //already *2/////
             matchedTable[result_offset + 0] = val;
             matchedTable[result_offset + 1] = value_prob;					
             local_counter[lid] += 2;
          }
		  count++;
		}
	}
}


typedef uint2 Record;
#define L_BLOCK_SIZE 256

//__attribute((num_simd_work_items(NUM_SIMD_MAP))) //
//__attribute((reqd_work_group_size(256,1,1))) 
__attribute((num_compute_units(NUM_CU_MAP)))
__kernel void//kid=20
filterImpl_map_kernel(__global Record* restrict d_Rin, int beginPos, int rLen, __global int* restrict d_mark, 
								  int smallKey, int largeKey, __global int* restrict d_temp ) //
{
	int iGID = get_global_id(0);
	Record value;
	int delta=get_global_size(0);
	int flag=0;
	for(int pos=iGID;pos<rLen;pos+=delta)
	{
		value = d_Rin[pos];
		//d_temp[pos] = value.x;
		int key = value.y;
		//the filter condition		
		if( ( key >= smallKey ) && ( key <= largeKey ) )
		{
			flag = 1;
		}
		else
		{
			flag=0;
		}
		d_mark[pos]=flag;
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
 


__attribute((num_compute_units(NUM_CU_SCAN)))
__kernel //kid=19
void ScanLargeArrays_kernel(__global int *restrict output,
               		__global int *restrict input,
              		// __local  int *block,	 // Size : block_size
					const uint block_size,	 // size of block				
              		const uint length,	 	 // no of elements 
					 __global int *restrict sumBuffer)  // sum of blocks
			
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
		
    /* store the value in sum buffer before making it to 0 */ 	
	sumBuffer[bid] = block[block_size - 1];

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
    /* scan back down the tree */

    /* clear the last element */
	block[block_size - 1] = 0;	

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
	{
		output[2*gid]     = block[2*tid];
		output[2*gid + 1] = block[2*tid + 1];
	}
}

__kernel //kid=18
void prefixSum_kernel(__global int * restrict output, __global int *restrict input, 
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
__kernel//kid=17
void blockAddition_kernel(__global int* restrict input, __global int* restrict output)
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



__kernel void //kid 21
filterImpl_outSize_kernel(__global int* d_outSize,__global int* d_mark,__global int* d_markOutput, int rLen )
{
	*d_outSize = d_mark[rLen-1] + d_markOutput[rLen-1];
}


//__attribute((num_simd_work_items(NUM_SIMD_FILTER))) //
//__attribute((reqd_work_group_size(256,1,1))) 
__attribute((num_compute_units(NUM_CU_FILTER)))
__kernel void//kid=23
filterImpl_write_kernel(__global Record* restrict d_Rout,__global Record* restrict d_Rin, __global int* restrict d_mark, __global int* restrict d_markOutput, int beginPos, int rLen )
{
	int iGID = get_global_id(0);
	Record value;
	int delta=get_global_size(0);
	int flag=0;
	int writePos=0;
	for(int pos=iGID;pos<rLen;pos+=delta)
	{
		flag=d_mark[pos];
		if(flag)
		{		
			writePos        =d_markOutput[pos];
			d_Rout[writePos]=d_Rin[pos];
		}
	}	
}

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
