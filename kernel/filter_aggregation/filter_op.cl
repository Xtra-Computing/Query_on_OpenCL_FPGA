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
