#include "common.h"
void radixSortImpl(cl_mem d_R, int rLen, int keybits, int numThreadPB, int numBlock,int *index,cl_event *eventList,cl_kernel *Kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU);
void testSortImpl(int rLen, int numThreadPB, int numBlock);
void kernel_enqueue(int size,int kid,cl_uint work_dim, const size_t *groups, size_t *threads,
	     cl_event *List,int* index,cl_kernel *kernel,int *Flag_CPU_GPU,double * burden,int _CPU_GPU);