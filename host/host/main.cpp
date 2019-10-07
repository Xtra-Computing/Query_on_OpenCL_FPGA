
//
// wzk, Xtra, SCE, NTU, Singapore.
///////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"
#include "common.h"

#include "testSort.h"
#include "testAggAfterGB.h"
extern void testScanImpl(int rLen);
extern void testFilterImpl( int rLen, int numThreadPB, int numBlock);
extern void testGroupByImpl( int rLen, int numThread, int numBlock);
extern void testSplitImpl(int rLen, int numPart, int numThreadPB , int numThreadBlock );
extern void dummy_kernel_call(int _CPU_GPU);

using namespace aocl_utils;

#define uint int
extern void dp_test(uint sub_index, float size);

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context Context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;

cl_program program_1 = NULL;
scoped_array<cl_command_queue> queue_1; // num_devices elements
scoped_array<cl_kernel> kernel_1; // num_devices elements
scoped_array<cl_kernel> kernel; // num_devices elements

cl_command_queue CommandQueue[1];

// Problem data.
const unsigned N = 1000000; // problem size
scoped_array<scoped_aligned_ptr<float> > input_a, input_b; // num_devices elements
scoped_array<scoped_aligned_ptr<float> > output; // num_devices elements
scoped_array<scoped_array<float> > ref_output; // num_devices elements
scoped_array<unsigned> n_per_device; // num_devices elements

// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void cleanup();
#if 0
struct kernel_info 
{   char names[32]; //name
	uint lut_c;     //LEs
	uint reg_c;     //Regs
	uint ram_c;     //RAMs
	uint dsp_c;     //DSPs
   float unit_cost; //
   float mem_extra; //
};

enum K_KERNEL_INDEX {
	K_SCAN_LARGE_ARRAY, //scan
	K_SCAN_PRE_SUM,     
	K_SCAN_BLOCK_ADD,

	K_FILTER_MAP,       //filter
	K_FILTER_WRITE,

	K_SORT_IM,             //sort

	K_GROUP_SCAN_GROUP_LABEL, //group
	K_GROUP_WRITE, 
	K_AGG, 

	K_NUM
};



uint opertor_seltection[5] = {
	K_FILTER_MAP, 
	K_SCAN_LARGE_ARRAY, //scan
	K_SCAN_PRE_SUM,     
	K_SCAN_BLOCK_ADD,
    K_FILTER_WRITE
};

uint opertor_ordering[1] = {
    K_SORT_IM
};

uint opertor_group_agg[7] = {
    K_SORT_IM,
	
	K_GROUP_SCAN_GROUP_LABEL, 
	
	K_SCAN_LARGE_ARRAY, //scan
	K_SCAN_PRE_SUM,     
	K_SCAN_BLOCK_ADD,

	K_GROUP_WRITE,
	K_AGG
};

////////////////////////////scan//////////////////////////
 struct kernel_info kernel_scan_large_array[5] = //kernel_scan_large_array[K_SCAN]
{
	{"ScanLargeArrays_kernel_1",  11888,  23356,  165,	8,  9.63, 1.01 }, 
	{"ScanLargeArrays_kernel_2",  22915,  45564,  330,	16,	 3.5, 1.01 }, 
	{"ScanLargeArrays_kernel_4",  45093,  90408,  664,	32, 2.46, 1.01 }, 
	{"ScanLargeArrays_kernel_8",  89077, 178812, 1320,	64,	0.88, 1.01 }, 
	{"ScanLargeArrays_kernel_10", 111131,223228, 1650,	80,  0.7, 1.01 }
};

static const struct kernel_info kernel_scan_prefixSum[1] = //kernel_scan_large_array[K_SCAN]
{
	{"prefixSum_kernel", 10598, 20726, 137, 8, 0.001, 0.01}
};

static const struct kernel_info kernel_scan_blockAddition[8] = //kernel_scan_large_array[K_SCAN]
{
	{"blockAddition_kerne_1_1",  7275, 10305,80, 0, 1.05, 1.0}, 
	{"blockAddition_kerne_1_2",  7010, 10163,80, 0,	0.53, 1.0}, 
	{"blockAddition_kerne_2_1",  13689,19462,160,0,	0.66, 1.0}, 
	{"blockAddition_kerne_2_4",  13229,19500,160,0, 0.29, 1.0}, 
	{"blockAddition_kerne_1_8",  7395, 10877, 80,0, 0.14, 1.0}, 
	{"blockAddition_kerne_4_8",  26997,40064,320,0,	0.12, 1.0}, 
	{"blockAddition_kerne_1_16", 8240, 12108, 80,0,	0.11, 1.0}, 
	{"blockAddition_kerne_8_4",  50333,74556,640,0, 0.11, 1.0}
};


////////////////////////////filter//////////////////////////
static const struct kernel_info kernel_filter_map[4] = //kernel_scan_large_array[K_SCAN]
{
	{"filterImpl_map_kernel_1",  4719,  7995, 38, 0, 1.07, 1.0 }, 
	{"filterImpl_map_kernel_2",  8577, 14842, 76, 0, 0.56, 1.0 }, 
	{"filterImpl_map_kernel_4",  16293,28536,152, 0, 0.41, 1.0 }, 
	{"filterImpl_map_kernel_8",  31725,55924,304, 0,  0.4, 1.0 }
};

static const struct kernel_info kernel_filter_write[4] = //kernel_scan_large_array[K_SCAN]
{
	{"filterImpl_write_kernel_1", 9598, 17788, 130, 0, 3.14, 1.0 }, 
	{"filterImpl_write_kernel_2", 18335,34428, 260, 0, 3.23, 1.0 }, 
	{"filterImpl_write_kernel_4", 35809,67708, 520, 0, 2.28, 1.0 }, 
	{"filterImpl_write_kernel_8", 70757,134268,1040,0, 1.86, 1.0 }
};


//////////////////////////sort//////////////////////////
static const struct kernel_info kernel_sort[6] = //kernel_scan_large_array[K_SCAN]
{
	{"sort_global_local_kernel_1_1", 20105,  71480, 473, 0, 147.35, 1.0 }, 
	{"sort_global_local_kernel_2_1", 30900, 121843, 770, 0,  86.06, 1.0 }, 
	{"sort_global_local_kernel_2_2", 38488, 140664, 922, 0,  94.94, 1.0 }, 
	{"sort_global_local_kernel_4_2", 53664, 178306, 1226,0,  62.86, 1.0 },
	{"sort_global_local_kernel_8_2", 84016, 253590, 1834,0,  38.05, 1.0 }, 
	{"sort_global_local_kernel_10_2",99192, 291232, 2138,0,  26.63, 1.0 }

};

//////////////////////group_by///////////////////////////////////////////////////////
static const struct kernel_info kernel_group_scanGroupLabel[5] = //kernel_scanGroupLabel[K_SCAN]
{
	{"scanGroupLabel_kernel_1", 6862, 12824, 66, 4,  5.6, 1.0 }, 
	{"scanGroupLabel_kernel_2", 12863,24500,132, 8, 5.02, 1.0 }, 
	{"scanGroupLabel_kernel_4", 24865,47852,264,16, 6.23, 1.0 }, 
	{"scanGroupLabel_kernel_8", 48869,94556,528,32,	3.35, 1.0 },
	{"scanGroupLabel_kernel_16",96877,187964,1056,64,1.46,1.0 }
};

static const struct kernel_info kernel_group_write[4] = //groupByImpl_write_kernel[K_SCAN]
{
	{"groupByImpl_write_kernel_1", 4173, 11921, 59, 4, 6.69, 1.0 }, 
	{"groupByImpl_write_kernel_2", 7485, 22694,118, 8, 6.89, 1.0 }, 
	{"groupByImpl_write_kernel_4", 14109,44240,236,16, 6.65, 1.0 }, 
	{"groupByImpl_write_kernel_8", 27357,87332,472,32, 5.04, 1.0 }
};

static const struct kernel_info kernel_group_parallelAggregate[4] = //kernel_scanGroupLabel[K_SCAN]
{
	{"parallelAggregate_kernel_1", 18018,  33510, 282, 10, 0.96, 1.0 }, 
	{"parallelAggregate_kernel_2", 35175,  65872, 540, 20, 0.54, 1.0 }, 
	{"parallelAggregate_kernel_4", 69489, 130596,1056, 40, 0.28, 1.0 }, 
	{"parallelAggregate_kernel_6", 103803,195320,1572, 60,  0.2, 1.0 }
};

uint kernel_addr[K_NUM] = {  // point to kernel_info*.....
   (uint)kernel_scan_large_array,
   (uint)kernel_scan_prefixSum,
   (uint)kernel_scan_blockAddition,

   (uint)kernel_filter_map,
   (uint)kernel_filter_write,

   (uint)kernel_sort,

   (uint)kernel_group_scanGroupLabel,
   (uint)kernel_group_write,
   (uint)kernel_group_parallelAggregate 
};

uint kernel_number[K_NUM] = {
	5, //scan
	1,     
	8,

	4,       //filter
	4,

	6,       //sort

	5, //group
	4, 
	4
};

enum OPERATORS {
	K_SCAN,
	K_FILTER,
	K_SORT,
	K_AGGREGATION,

	K_NUM_KERNELS
};

static int operator_kernel_number[K_NUM_KERNELS] =
{
  6, //scan
  8, //filter
  6, //aggregation
  6  // sort
};
#endif

static const char* kernel_names_scan[13] = //kernel_number[K_SCAN]
{
	"scan_CU_op_1_4_8", 
	"scan_CU_op_1_8_4",
	"scan_CU_op_2_1_8",
	"scan_CU_op_2_8_1",
	"scan_CU_op_4_1_2",
	"scan_CU_op_4_2_1",
	"scan_CU_op_8_2_4",
	"scan_CU_op_8_4_2",
	"scan_CU_op_10_1_1", //8: new one
	"scan_CU_op_1_1_16",
	"scan_CU_op_1_2_16",
	"scan_CU_op_1_4_16",
	"scan_CU_op_1_8_16"

};

static const char* kernel_names_filter[10] = //kernel_number[K_FILTER]
{
	"filter_CU_op_1_1_1_1_1_1_1", 
	"filter_CU_op_1_1_1_1_1_2_2",
	"filter_CU_op_1_1_1_1_1_4_4",
	"filter_CU_op_1_1_1_1_1_8_8",
	"filter_CU_op_1_1_2_2_1_2_2",

	"filter_CU_op_1_1_4_16_1_1_1",            //
	"filter_CU_op_1_1_16_4_1_1_1",

	"filter_CU_op_1_1_8_8_1_8_8",
	"filter_CU_op_1_1_4_4_1_4_4",
	"filter_op_2_8_1-16_4"
};

#ifdef USE_SHARED_MEMORY_ENALBLE
static const char* kernel_names_sort[12] = //kernel_number[K_SORT]
{
	"sort_sm_CU_op_stage_r_1_1",
	"sort_sm_CU_op_stage_r_2_2",
	"sort_sm_CU_op_stage_r_2_4",
	"sort_sm_CU_op_stage_r_4_4",
	"sort_sm_CU_op_stage_r_6_4",
	"sort_sm_CU_op_stage_r_2_1",
	"sort_sm_CU_op_stage_r_4_2",
	"sort_sm_CU_op_stage_r_10_2",
	"sort_sm_CU_op_stage_r_8_2",
    "sort_sm_CU_op_stage_r_4_1",
	"sort_sm_CU_op_stage_r_8_1",
	"sort_sm_CU_op_stage_r_10_1"
};
#else
static const char* kernel_names_sort[6] = //kernel_number[K_SCAN]
{
	"sort_CU_1", 
	"sort_CU_2",
	"sort_CU_4",
	"sort_CU_8",
	"sort_CU_16",
	"sort_CU_24"
};
#endif

static const char* kernel_names_aggr[12] = //kernel_number[K_SCAN]
{
	"group_by_cu_op_1_1_1_1_1_1", 
	"group_by_cu_op_1_1_1_1_1_2",
	"group_by_cu_op_1_1_1_1_1_4",
	"group_by_cu_op_1_1_1_1_2_1",
	"group_by_cu_op_1_1_1_1_4_1",
	"group_by_cu_op_1_1_1_1_8_1",

	"group_by_cu_op_1_1_1_1_1_6", 
	"group_by_cu_op_1_2_1_1_1_1", 
	"group_by_cu_op_1_4_1_1_1_1", 
	"group_by_cu_op_1_8_1_1_1_1", 
	"group_by_cu_op_1_16_1_1_1_1",
	"group_by_cu_op_dp_4_2_1_1_1-16_1_1"
};


static const char* kernel_names_aggr_0[1] = //kernel_number[K_SCAN]
{
	"group_by_cu_op_dp_1__10_2"
};

static const char* kernel_names_aggr_1[2] = //kernel_number[K_SCAN]
{
	"group_by_cu_op_dp_2_8_2_1-16_8_1",
	"group_by_cu_op_dp_2_8_2_1-16_8_1_dummy"
};

/*
static const char* kernel_names[K_NUM_KERNELS] = 
{
	"scan_use", 
	"filter_use",
	"group_by_real_use",
	"sort_use"
};
*/
  // representative device (assuming all device are of the same type). group_by_use split_use
 // std::string binary_file = getBoardBinaryFile("filter_use", device[0]); //   

int kernel_index   = 0;
int test_size      = 0;
int kernel_index_1 = 0;
int sub_index;
char bin_file[100]; char bin_file_1[100];


// Entry point.
int main(int argc, char **argv) {

 if (argc > 1)
  {	   
    kernel_index = atoi(argv[1]);
    test_size    = atoi(argv[2]);
    sub_index    = atoi(argv[3]);
  }
 else if (argc == 1)
 {
    kernel_index = 10;  //2; //
    test_size    = 4*1024*1024;  //*1024*1024;
    sub_index    = 2;   //7;	0 //11;
 }

 if (kernel_index == 10) //for dynamic programming...
 {   
     if (test_size >=  1024*1024)
	    test_size  = test_size/(1024*1024); 
     dp_test(sub_index, (float)test_size);
	 return 0;
 }

 if (kernel_index == 0)       //scan
    strcpy(bin_file,kernel_names_scan[sub_index]); 
 else if (kernel_index == 1)  //filter
    strcpy(bin_file,kernel_names_filter[sub_index]); 
 else if (kernel_index == 2)  //sort
    strcpy(bin_file,kernel_names_sort[sub_index]); 
 else if (kernel_index == 3)  //aggregation
    strcpy(bin_file,kernel_names_aggr[sub_index]); 

 else if (kernel_index == 9)  //dummy kernel launching
    strcpy(bin_file,"blank"); 

 else if (kernel_index == 30)  //dummy kernel launching
 {
    strcpy(bin_file,  kernel_names_aggr_0[sub_index]); //"blank"
    strcpy(bin_file_1,kernel_names_aggr_1[sub_index+1]);     //"blank"
 }


 else 
    printf("bug for input configuration\n"); //

 printf("\nkernel_index = %d, sub_index = %d, test_size = 0x%x\n", kernel_index, sub_index, test_size);


  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  // Initialize the problem data.
  // Requires the number of devices to be known.
  init_problem();

  // Run the kernel.
  run();

  // Free the resources allocated
  //cleanup();

  return 0;
}

/////// HELPER FUNCTIONS ///////

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

 // printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
 // printf("platform: %s\n", getPlatformName(platform).c_str());
 // printf("Using %d device(s)\n", num_devices);
//  for(unsigned i = 0; i < num_devices; ++i) {
 //   printf("  %s\n", getDeviceName(device[i]).c_str());
//  }

  // Create the Context.
  Context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
  checkError(status, "Failed to create Context");

  // Create the program for all device. Use the first device as the vectorAdd kernel_names[kernel_index]
  // representative device (assuming all device are of the same type). group_by_use split_use scan_use filter sort_use group_by_real_use
  std::string binary_file = getBoardBinaryFile(bin_file, device[0]); //filter_use
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(Context, binary_file.c_str(), device, num_devices);
  //char binary_file[] = "vectorAdd.aocx"; 
 // program = createProgramFromBinary(Context, binary_file, device, num_devices);


  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");
  queue.reset(num_devices);

  if (kernel_index == 30)  //dummy kernel launching
  {
     std::string binary_file_1 = getBoardBinaryFile(bin_file_1, device[0]); //filter_use
     printf("Using AOCX_1: %s\n", binary_file_1.c_str());
     program_1 = createProgramFromBinary(Context, binary_file_1.c_str(), device, num_devices);
     // Build the program that was just created.
     status = clBuildProgram(program_1, 0, NULL, "", NULL, NULL);
     checkError(status, "Failed to build program");
     queue_1.reset(num_devices);
  }


  kernel.reset(num_devices);
  n_per_device.reset(num_devices);
 // input_a_buf.reset(num_devices);
 // input_b_buf.reset(num_devices);
//  output_buf.reset(num_devices);

   CommandQueue[0] = clCreateCommandQueue(Context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
   checkError(status, "Failed to create command queue CommandQueue");

  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(Context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    queue_1[i] = clCreateCommandQueue(Context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Determine the number of elements processed by this device.
    n_per_device[i] = N / num_devices; // number of elements handled by this device

    // Spread out the remainder of the elements over the first
    // N % num_devices.
    if(i < (N % num_devices)) {
      n_per_device[i]++;
    }
  }

  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }
}

void run() {
  cl_int status;

 // const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  scoped_array<cl_event> kernel_event(num_devices);
  scoped_array<cl_event> finish_event(num_devices);


  // Wait for all devices to finish.
  //clWaitForEvents(num_devices, finish_event);

   
    //testScanImpl(64*1024*1024);
   // testFilterImpl( 16*1024*1024, 256, 128);
	//testSortImpl(16*1024*1024, 256, 4); 
    //testGroupByImpl(1024*1024, 256, 4);
    //testSplitImpl(16*1024*1024, 32, 128 , 128);
    //testAggAfterGroupByImpl( 16*1024*1024, REDUCE_MAX, 256, 1024);
//  const double end_time = getCurrentTimestamp();
  double start_time, end_time;
               switch (kernel_index) {

                    case 0:{
						 //printf("\nscan:");
                         testScanImpl(test_size);
                         break;   
                        }
                    case 1:{
 						 printf("filter:\n");
                        testFilterImpl(test_size, 256, 128);
                         break;   
                        }

                    case 2:
						 printf("sort:");
                         testSortImpl(test_size, 256, 4); //testSortImpl(test_size, 256, 128);
                         break;
					case 3  :
					case 30 :
						printf("agg:\n");
                         testAggAfterGroupByImpl(test_size, REDUCE_MAX, 256, 128 );
                         break;

                    case 4:
						 printf("split:\n");
                         testSplitImpl(16*1024*1024, 32, 128 , 128);//testSplitImpl(test_size, 8, 256, 128);
                         break;

                    case 9:

						 printf("dummy kernel launching:\n");
                           start_time = getCurrentTimestamp();
                         dummy_kernel_call(0);
                           end_time = getCurrentTimestamp();
                         printf("dummy kernel Time: %0.3f ms\n",   (end_time - start_time)* 1e3);
                         break;
						//  else if (kernel_index == 9)  //dummy kernel launching

                    default:
                        break;
        }

  // Wall-clock time taken.
//  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);
/*
  // Get kernel times using the OpenCL event profiling API.
  for(unsigned i = 0; i < num_devices; ++i) {
    cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
  }
*/
  // Release all events.
  for(unsigned i = 0; i < num_devices; ++i) {
  //  clReleaseEvent(kernel_event[i]);
  //  clReleaseEvent(finish_event[i]);
  }
/*
  // Verify results.
  bool pass = true;
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    for(unsigned j = 0; j < n_per_device[i] && pass; ++j) {
      if(fabsf(output[i][j] - ref_output[i][j]) > 1.0e-5f) {
        printf("Failed verification @ device %d, index %d\nOutput: %f\nReference: %f\n",
            i, j, output[i][j], ref_output[i][j]);
        pass = false;
      }
    }
  }

  printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");
*/
}

// Free the resources allocated during initialization
void cleanup() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
  }
  
  if(program) {
    clReleaseProgram(program);
  }
  if(Context) {
    clReleaseContext(Context);
  }
}

