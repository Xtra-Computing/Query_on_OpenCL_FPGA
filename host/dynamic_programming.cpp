//This file it used to compute the FPGA configuration for each  
// wzk, Xtra, SCE, NTU, Singapore.
///////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "AOCL_Utils.h"


using namespace aocl_utils;

#define uint int

uint data_size = 64.0;

struct kernel_info 
{   char names[32]; //name
	uint lut_c;     //LEs
	uint reg_c;     //Regs
	uint ram_c;     //RAMs
	uint dsp_c;     //DSPs
   float unit_cost; //
   float unit_mem_add; //
   float unit_mem_sub; //

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

#define SELECTION_RATE 0.4
/*
////////////////////////////scan//////////////////////////
static const struct kernel_info kernel_scan_large_array[5] = //kernel_scan_large_array[K_SCAN]
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
*/
////////////////////////////scan//////////////////////////
static const struct kernel_info kernel_primitive[41] = //kernel_scan_large_array[K_SCAN]
{
	{"ScanLargeArrays_kernel_1",  11888,  23356,  165,	8,  9.63, 1.01, 0.0  }, //1.01
	{"ScanLargeArrays_kernel_2",  22915,  45564,  330,	16,	 3.5, 1.01, 0.0  }, 
	{"ScanLargeArrays_kernel_4",  45093,  90408,  664,	32, 2.46, 1.01, 0.0  }, 
	{"ScanLargeArrays_kernel_8",  89077, 178812, 1320,	64,	0.88, 1.01, 0.0  }, 
	{"ScanLargeArrays_kernel_10", 111131,223228, 1650,	80,  0.7, 1.01, 0.0  },

	{"prefixSum_kernel", 10598, 20726, 137, 8, 0.001, 0.01, 0.0 },              //1.01

	{"blockAddition_kerne_1_1",  7275, 10305,80, 0, 1.05, 0.0 , -1.02},         //1.0
	{"blockAddition_kerne_1_2",  7010, 10163,80, 0,	0.53, 0.0 , -1.02}, 
	{"blockAddition_kerne_2_1",  13689,19462,160,0,	0.66, 0.0 , -1.02}, 
	{"blockAddition_kerne_2_4",  13229,19500,160,0, 0.29, 0.0 , -1.02}, 
	{"blockAddition_kerne_1_8",  7395, 10877, 80,0, 0.14, 0.0 , -1.02}, 
	{"blockAddition_kerne_4_8",  26997,40064,320,0,	0.12, 0.0 , -1.02}, 
	{"blockAddition_kerne_1_16", 8240, 12108, 80,0,	0.11, 0.0 , -1.02}, 
	{"blockAddition_kerne_8_4",  50333,74556,640,0, 0.11, 0.0 , -1.02},

	{"filterImpl_map_kernel_1",  4719,  7995, 38, 0, 1.07, 1.0, 0.0 },        //1.0
	{"filterImpl_map_kernel_2",  8577, 14842, 76, 0, 0.56, 1.0, 0.0 }, 
	{"filterImpl_map_kernel_4",  16293,28536,152, 0, 0.41, 1.0, 0.0 }, 
	{"filterImpl_map_kernel_8",  31725,55924,304, 0,  0.4, 1.0, 0.0 },

	{"filterImpl_write_kernel_1", 9598, 17788, 130, 0, 3.14, SELECTION_RATE, -2.0 },      //checked  ...could be -2.0
	{"filterImpl_write_kernel_2", 18335,34428, 260, 0, 2.02, SELECTION_RATE, -2.0 },   //2.45, 
	{"filterImpl_write_kernel_4", 35809,67708, 520, 0, 1.01, SELECTION_RATE, -2.0 },       //1.78
	{"filterImpl_write_kernel_8", 70757,134268,1040,0, 0.55, SELECTION_RATE, -2.0 },           //1.02

	{"sort_global_local_kernel_1_1", 20105,  71480, 473, 0, 147.35, 0.0, 0.0 }, // in-place sorting: 0.0
	{"sort_global_local_kernel_2_1", 30900, 121843, 770, 0,  86.06, 0.0, 0.0 }, 
	{"sort_global_local_kernel_2_2", 38488, 140664, 922, 0,  94.94, 0.0, 0.0 }, 
	{"sort_global_local_kernel_4_2", 53664, 178306, 1226,0,  62.86, 0.0, 0.0 },
	{"sort_global_local_kernel_8_2", 84016, 253590, 1834,0,  38.05, 0.0, 0.0 }, 
	{"sort_global_local_kernel_10_2",99192, 291232, 2138,0,  26.63, 0.0, 0.0 },

	{"scanGroupLabel_kernel_1", 6862, 12824, 66, 4,  5.6, 1.0, 0.0  },          // right----groupLabel: 1.0
	{"scanGroupLabel_kernel_2", 12863,24500,132, 8, 5.02, 1.0, 0.0  }, 
	{"scanGroupLabel_kernel_4", 24865,47852,264,16, 6.23, 1.0, 0.0  }, 
	{"scanGroupLabel_kernel_8", 48869,94556,528,32,	3.35, 1.0, 0.0  },
	{"scanGroupLabel_kernel_16",96877,187964,1056,64,1.46,1.0, 0.0  },

	{"groupByImpl_write_kernel_1", 4173, 11921, 59, 4, 6.69, SELECTION_RATE, -2.0 },       // not determined 0.5 - 2.0
	{"groupByImpl_write_kernel_2", 7485, 22694,118, 8, 6.89, SELECTION_RATE, -2.0 }, 
	{"groupByImpl_write_kernel_4", 14109,44240,236,16, 6.65, SELECTION_RATE, -2.0 }, 
	{"groupByImpl_write_kernel_8", 27357,87332,472,32, 5.04, SELECTION_RATE, -2.0 },


	{"parallelAggregate_kernel_1", 18018,  33510, 282, 10, 0.96, SELECTION_RATE,-1.0+SELECTION_RATE }, //not determined 0.5   
	{"parallelAggregate_kernel_2", 35175,  65872, 540, 20, 0.54, SELECTION_RATE,-1.0+SELECTION_RATE }, 
	{"parallelAggregate_kernel_4", 69489, 130596,1056, 40, 0.28, SELECTION_RATE,-1.0+SELECTION_RATE }, 
	{"parallelAggregate_kernel_6", 103803,195320,1572, 60,  0.2, SELECTION_RATE,-1.0+SELECTION_RATE }
};

/*
struct kernel_info *kernel_addr[K_NUM] = {  // point to kernel_info*.....
   kernel_scan_large_array,
   kernel_scan_prefixSum,
   kernel_scan_blockAddition,

   kernel_filter_map,
   kernel_filter_write,

   kernel_sort,

   kernel_group_scanGroupLabel,
   kernel_group_write,
   kernel_group_parallelAggregate 
};
*/
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

uint kernel_offset[K_NUM] = {  // point to kernel_info*.....
   0, //(uint)kernel_scan_large_array,
   5, //(uint)kernel_scan_prefixSum,
   6, //(uint)kernel_scan_blockAddition,
   14, //(uint)kernel_filter_map,
   18, //(uint)kernel_filter_write,
   22, //(uint)kernel_sort,
   28, //(uint)kernel_group_scanGroupLabel,
   33, //(uint)kernel_group_write,
   37  //(uint)kernel_group_parallelAggregate 
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

uint opertor_filter_group_agg[12] = {
	K_FILTER_MAP, 
	K_SCAN_LARGE_ARRAY, //scan
	K_SCAN_PRE_SUM,     
	K_SCAN_BLOCK_ADD,
    K_FILTER_WRITE,

    K_SORT_IM,
	
	K_GROUP_SCAN_GROUP_LABEL, 
	
	K_SCAN_LARGE_ARRAY, //scan
	K_SCAN_PRE_SUM,     
	K_SCAN_BLOCK_ADD,

	K_GROUP_WRITE,
	K_AGG
};

//for each evaluate.....
struct image_info 
{
	char kernel_indexs[16];
	char kernel_id[16];

	uint kernel_valid_num;
	uint luts;
	uint regs;
	uint rams;
	uint dsps;

	float total_unit_cost;
	float freq;
	float exec_time;
	float mem_inc;
	float reconfig_time;
};

struct S_Min_Exec_Time 
{
	float exec_time;
	uint image_num;
	
	struct image_info  inst_image[10];
};

//#define data_size 64.0  //1.0//

uint l_kernel_index[100];
uint l_kernel_id[100];

float compute_freq(uint luts, uint regs, uint rams, uint dsps )
{
  uint luts_all = luts + 35437;
  uint regs_all = regs + 54855;
  uint rams_all = rams + 283;
  uint dsps_all = dsps + 0;
  float luts_rate = ( (float)luts_all )/636928.0;
  float regs_rate = ( (float)regs_all )/944128.0;
  float rams_rate = ( (float)rams_all )/2560.0;
  float dsps_rate = ( (float)dsps_all )/2560.0;
  
  float max_rate_0 = luts_rate > regs_rate? luts_rate:regs_rate;
  float max_rate_1 = rams_rate > dsps_rate? rams_rate:dsps_rate;
  float max_rate   = max_rate_0>max_rate_1? max_rate_0:max_rate_1;

  float freq;
  if ((luts_rate < 0.85) && (regs_rate < 0.85) && (rams_rate < 0.95)) 
    freq = -85.69*max_rate*max_rate - 22.02*max_rate + 278.97;
  else
    freq = 0.01;
  return freq;
}

float recursive_time_compute(uint *opertor_seltection, uint *check_len, uint luts_base, uint regs_base, uint rams_base, uint dsps_base, float unit_cost_base,
							uint num_id, uint num, struct image_info *image_temp)
{
	if (num_id == 0)
		image_temp->exec_time  = 99999999.0;

    uint index                             = opertor_seltection[num_id];
	uint k_num                             = kernel_number[index];
	uint offset                            = kernel_offset[index];
	uint times_len                         = check_len[num_id];

	uint num_id_next                             = num_id + 1;

	l_kernel_id[num_id]                          = index;

    for (uint i = 0; i < k_num; i++)
    {    
      uint luts       = kernel_primitive[offset + i].lut_c;    //p_primitive_offset[i].lut_c;
      uint regs       = kernel_primitive[offset + i].reg_c;    //p_primitive_offset[i].reg_c;
      uint rams       = kernel_primitive[offset + i].ram_c;    //p_primitive_offset[i].ram_c;
      uint dsps       = kernel_primitive[offset + i].dsp_c;    //p_primitive_offset[i].dsp_c;
	  float unit_cost = times_len * kernel_primitive[offset + i].unit_cost;//p_primitive_offset[i].unit_cost;

      //bug for the sort, which is x^2,not linear.
      if ( (index == K_SORT_IM) && (data_size < 16) )
          unit_cost = unit_cost - (128 - data_size)*0.2; //rduce the weight for the sort with small number of tuples... 0.12

	  l_kernel_index[num_id]     = i;

	  if (num_id == (num-1)) //recursive to the nestest loop.....
	  {
	    float freq      = compute_freq(luts_base + luts, regs_base + regs, rams_base + rams, dsps_base + dsps); //compute the freq...

        float time      = ( (unit_cost_base + unit_cost)  * (float)data_size )/freq;

	//	printf("\n l_kernel_index[0] = %d, i =%d, num_id = %d, freq = %f, time = %f, unit_cost = %f, luts = %d, regs = %d, rams = %d, dsps = %d\n", 
	//		     l_kernel_index[0],        i, num_id,      freq,      time,      unit_cost_base + unit_cost, luts_base + luts, regs_base + regs, rams_base + rams, dsps_base + dsps);

	    if (time < image_temp->exec_time) //initiazed to 99999999.0
	    { image_temp->total_unit_cost = unit_cost_base + unit_cost;
	      image_temp->exec_time   = time;
		  image_temp->freq        = freq;
		  image_temp->kernel_valid_num = num;

		  for (uint ii = 0; ii < num; ii++)
		  {
			  image_temp->kernel_indexs[ii] = l_kernel_index[ii]; //index_min  = i;
		      image_temp->kernel_id[ii]     = opertor_seltection[ii];
		  }

		  image_temp->luts = luts_base + luts;
		  image_temp->regs = regs_base + regs;
		  image_temp->rams = rams_base + rams;
		  image_temp->dsps = dsps_base + dsps;
	    }	  
	  }
	  else
	  {

        recursive_time_compute(opertor_seltection, check_len, luts_base + luts, regs_base + regs, rams_base + rams, dsps_base + dsps, 
			                unit_cost_base + unit_cost, num_id_next, num, image_temp);	  
	  }

    }
	return image_temp->exec_time;
}

void one_image_min_exec_time (uint *opertor_seltection, uint num, struct image_info *image_temp)
{
 if (num == 1)
 {
	float time_min                           = 99999999.0;
	uint index_min                           = 0;
	uint luts_min, regs_min, rams_min, dsps_min;
	float freq_min;

    uint index_0                               = opertor_seltection[0];
	uint k_num_0                             = kernel_number[index_0];
	uint offset_0                            = kernel_offset[index_0];
	  
	printf("index_0 = %d\n",                 index_0);
	printf("k_num_0 = %d\n",                 k_num_0);
	printf("offset_0 = 0x%x\n",              offset_0);
		
/*		
    struct kernel_info *p_primitive_offset = (struct kernel_info *) (kernel_addr[offset_0]);
    uint option_num_0                          = kernel_number[offset_0];

    printf("p_primitive_offset    = 0x%x\n", (uint)p_primitive_offset );
	printf("kernel_filter_map     = 0x%x\n", (uint)kernel_filter_map );
	//kernel_filter_map
*/
    for (uint i = 0; i < k_num_0; i++)
    {
      uint luts       = kernel_primitive[offset_0 + i].lut_c;    //p_primitive_offset[i].lut_c;
      uint regs       = kernel_primitive[offset_0 + i].reg_c;    //p_primitive_offset[i].reg_c;
      uint rams       = kernel_primitive[offset_0 + i].ram_c;    //p_primitive_offset[i].ram_c;
      uint dsps       = kernel_primitive[offset_0 + i].dsp_c;    //p_primitive_offset[i].dsp_c;
	  float unit_cost = kernel_primitive[offset_0 + i].unit_cost;//p_primitive_offset[i].unit_cost;

	  float freq      = compute_freq(luts, regs, rams, dsps); //compute the freq...
      float time      = ( unit_cost * (float)data_size )/freq;
	  if (time < time_min)
	  {
	    time_min   = time;
		freq_min   = freq;
		index_min  = i;
		luts_min   = luts;
		regs_min   = regs;
		rams_min   = rams;
		dsps_min   = dsps;
	  }
    }
	///store the min info to the image...
	image_temp->exec_time        = time_min;
	image_temp->kernel_indexs[0] = index_min;
    image_temp->freq             = freq_min;

	image_temp->luts             = luts_min;
	image_temp->regs             = regs_min;
	image_temp->rams             = rams_min;
	image_temp->dsps             = dsps_min;

	image_temp->kernel_valid_num = 1;
	image_temp->reconfig_time    = 0;
 }
 else if (num == 2)
 {
	float time_min                           = 99999999.0;
	uint index_min_0                         = 0;
	uint index_min_1                         = 0;
	uint luts_min, regs_min, rams_min, dsps_min;
	float freq_min;
	uint index_0                             = opertor_seltection[0];
	uint index_1                             = opertor_seltection[1];

    uint offset_0                            = kernel_offset[index_0];
    uint offset_1                            = kernel_offset[index_1];
    uint option_num_0                        = kernel_number[index_0];
    uint option_num_1                        = kernel_number[index_1];


//    struct kernel_info *p_primitive_offset = (struct kernel_info *) kernel_addr[offset_0];
//     struct kernel_info *p_primitive_offset_1 = (struct kernel_info *) kernel_addr[offset_1];

    for (uint i = 0; i < option_num_0; i++)
    {
     for (uint j = 0; j < option_num_1; j++)
	 {
      uint luts_0       =  kernel_primitive[offset_0 + i].lut_c;
      uint regs_0       =  kernel_primitive[offset_0 + i].reg_c;
      uint rams_0       =  kernel_primitive[offset_0 + i].ram_c;
      uint dsps_0       =  kernel_primitive[offset_0 + i].dsp_c;
	  float unit_cost_0 =  kernel_primitive[offset_0 + i].unit_cost;

      uint luts_1       =  kernel_primitive[offset_1 + j].lut_c;
      uint regs_1       =  kernel_primitive[offset_1 + j].reg_c;
      uint rams_1       =  kernel_primitive[offset_1 + j].ram_c;
      uint dsps_1       =  kernel_primitive[offset_1 + j].dsp_c;
	  float unit_cost_1 =  kernel_primitive[offset_1 + j].unit_cost;

	  float freq      = compute_freq(luts_0+luts_1, regs_0+regs_1, rams_0+rams_1, dsps_0+dsps_1); //compute the freq...

	  float time      = ( (unit_cost_0 + unit_cost_1) * (float)data_size )/freq;
	  if (time < time_min)
	  {
	    time_min   = time;
		freq_min   = freq;

		index_min_0= i;
		index_min_1= j;
		luts_min   = luts_0+luts_1;
		regs_min   = regs_0+regs_1;
		rams_min   = rams_0+rams_1;
		dsps_min   = dsps_0+dsps_1;

	  }
     }
    }
	///store the min info to the image...
	image_temp->exec_time        = time_min;
	image_temp->kernel_indexs[0] = index_min_0;
	image_temp->kernel_indexs[1] = index_min_1;

	image_temp->freq             = freq_min;

	image_temp->luts             = luts_min;
	image_temp->regs             = regs_min;
	image_temp->rams             = rams_min;
	image_temp->dsps             = dsps_min;

	image_temp->kernel_valid_num = 2;
	image_temp->reconfig_time    = 0;
 }
 else if (num == 3)
 {
	float time_min                           = 99999999.0;
	uint index_min_0                         = 0;
	uint index_min_1                         = 0;
	uint index_min_2                         = 0;
	uint luts_min, regs_min, rams_min, dsps_min;
	float freq_min;
	uint index_0                             = opertor_seltection[0];
	uint index_1                             = opertor_seltection[1];
	uint index_2                             = opertor_seltection[2];

    uint offset_0                            = kernel_offset[index_0];
    uint offset_1                            = kernel_offset[index_1];
    uint offset_2                            = kernel_offset[index_2];

	uint option_num_0                        = kernel_number[index_0];
    uint option_num_1                        = kernel_number[index_1];
    uint option_num_2                        = kernel_number[index_2];


//    struct kernel_info *p_primitive_offset = (struct kernel_info *) kernel_addr[offset_0];
//     struct kernel_info *p_primitive_offset_1 = (struct kernel_info *) kernel_addr[offset_1];

    for (uint i = 0; i < option_num_0; i++)
    {
     for (uint j = 0; j < option_num_1; j++)
	 {
     for (uint k = 0; k < option_num_2; k++)
	 {
	  uint luts_0       =  kernel_primitive[offset_0 + i].lut_c;
      uint regs_0       =  kernel_primitive[offset_0 + i].reg_c;
      uint rams_0       =  kernel_primitive[offset_0 + i].ram_c;
      uint dsps_0       =  kernel_primitive[offset_0 + i].dsp_c;
	  float unit_cost_0 =  kernel_primitive[offset_0 + i].unit_cost;

      uint luts_1       =  kernel_primitive[offset_1 + j].lut_c;
      uint regs_1       =  kernel_primitive[offset_1 + j].reg_c;
      uint rams_1       =  kernel_primitive[offset_1 + j].ram_c;
      uint dsps_1       =  kernel_primitive[offset_1 + j].dsp_c;
	  float unit_cost_1 =  kernel_primitive[offset_1 + j].unit_cost;

      uint luts_2       =  kernel_primitive[offset_2 + k].lut_c;
      uint regs_2       =  kernel_primitive[offset_2 + k].reg_c;
      uint rams_2       =  kernel_primitive[offset_2 + k].ram_c;
      uint dsps_2       =  kernel_primitive[offset_2 + k].dsp_c;
	  float unit_cost_2 =  kernel_primitive[offset_2 + k].unit_cost;


	  float freq      = compute_freq(luts_0+luts_1+luts_2, regs_0+regs_1+regs_2, rams_0+rams_1+rams_2, dsps_0+dsps_1+dsps_2); //compute the freq...

	  float time      = ( (unit_cost_0 + unit_cost_1 + unit_cost_2) * (float)data_size )/freq;
	  if (time < time_min)
	  {
	    time_min   = time;
		freq_min   = freq;

		index_min_0= i;
		index_min_1= j;
		index_min_2= k;
		luts_min   = luts_0+luts_1+luts_2;
		regs_min   = regs_0+regs_1+regs_2;
		rams_min   = rams_0+rams_1+rams_2;
		dsps_min   = dsps_0+dsps_1+dsps_2;
	  }
     }
    }
	///store the min info to the image...
	image_temp->exec_time        = time_min;
	image_temp->kernel_indexs[0] = index_min_0;
	image_temp->kernel_indexs[1] = index_min_1;
	image_temp->kernel_indexs[2] = index_min_2;

	image_temp->freq             = freq_min;

	image_temp->luts             = luts_min;
	image_temp->regs             = regs_min;
	image_temp->rams             = rams_min;
	image_temp->dsps             = dsps_min;

	image_temp->kernel_valid_num = 3;
	image_temp->reconfig_time    = 0;
  }
 }
 else if (num == 4)
 {
	float time_min                           = 99999999.0;
	uint index_min_0                         = 0;
	uint index_min_1                         = 0;
	uint index_min_2                         = 0;
	uint index_min_3                         = 0;

	uint luts_min, regs_min, rams_min, dsps_min;
	float freq_min;
	uint index_0                             = opertor_seltection[0];
	uint index_1                             = opertor_seltection[1];
	uint index_2                             = opertor_seltection[2];
	uint index_3                             = opertor_seltection[3];

    uint offset_0                            = kernel_offset[index_0];
    uint offset_1                            = kernel_offset[index_1];
    uint offset_2                            = kernel_offset[index_2];
    uint offset_3                            = kernel_offset[index_3];

	uint option_num_0                        = kernel_number[index_0];
    uint option_num_1                        = kernel_number[index_1];
    uint option_num_2                        = kernel_number[index_2];
    uint option_num_3                        = kernel_number[index_3];


//    struct kernel_info *p_primitive_offset = (struct kernel_info *) kernel_addr[offset_0];
//     struct kernel_info *p_primitive_offset_1 = (struct kernel_info *) kernel_addr[offset_1];

    for (uint i = 0; i < option_num_0; i++)
    {
      uint luts_0       =  kernel_primitive[offset_0 + i].lut_c;
      uint regs_0       =  kernel_primitive[offset_0 + i].reg_c;
      uint rams_0       =  kernel_primitive[offset_0 + i].ram_c;
      uint dsps_0       =  kernel_primitive[offset_0 + i].dsp_c;
	  float unit_cost_0 =  kernel_primitive[offset_0 + i].unit_cost;

     for (uint j = 0; j < option_num_1; j++)
	 {
        uint luts_1       =  kernel_primitive[offset_1 + j].lut_c;
        uint regs_1       =  kernel_primitive[offset_1 + j].reg_c;
        uint rams_1       =  kernel_primitive[offset_1 + j].ram_c;
        uint dsps_1       =  kernel_primitive[offset_1 + j].dsp_c;
	    float unit_cost_1 =  kernel_primitive[offset_1 + j].unit_cost;

     for (uint k = 0; k < option_num_2; k++)
	 {
         uint luts_2       =  kernel_primitive[offset_2 + k].lut_c;
         uint regs_2       =  kernel_primitive[offset_2 + k].reg_c;
         uint rams_2       =  kernel_primitive[offset_2 + k].ram_c;
         uint dsps_2       =  kernel_primitive[offset_2 + k].dsp_c;
	     float unit_cost_2 =  kernel_primitive[offset_2 + k].unit_cost;
	  for (uint m = 0; m < option_num_3; m++)
	   {
           uint luts_3       =  kernel_primitive[offset_3 + m].lut_c;
           uint regs_3       =  kernel_primitive[offset_3 + m].reg_c;
           uint rams_3       =  kernel_primitive[offset_3 + m].ram_c;
           uint dsps_3       =  kernel_primitive[offset_3 + m].dsp_c;
           float unit_cost_3 =  kernel_primitive[offset_3 + m].unit_cost;

           float freq      = compute_freq(luts_0+luts_1+luts_2+luts_3, 
		                                  regs_0+regs_1+regs_2+regs_3, 
									      rams_0+rams_1+rams_2+rams_3, 
									      dsps_0+dsps_1+dsps_2+dsps_3
									     ); //compute the freq...

	      float time      = ( (unit_cost_0 + unit_cost_1 + unit_cost_2 + unit_cost_3) * (float)data_size )/freq;
          if (time < time_min)
          {
            time_min   = time;
            freq_min   = freq;

            index_min_0= i;
            index_min_1= j;
            index_min_2= k;
            index_min_3= m;
            luts_min   = luts_0+luts_1+luts_2+luts_3;
            regs_min   = regs_0+regs_1+regs_2+regs_3;
            rams_min   = rams_0+rams_1+rams_2+rams_3;
            dsps_min   = dsps_0+dsps_1+dsps_2+dsps_3;
	     }
       }
     }
	///store the min info to the image...
	image_temp->exec_time        = time_min;
	image_temp->kernel_indexs[0] = index_min_0;
	image_temp->kernel_indexs[1] = index_min_1;
	image_temp->kernel_indexs[2] = index_min_2;
	image_temp->kernel_indexs[3] = index_min_3;

	image_temp->freq             = freq_min;

	image_temp->luts             = luts_min;
	image_temp->regs             = regs_min;
	image_temp->rams             = rams_min;
	image_temp->dsps             = dsps_min;

	image_temp->kernel_valid_num = 4;
	image_temp->reconfig_time    = 0;
   }
  }
 }
 else if (num == 5)
 {
	float time_min                           = 99999999.0;
	uint index_min_0                         = 0;
	uint index_min_1                         = 0;
	uint index_min_2                         = 0;
	uint index_min_3                         = 0;
	uint index_min_4                         = 0;

	uint luts_min, regs_min, rams_min, dsps_min;
	float freq_min;
	uint index_0                             = opertor_seltection[0];
	uint index_1                             = opertor_seltection[1];
	uint index_2                             = opertor_seltection[2];
	uint index_3                             = opertor_seltection[3];
	uint index_4                             = opertor_seltection[4];

    uint offset_0                            = kernel_offset[index_0];
    uint offset_1                            = kernel_offset[index_1];
    uint offset_2                            = kernel_offset[index_2];
    uint offset_3                            = kernel_offset[index_3];
    uint offset_4                            = kernel_offset[index_4];

	uint option_num_0                        = kernel_number[index_0];
    uint option_num_1                        = kernel_number[index_1];
    uint option_num_2                        = kernel_number[index_2];
    uint option_num_3                        = kernel_number[index_3];
    uint option_num_4                        = kernel_number[index_4];

	printf("index_4 = %d\n",                 index_4);
	printf("option_num_4 = %d\n",       option_num_4);
	printf("offset_4 = %d\n",               offset_4);

//    struct kernel_info *p_primitive_offset = (struct kernel_info *) kernel_addr[offset_0];
//     struct kernel_info *p_primitive_offset_1 = (struct kernel_info *) kernel_addr[offset_1];

    for (uint i = 0; i < option_num_0; i++)
    {
      uint luts_0       =  kernel_primitive[offset_0 + i].lut_c;
      uint regs_0       =  kernel_primitive[offset_0 + i].reg_c;
      uint rams_0       =  kernel_primitive[offset_0 + i].ram_c;
      uint dsps_0       =  kernel_primitive[offset_0 + i].dsp_c;
	  float unit_cost_0 =  kernel_primitive[offset_0 + i].unit_cost;

     for (uint j = 0; j < option_num_1; j++)
	 {
        uint luts_1       =  kernel_primitive[offset_1 + j].lut_c;
        uint regs_1       =  kernel_primitive[offset_1 + j].reg_c;
        uint rams_1       =  kernel_primitive[offset_1 + j].ram_c;
        uint dsps_1       =  kernel_primitive[offset_1 + j].dsp_c;
	    float unit_cost_1 =  kernel_primitive[offset_1 + j].unit_cost;

     for (uint k = 0; k < option_num_2; k++)
	 {
         uint luts_2       =  kernel_primitive[offset_2 + k].lut_c;
         uint regs_2       =  kernel_primitive[offset_2 + k].reg_c;
         uint rams_2       =  kernel_primitive[offset_2 + k].ram_c;
         uint dsps_2       =  kernel_primitive[offset_2 + k].dsp_c;
	     float unit_cost_2 =  kernel_primitive[offset_2 + k].unit_cost;
	  for (uint m = 0; m < option_num_3; m++)
	   {
           uint luts_3       =  kernel_primitive[offset_3 + m].lut_c;
           uint regs_3       =  kernel_primitive[offset_3 + m].reg_c;
           uint rams_3       =  kernel_primitive[offset_3 + m].ram_c;
           uint dsps_3       =  kernel_primitive[offset_3 + m].dsp_c;
           float unit_cost_3 =  kernel_primitive[offset_3 + m].unit_cost;
	    for (uint n = 0; n < option_num_4; n++)
	     {
              uint luts_4       =  kernel_primitive[offset_4 + n].lut_c;
              uint regs_4       =  kernel_primitive[offset_4 + n].reg_c;
              uint rams_4       =  kernel_primitive[offset_4 + n].ram_c;
              uint dsps_4       =  kernel_primitive[offset_4 + n].dsp_c;
              float unit_cost_4 =  kernel_primitive[offset_4 + n].unit_cost;



           float freq      = compute_freq(luts_0+luts_1+luts_2+luts_3+luts_4, 
		                                  regs_0+regs_1+regs_2+regs_3+regs_4, 
									      rams_0+rams_1+rams_2+rams_3+rams_4, 
									      dsps_0+dsps_1+dsps_2+dsps_3+dsps_4
									     ); //compute the freq...

	      float time      = ( (unit_cost_0 + unit_cost_1 + unit_cost_2 + unit_cost_3 + unit_cost_4) * (float)data_size )/freq;
          if (time < time_min)
          {
            time_min   = time;
            freq_min   = freq;

            index_min_0= i;
            index_min_1= j;
            index_min_2= k;
            index_min_3= m;
            index_min_4= n;

            luts_min   = luts_0+luts_1+luts_2+luts_3+luts_4;
            regs_min   = regs_0+regs_1+regs_2+regs_3+regs_4;
            rams_min   = rams_0+rams_1+rams_2+rams_3+rams_4;
            dsps_min   = dsps_0+dsps_1+dsps_2+dsps_3+dsps_4;
	     }
       }
     }
   }
  }
 }
	 ///store the min info to the image...
	image_temp->exec_time        = time_min;
	image_temp->kernel_indexs[0] = index_min_0;
	image_temp->kernel_indexs[1] = index_min_1;
	image_temp->kernel_indexs[2] = index_min_2;
	image_temp->kernel_indexs[3] = index_min_3;
	image_temp->kernel_indexs[4] = index_min_4;

	image_temp->freq             = freq_min;

	image_temp->luts             = luts_min;
	image_temp->regs             = regs_min;
	image_temp->rams             = rams_min;
	image_temp->dsps             = dsps_min;

	image_temp->kernel_valid_num = 5;
	image_temp->reconfig_time    = 0;

 }
}

void print_image_info(image_info *image_temp)
{
//	image_temp->freq             = freq_min;
	printf("luts, regs, rams, dsps = %d, %d, %d, %d\n",image_temp->luts, image_temp->regs, image_temp->rams, image_temp->dsps);
	printf("total_unit_cost = %f, exec_time = %f\n", image_temp->total_unit_cost, image_temp->exec_time); //image_temp->exec_time        = time_min;
	printf("freq = %f\n", image_temp->freq); //image_temp->freq        = freq_min;
	printf("mem_inc = %f, reconfig_time = %f\n", image_temp->mem_inc, image_temp->reconfig_time); //image_temp->freq        = freq_min;

	printf("id: kernel_id, sub_id,                name\n");
	for (uint i=0; i < image_temp->kernel_valid_num; i++)
	{
		//printf("image_temp->kernel_indexs[%d] = %d\n", i, image_temp->kernel_indexs[i]); //image_temp->kernel_indexs[0] = index_min;
		printf("%d:      %d,        %d      %s\n", i, image_temp->kernel_id[i], image_temp->kernel_indexs[i], 
			                               kernel_primitive[kernel_offset[image_temp->kernel_id[i]] + image_temp->kernel_indexs[i] ].names); //image_temp->kernel_indexs[0] = index_min;
	
	}

	//image_temp->kernel_valid_num = 2;
	//image_temp->reconfig_time    = 0;

}

void print_exec_time_info(struct S_Min_Exec_Time *p_selection_exec_time)
{   
	uint image_number = p_selection_exec_time->image_num;
	printf("The optimum task scheduling: \n total execution time: %f\n", p_selection_exec_time->exec_time); 
	printf("image_num = %d\n\n", p_selection_exec_time->image_num); 
	//printf("id: kernel_id, sub_id\n");
	for (uint i=0; i < image_number; i++)
	{   
		printf("The %d-th FPGA image:\n", i);
		print_image_info(&p_selection_exec_time->inst_image[i]);
		printf("\n");
		//printf("image_temp->kernel_indexs[%d] = %d\n", i, image_temp->kernel_indexs[i]); //image_temp->kernel_indexs[0] = index_min;
		//printf("%d:      %d,        %d\n", i, opertor_seltection[i], image_temp->kernel_indexs[i]); //image_temp->kernel_indexs[0] = index_min;
	}

}


void dp_for_selection(uint *opertor_seltection, uint number_kernel)//, struct conf_k*conf_selection
{
  uint i;
///////////////////////for memory footprint overhead////////////////////////
  float mem_footprint[20]         = {0.0};
  float mem_reconfig_overhead[20] = {0.0};
  mem_footprint[0] = 1.0;
///////////////////////////////////////////////////////////
  struct S_Min_Exec_Time selection_exec_time[20] = {0};

  selection_exec_time[0].image_num = 0;  //initialization of first struct to 0. 
  selection_exec_time[0].exec_time = 0;  //////////////////////////////

  struct image_info *p_image_temp, *p_image_result;

  p_image_temp   = (struct image_info *) malloc(sizeof(struct image_info));

  //for storing the best FPGA image configuration.
  p_image_result = (struct image_info *) malloc(sizeof(struct image_info));
  uint break_index;
//  one_image_min_exec_time(opertor_seltection, number_kernel, p_image_temp);
//  print_image_info(p_image_temp);

 // printf("\nsrecursive result: \n");
 // recursive_time_compute(opertor_seltection, 0, 0, 0, 0, 0, 0, number_kernel, p_image_temp);
 // one_image_min_exec_time(opertor_seltection, number_kernel, p_image_temp);
 // print_image_info(p_image_temp);

  	  /*consider the input kernel one by one*/
  for (i = 1; i <= number_kernel; i++) // compute one selection_exec_time[i-1] for each loop
  {
	uint new_id          = i - 1;
	uint i_kernel_id     = opertor_seltection[new_id];
	uint i_kernel_offset = kernel_offset[i_kernel_id];

   mem_reconfig_overhead[i] = mem_footprint[i-1] + kernel_primitive[i_kernel_offset].unit_mem_add;
   mem_footprint[i]         = mem_reconfig_overhead[i] + kernel_primitive[i_kernel_offset].unit_mem_sub;

    //selection_exec_time[i].exec_time = 0x8fffffff; // max value for execution time
    //selection_exec_time[i].image_num = 0; 
	float time_min = 99999999.9;

    for (uint j = 0; j < i; j++)    //find the fastest implementation for the new_id-th kernel....
	{
		float optimum_sub_exec_time = selection_exec_time[j].exec_time;//selection_exec_time[i] 

		////find the replicated ones.
		uint *p_operator_tmp;
		p_operator_tmp  = opertor_seltection+j;

		uint overall_check_len = i-j;
		uint check_list[100] = {0};
		uint check_len[100]  = {0};
		uint unique_kernel_sz = 0;
		for (uint kk = 0; kk < overall_check_len; kk++)
		{
			int matched = 0;
			int matched_id;
		  for (uint kkk = 0; kkk < kk; kkk++)
		  {
		    if (check_list[kkk] == p_operator_tmp[kk])
			{
			   matched = 1;
			   matched_id = kkk;
			}
		  }

          if (matched == 1)
				check_len[matched_id]++;
		  else 
		  {
		    check_list[unique_kernel_sz] = p_operator_tmp[kk];
		    check_len[unique_kernel_sz++]  = 1;
		  }
		}

	    float new_image_exec_time   = recursive_time_compute(check_list, check_len, 0, 0, 0, 0, 0.0, 0, unique_kernel_sz, p_image_temp);
		//recursive_time_compute(opertor_seltection+j, 0, 0, 0, 0, 0.0, 0, i-j, p_image_temp);// resource initialized to 0....

		float reconfig_time = 0.0;
		if (j != 0)
			reconfig_time = (0.993* data_size* 4.0 * mem_reconfig_overhead[j+1] + 1914.6)/1000.0; //64.0

		if ( (optimum_sub_exec_time + new_image_exec_time + reconfig_time) < time_min) //new candicate.
		{
		  p_image_temp->mem_inc       = mem_reconfig_overhead[j+1];
		  p_image_temp->reconfig_time = reconfig_time;
		  break_index = j;
		  memcpy( (void *)p_image_result, (void *)p_image_temp, sizeof(struct image_info));
		  time_min    = optimum_sub_exec_time + new_image_exec_time + reconfig_time;
		}
	}
    selection_exec_time[i].exec_time    = time_min;

    uint pre_image_num = selection_exec_time[break_index].image_num;

	selection_exec_time[i].image_num    = pre_image_num + 1;

	for (uint k = 0; k < pre_image_num; k++)
		  memcpy( (void *)&(selection_exec_time[i].inst_image[k]), (void *)&(selection_exec_time[break_index].inst_image[k]), sizeof(struct image_info));
 
	memcpy( (void *)&(selection_exec_time[i].inst_image[pre_image_num]), (void *)p_image_result, sizeof(struct image_info));

  }
/*
  print_exec_time_info(&selection_exec_time[1]);

  printf("\nsecond sub-optimum structure:\n");
  print_exec_time_info(&selection_exec_time[2]);

  printf("\nthird sub-optimum structure:\n");
  print_exec_time_info(&selection_exec_time[3]);
*/

  printf("\noptimum structure:\n");
  print_exec_time_info(&selection_exec_time[number_kernel]);

  for (uint ii = 0; ii < number_kernel; ii++)
  {
    printf( "mem_footprint[%d] = %f, mem_reconfig_overhead[%d] = %f\n",ii, mem_footprint[ii],ii, mem_reconfig_overhead[ii]);
  }
  //float mem_reconfig_overhead[8] = {0.0};  }
} 


//extern int dp_for_selection(uint *opertor_seltection, uint number_kernel);//, struct conf_k*conf_selection
void dp_test(int sub_index, float size)
{
  data_size = size * 1.0;//4.0;
  if (sub_index == 0)
  { 
	printf("task scheduling for selection, data size = %f (M)\n", size);
    dp_for_selection(opertor_seltection, 5);
  }
  else if (sub_index == 1)
  {
      printf("task scheduling for aggregation, data size = %f (M)\n", size);
	  dp_for_selection(opertor_group_agg, 7);
  }
  else if (sub_index == 2)
  {
      printf("task scheduling for filter+aggregation, data size = %f (M)\n", size);
	  dp_for_selection(opertor_filter_group_agg, 12);
  }
  else if (sub_index == 3)
  {
      printf("task scheduling for pre-scan, data size = %f (M)\n", size);
	  dp_for_selection(opertor_seltection+1, 3);
  }
    return;
}