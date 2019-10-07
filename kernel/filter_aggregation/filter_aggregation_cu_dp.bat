
::aoc --fpc --fp-relaxed -D NUM_CU_MAP=1 -D NUM_CU_FILTER=1 -D NUM_CU_S=2 -D NUM_CU_G=1  -D NUM_CU_SCAN=1 -D NUM_CU_SCAN_LARGE=2 -D NUM_CU_ADD=1 -D NUM_SIMD_ADD=16 -D NUM_CU_WRITE=1 -D NUM_CU_AGG=1 -v filter_aggregation_cu_op_dp.cl -o filter_aggregation_cu_op_dp_1_1_2_1_1_2_1-16_1_1.aocx --report


aoc --fpc --fp-relaxed -D NUM_CU_MAP=1 -D NUM_CU_FILTER=1 -D NUM_CU_S=1 -D NUM_CU_G=1  -D NUM_CU_SCAN=1 -D NUM_CU_SCAN_LARGE=1 -D NUM_CU_ADD=1 -D NUM_SIMD_ADD=1 -D NUM_CU_WRITE=1 -D NUM_CU_AGG=1 -v filter_aggregation_cu_op_dp.cl -o filter_aggregation_cu_op_dp_1_1_1_1_1_1_1-1_1_1.aocx --report
