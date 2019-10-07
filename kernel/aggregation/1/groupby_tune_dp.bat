
::aoc --fpc --fp-relaxed -D NUM_CU_S=4 -D NUM_CU_G=2  -D NUM_CU_SCAN=1 -D NUM_CU_SCAN_LARGE=1 -D NUM_CU_ADD=1 -D NUM_SIMD_ADD=16 -D NUM_CU_WRITE=1 -D NUM_CU_AGG=1 -v group_by_cu_op_dp.cl -o group_by_cu_op_dp_4_2_1_1_1-16_1_1.aocx --report

aoc --fpc --fp-relaxed -D NUM_CU_S=1 -D NUM_CU_G=1  -D NUM_CU_SCAN=1 -D NUM_CU_SCAN_LARGE=1 -D NUM_CU_ADD=1 -D NUM_SIMD_ADD=1 -D NUM_CU_WRITE=1 -D NUM_CU_AGG=1 -v group_by_cu_op_dp.cl -o group_by_cu_op_dp_1_1_1_1_1-1_1_1.aocx --report
