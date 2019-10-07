
aoc --fpc --fp-relaxed -D NUM_CU_SCAN=8 -D NUM_CU_SCAN_LARGE=2 -D NUM_CU_ADD=1 -D NUM_SIMD_ADD=16 -D NUM_CU_WRITE=8 -D NUM_CU_AGG=1 -v group_by_cu_op_dp_2.cl -o group_by_cu_op_dp_2_8_2_1-16_8_1.aocx --report


aoc --fpc --fp-relaxed -D NUM_CU_SCAN=8 -D NUM_CU_SCAN_LARGE=2 -D NUM_CU_ADD=1 -D NUM_SIMD_ADD=16 -D NUM_CU_WRITE=8 -D NUM_CU_AGG=1 -v group_by_cu_op_dp_2.cl -o group_by_cu_op_dp_2_8_2_1-16_8_1_dummy.aocx --report
