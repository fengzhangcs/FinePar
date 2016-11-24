mkdir -p generatedCode
ninja; ./bin/fineParP2S codePercent/spmv_csr.cpp -- -I/Users/zhangfeng/zf/llvm/build/zf/codeTrans/AMDAPP/include
ninja; ./bin/fineParP2Scl codePercent/spmv_csr_vector.cl -- -I/Users/zhangfeng/zf/llvm/build/zf/codeTrans/AMDAPP/include
