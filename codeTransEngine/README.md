This is a prototype of FinePar source to source transformation.
This is useful to Clang libtooling developers.

fineParP2S is to transform c++ code.

fineParP2Scl is to transform OpenCL code.

1. Put these 2 files under "llvm/tools/clang/tools/extra" and add the follows 2 lines to "llvm/tools/clang/tools/extra/CMakeLists.txt".

add_subdirectory(fineParP2S)

add_subdirectory(fineParP2Scl)

2. Go to the "build" directory of LLVM and run "ninja" for compilition.

3. Please read the "run.sh". Change the related files such as the AMDAPP include.
Then run the script by "bash run.sh".

4. Under the "generatedCode" directory, there are the code files for the c++ and OpenCL programs.
After copy other related needed files to this directory, execute the "make" and "bash run_scale20.sh" will run the new FinePar transformed program.
