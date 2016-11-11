main: *.cpp *.h
	g++  -O3 -g  -w  -c spmv_csr.cpp  -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64 -lOpenCL -lm 
	g++  -O3 -g  -w  -c spmv_csr_cpu.cpp  -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64 -lOpenCL -lm 
	g++  -O3  -g  -w  -c spmv_util.cpp  -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64 -lOpenCL -lm 
	g++ -O3  -g -w  spmv_util.o spmv_csr.o -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64 -lOpenCL -lm -o spmv
	g++ -O3  -g -w  spmv_util.o spmv_csr_cpu.o -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64 -lOpenCL -lm -o spmv_cpu
	rm -f *.o
clean:
	rm -f spmv spmv_cpu main *.o
