main: *.cpp *.h
	g++  -O3 -g  -w  -c spmv_csr.cpp  -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64 -lOpenCL -lm -lpthread
	g++  -O3  -g  -w  -c spmv_util.cpp  -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64 -lOpenCL -lm 
	g++ -O3  -g -w  spmv_util.o spmv_csr.o -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64 -lOpenCL -lm -o spmv -lpthread
	rm -f *.o
clean:
	rm -f spmv main *.o
