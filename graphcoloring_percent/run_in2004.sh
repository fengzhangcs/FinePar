 make clean; make
 sleep 3
 for i in 0 10 20 30 40 50 60 70 80 90 100
 do
 ./spmv ../input/in-2004/in-2004.mtx  $i
 done
