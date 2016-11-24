 make clean; make
 for i in 0 10 20 30 40 50 60 70 80 90 100
 do
 sleep 3
 ./spmv ~/FinePar/input/scale20/matrxtest.txt  $i
 done
