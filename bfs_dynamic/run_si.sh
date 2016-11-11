make clean;make

for i in 0 10 20 30 40 50 60 70 80 90 100
do

./bfs ~/zf/spmv/2graph_mediumsize/Si41Ge41H72.txt  $i
sleep 3
done


