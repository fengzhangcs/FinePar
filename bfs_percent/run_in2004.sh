make clean;make
for i in 0 10 20 30 40 50 60 70 80 90 100
do
sleep 3
./bfs ../input/in-2004/in-2004.txt $i

done
