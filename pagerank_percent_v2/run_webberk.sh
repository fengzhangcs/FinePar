make clean;make

for i in 0 10 20 30 40 50 60 70 80 90 100
do

./spmv ~/zf/spmv/4download/webberk/web.txt $i
sleep 3
done


