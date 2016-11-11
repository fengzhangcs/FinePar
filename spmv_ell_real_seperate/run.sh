make
mkdir -p res
a=`ls ~/zf/spmv/1*/`
programdir="/home/pacman/zf/spmv/1matrix_small/"
tmp=`pwd`
resdir=$tmp/res/
for i in $a
do
  echo "This is $i"
  name=`echo $i|sed 's/.mtx//g'`
  #echo $name
  ./spmv $programdir$i | tee  $resdir/$name.txt
done
