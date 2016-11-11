#make
mkdir -p res
a=`ls ~/zf/spmv/2m*/`
programdir="/home/pacman/zf/spmv/2matrix_mediumsize/"

#a=`ls ~/zf/spmv/1*/`
#programdir="/home/pacman/zf/spmv/1matrix_small/"
resdir="/home/pacman/zf/wubo/apu_corun/zfcsr/res32/"
resdir64="/home/pacman/zf/wubo/apu_corun/zfcsr/res64/"
#resdir="/home/pacman/zf/wubo/apu_corun/zfcsr/res/"
for i in $a
do
  echo "This is $i"
  name=`echo $i|sed 's/.mtx//g'`
  #echo $name
  ./spmv32 $programdir$i | tee  $resdir/$name.txt
  sleep 5
done
  sleep 15

for i in $a
do
  echo "This is $i"
  name=`echo $i|sed 's/.mtx//g'`
  #echo $name
  ./spmv64 $programdir$i | tee  $resdir64/$name.txt
  sleep 5
done
