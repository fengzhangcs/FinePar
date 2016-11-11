a=`ls *.txt`
echo "Name,NNZ,GFlops,Time"
for i in $a
do
  name=`echo $i|sed 's/.txt//g'`
  gflops=`cat $i | grep "timeinms:"|awk '{print $6}'`
  nItems=`cat $i | grep "timeinms:"|awk '{print $4}'`
  time=`cat $i | grep "timeinms:"|awk '{print $10}'`
  echo $name","$nItems","$gflops","$time
done

echo
echo "GFLOPS"
for i in $a
do
  name=`echo $i|sed 's/.txt//g'`
  gflops=`cat $i | grep "timeinms:"|awk '{print $6}'`
  nItems=`cat $i | grep "timeinms:"|awk '{print $4}'`
  time=`cat $i | grep "timeinms:"|awk '{print $10}'`
  echo $gflops
done

echo
echo "TIME"
for i in $a
do
  name=`echo $i|sed 's/.txt//g'`
  gflops=`cat $i | grep "timeinms:"|awk '{print $6}'`
  nItems=`cat $i | grep "timeinms:"|awk '{print $4}'`
  time=`cat $i | grep "timeinms:"|awk '{print $10}'`
  echo $time
done
