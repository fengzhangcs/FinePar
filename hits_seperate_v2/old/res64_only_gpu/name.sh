a=`ls *.txt`
echo "Name,NNZ,GFlops,Time"
for i in $a
do
  name=`echo $i|sed 's/.txt//g'`
  gflops=`cat $i | grep "CSR VEC without padding best time"|awk '{print $13}'`
  nItems=`cat $i | grep "MatInfo:"|awk '{print $7}'`
  time=`cat $i | grep "CSR VEC without padding best time"|awk '{print $7}'|sed 's/s//g'`
  echo $name","$nItems","$gflops","$time
done

echo
echo
echo
for i in $a
do
  name=`echo $i|sed 's/.txt//g'`
  gflops=`cat $i | grep "CSR VEC without padding best time"|awk '{print $13}'`
  nItems=`cat $i | grep "MatInfo:"|awk '{print $7}'`
  time=`cat $i | grep "CSR VEC without padding best time"|awk '{print $7}'|sed 's/s//g'`
  echo $gflops
done


