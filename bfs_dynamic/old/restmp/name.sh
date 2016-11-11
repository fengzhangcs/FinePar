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
echo "gflops:"
for i in $a
do
  name=`echo $i|sed 's/.txt//g'`
  gflopsout=`cat $i | grep "CSR VEC without padding best time"|awk '{print $13}'`
  gflopswith=`cat $i | grep "CSR VEC with padding best time"|awk '{print $13}'`
  nItems=`cat $i | grep "MatInfo:"|awk '{print $7}'`
  time=`cat $i | grep "CSR VEC without padding best time"|awk '{print $7}'|sed 's/s//g'`
  if [ $(echo "$gflopsout >= $gflopswith"|bc) = 1 ]
  then 
    echo $gflopsout 
  else
    echo $gflopswith
  fi
done

echo
echo
echo "time:"
for i in $a
do
  name=`echo $i|sed 's/.txt//g'`
  gflopsout=`cat $i | grep "CSR VEC without padding best time"|awk '{print $13}'`
  gflopswith=`cat $i | grep "CSR VEC with padding best time"|awk '{print $13}'`
  nItems=`cat $i | grep "MatInfo:"|awk '{print $7}'`
  timeout=`cat $i | grep "CSR VEC without padding best time"|awk '{print $7}'|sed 's/s//g'`
  timewith=`cat $i | grep "CSR VEC with padding best time"|awk '{print $7}'|sed 's/s//g'`
  if [ $(echo "$timeout >= $timewith"|bc) = 1 ]
  then 
    echo $timewith 
  else
    echo $timeout
  fi
done

