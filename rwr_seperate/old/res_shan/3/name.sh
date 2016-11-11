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
echo"ahahah" 
for i in $a
do
  name=`echo $i|sed 's/.txt//g'`
  gflops=`cat $i | grep "CSR VEC without padding best time"|awk '{print $13}'`
  nItems=`cat $i | grep "MatInfo:"|awk '{print $7}'`
  time=`cat $i | grep "CSR VEC without padding best time"|awk '{print $7}'|sed 's/s//g'`
  echo $gflops
done



echo
echo
echo "name:"
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

