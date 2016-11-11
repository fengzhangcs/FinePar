#27 is memory unit busy
#28 is memory unit stalled
#21 VALUUtilization
#7 time
#22 valubusy
#23 salubusy
#26 cache hit 
#8 local mem
#6 local group size
#5 global threads size
#13 wavefront
#9 VGPRs
#10 SGPRs
#24 fetch size
num=7
a=`ls *.csv`
for i in $a
do
  name=`echo $i|sed 's/.csv//g'`
  item=`sed -n 24p $i | awk -F, '{print $'$num'}'`
  echo $name","$item

done


echo "only number:"
for i in $a
do
  name=`echo $i|sed 's/.csv//g'`
  item=`sed -n 24p $i | awk -F, '{print $'$num'}'`
  echo $item

done
