
DIR=`pwd`
item=`ls |grep scale`
cd $DIR
for i in $item
do
  echo $i
  cd ${DIR}/${i}/dynamic


cd bfs_dynamic
res_bfs_dynamic=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../connected_dynamic
res_cc_dynamic=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../graphcolor_dynamic
res_gc_dynamic=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../spmvcsr_dynamic
res_csr_dynamic=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../spmvell_dynamic
res_ell_dynamic=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../hits_dynamic
res_hits_dynamic=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../pagerank_dynamic
res_page_dynamic=(`grep CAUT res.txt| awk '{print $4}'`)



names=(dynamic BFS ConnectedComponent GraphColoring HITS PageRank SPMVCSR SPMVELL)
dynamic=(time 0 10 20 30 40 50 60 70 80 90 100)

for i in ${names[@]}
do
  echo -n "$i,    "
done
echo 

#for i in {0..10}
#do
  for j in dynamic res_bfs_dynamic res_cc_dynamic res_gc_dynamic res_hits_dynamic res_page_dynamic  res_csr_dynamic res_ell_dynamic
  do
    eval tmp=\${$j[$i]}
    echo -n "$tmp,    "
  done
  echo
#done
echo

done
