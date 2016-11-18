#cd /home/pacman/FineParBakup/results/scale20/percent

DIR=`pwd`
item=`ls |grep scale`
#echo $DIR
#echo $item
cd $DIR
for i in $item
do
  echo $i
  cd ${DIR}/${i}/percent


cd bfs_percent
res_bfs_percent=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../connected_percent
res_cc_percent=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../graphcolor_percent
res_gc_percent=(`grep CAUT res.txt| awk '{print $4}'`)

#cd ../hits_percent
#res_hits_percent=(`grep CAUT res.txt| awk '{print $4}'`)

#cd ../pagerank_percent
#res_page_percent=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../spmvcsr_percent_v2
res_csr_percent_v2=(`grep CAUT res.txt| awk '{print $4}'`)


#cd ../spmvcsr_percent
#res_csr_percent=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../spmvell_percent
res_ell_percent=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../hits_percent_v2
res_hits_percent_v2=(`grep CAUT res.txt| awk '{print $4}'`)

cd ../pagerank_percent_v2
res_page_percent_v2=(`grep CAUT res.txt| awk '{print $4}'`)



names=(percentage BFS ConnectedComponent GraphColoring HITS_v2 PageRank_v2 SPMVCSR_v2 SPMVELL)
percent=(0 10 20 30 40 50 60 70 80 90 100)

for i in ${names[@]}
do
  echo -n "$i,    "
done
echo 

for i in {0..10}
do
  for j in percent res_bfs_percent res_cc_percent res_gc_percent res_hits_percent_v2 res_page_percent_v2  res_csr_percent_v2 res_ell_percent
  do
    eval tmp=\${$j[$i]}
    echo -n "$tmp,    "
  done
  echo
done
echo

done
