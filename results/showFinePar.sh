
DIR=`pwd`
item=`ls |grep scale`
#echo $DIR
#echo $item
cd $DIR
for i in $item
do
  echo $i
  echo "FinePar,    BFS,    ConnectedComponent,    GraphColoring,    HITS_v2,    PageRank_v2,    SPMVCSR_v2,    SPMVELL,"
  for j in 32 64 128 256 512 1024 2048
  do
  cd ${DIR}/${i}/*$j

keyword="CAUT"

cd bfs
res_bfs=(`grep $keyword res.txt| awk '{print $4}'`)

cd ../connected
res_cc=(`grep $keyword res.txt| awk '{print $4}'`)

cd ../graphcolor
res_gc=(`grep $keyword res.txt| awk '{print $4}'`)

cd ../hits_v2
res_hits_v2=(`grep $keyword res.txt| awk '{print $4}'`)

cd ../pagerank_v2
res_page_v2=(`grep $keyword res.txt| awk '{print $4}'`)

cd ../spmvcsr_v2
res_csr_v2=(`grep $keyword res.txt| awk '{print $4}'`)

cd ../spmvell
res_ell=(`grep $keyword res.txt| awk '{print $4}'`)

space=",        "
echo -n $j $space $res_bfs $space  $res_cc $space  $res_gc $space    $res_hits_v2 $space  $res_page_v2 $space  $res_csr_v2 $space  $res_ell
echo
done



done
