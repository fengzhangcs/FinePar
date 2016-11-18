PROGRAMDIR=$1
#PROGRAMDIR="/home/pacman/FinePar/"




BASHNAME="run_webberk.sh"
RESULTDIR="${PROGRAMDIR}/results/scale_webberk/percent/"

mkdir -p $RESULTDIR

cd $RESULTDIR
mkdir -p bfs_percent connected_percent graphcolor_percent spmvell_percent spmvcsr_percent_v2  hits_percent_v2 pagerank_percent_v2 rwr_percent



cd $PROGRAMDIR/connectedComp_percent/
bash $BASHNAME | tee $RESULTDIR/connected_percent/res.txt

cd $PROGRAMDIR/graphcoloring_percent/
bash $BASHNAME | tee $RESULTDIR/graphcolor_percent/res.txt

cd $PROGRAMDIR/bfs_percent/
bash $BASHNAME | tee $RESULTDIR/bfs_percent/res.txt

cd $PROGRAMDIR/spmv_ell_real_percent/
bash $BASHNAME | tee $RESULTDIR/spmvell_percent/res.txt

cd $PROGRAMDIR/zfcsr_percent_v2/
bash $BASHNAME | tee $RESULTDIR/spmvcsr_percent_v2/res.txt

cd $PROGRAMDIR/hits_percent_v2/
bash $BASHNAME | tee $RESULTDIR/hits_percent_v2/res.txt

cd $PROGRAMDIR/pagerank_percent_v2/
bash $BASHNAME | tee $RESULTDIR/pagerank_percent_v2/res.txt

cd $PROGRAMDIR/rwr_percent/
bash $BASHNAME | tee $RESULTDIR/rwr_percent/res.txt


