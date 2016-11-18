PROGRAMDIR=$1
#PROGRAMDIR="/home/pacman/FinePar/"




BASHNAME="run_circuit5M.sh"
RESULTDIR="${PROGRAMDIR}/results/scale_circuit5M/dynamic/"

mkdir -p $RESULTDIR

cd $RESULTDIR
mkdir -p bfs_dynamic graphcolor_dynamic pagerank_dynamic hits_dynamic connected_dynamic spmvell_dynamic spmvcsr_dynamic rwr_dynamic    


cd $PROGRAMDIR/connectedComp_dynamic/
bash $BASHNAME | tee $RESULTDIR/connected_dynamic/res.txt

cd $PROGRAMDIR/graphcoloring_dynamic/
bash $BASHNAME | tee $RESULTDIR/graphcolor_dynamic/res.txt

cd $PROGRAMDIR/bfs_dynamic/
bash $BASHNAME | tee $RESULTDIR/bfs_dynamic/res.txt

cd $PROGRAMDIR/hits_dynamic/
bash $BASHNAME | tee $RESULTDIR/hits_dynamic/res.txt

cd $PROGRAMDIR/pagerank_dynamic/
bash $BASHNAME | tee $RESULTDIR/pagerank_dynamic/res.txt

cd $PROGRAMDIR/spmv_ell_real_dynamic/
bash $BASHNAME | tee $RESULTDIR/spmvell_dynamic/res.txt

cd $PROGRAMDIR/zfcsr_dynamic/
bash $BASHNAME | tee $RESULTDIR/spmvcsr_dynamic/res.txt

cd $PROGRAMDIR/rwr_dynamic/
bash $BASHNAME | tee $RESULTDIR/rwr_dynamic/res.txt

