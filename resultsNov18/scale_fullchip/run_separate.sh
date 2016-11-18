PROGRAMDIR=$1
#PROGRAMDIR="/home/pacman/FinePar"


bfsdata="FullChip/FullChip.txt"
generaldata="FullChip/FullChip.mtx"

for cpuoffset  in 32 64 128 256 512 1024 2048
do

RESULTDIR="${PROGRAMDIR}/results/scale_fullchip/fullchip_separate${cpuoffset}/"
mkdir -p $RESULTDIR
cd $RESULTDIR
mkdir -p bfs connected graphcolor spmvell spmvcsr_v2  hits_v2 pagerank_v2

cd $PROGRAMDIR/bfs_separate/
make
./bfs  ${PROGRAMDIR}/input/${bfsdata}  $cpuoffset | tee $RESULTDIR/bfs/res.txt

cd $PROGRAMDIR/connectedComp_separate/
make
./spmv ${PROGRAMDIR}/input/${generaldata}  $cpuoffset | tee $RESULTDIR/connected/res.txt

cd $PROGRAMDIR/graphcoloring_separate/
make
./spmv ${PROGRAMDIR}/input/${generaldata} $cpuoffset | tee $RESULTDIR/graphcolor/res.txt

cd $PROGRAMDIR/spmv_ell_real_separate/
make
./spmv ${PROGRAMDIR}/input/${generaldata} $cpuoffset | tee $RESULTDIR/spmvell/res.txt

cd $PROGRAMDIR/zfcsr_separate_v2/
make
./spmv ${PROGRAMDIR}/input/${generaldata} $cpuoffset | tee $RESULTDIR/spmvcsr_v2/res.txt

cd $PROGRAMDIR/hits_separate_v2/
make
./spmv  ${PROGRAMDIR}/input/${generaldata}  $cpuoffset | tee $RESULTDIR/hits_v2/res.txt

cd $PROGRAMDIR/pagerank_separate_v2/
make
./spmv ${PROGRAMDIR}/input/${generaldata} $cpuoffset | tee $RESULTDIR/pagerank_v2/res.txt

done

