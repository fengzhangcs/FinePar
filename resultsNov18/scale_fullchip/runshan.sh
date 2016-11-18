#PROGRAMDIR=$1
PROGRAMDIR="/home/pacman/FinePar/"




BASHNAME="run_fullchip.sh"
RESULTDIR="${PROGRAMDIR}/results/scale_fullchip/percent/"

mkdir -p $RESULTDIR

cd $RESULTDIR
mkdir -p bfs_percent 


cd $PROGRAMDIR/bfs_percent/
bash $BASHNAME | tee $RESULTDIR/bfs_percent/res.txt

