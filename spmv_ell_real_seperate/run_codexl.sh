make
mkdir -p res_codexl
a=`ls ~/zf/spmv/1*/`
programdir="/home/pacman/zf/spmv/1matrix_small/"
codedir=`pwd`
#codedir="/home/pacman/zf/shoc/shoc_seperate/spmv_ell_real/"
resdir=$codedir/res_codexl/
for i in $a
do
  echo "This is $i"
  name=`echo $i|sed 's/.mtx//g'`
  #echo $name
/opt/AMD/CodeXL_1.8-9654/x86_64/sprofile -o "$resdir/$name.csv" -p -w "$codedir/" "$codedir/spmv" $programdir$i 
done
