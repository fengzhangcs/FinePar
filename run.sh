PROGRAMDIR="/home/pacman/FinePar/"

rm -rf results
cp -r oriRes/ results
for i in scale20 scale_circuit5M scale_eu2005 scale_fullchip scale_in2004 scale_webberk
do
  cd ${PROGRAMDIR}/results/$i/
  bash run_dynamic.sh ${PROGRAMDIR}
  bash run_percent.sh ${PROGRAMDIR}
  bash run_separate.sh ${PROGRAMDIR}
done 
