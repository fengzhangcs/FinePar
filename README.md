FinePar: Irregularity-Aware Fine-Grained Workload Partitioning on Integrated Architectures

## FinePar ##
FinePar is a method for fine-grained workload partitioning for irregular applications on integrated architectures.

FinePar is tested on 8 programs.
We select 5 programs from the  [GraphBIG benchmark suite](https://github.com/graphbig/graphBIG), the [Rodinia benchmark suite](http://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Rodinia:Accelerating_Compute-Intensive_Applications_with_Accelerators), and the [SHOC benchmark suite](https://github.com/vetter/shoc). Since AMD’s current integrated architectures do not support atomic operations between the CPU and GPU, we only choose programs without using atomic operations. We also implement 3 well-known algorithms in OpenCL, which brings the total number of evaluated benchmarks to 8.

##Platform##
Currently, FinePar is only tested on the following platform, but it is expected to work on other APUs.
* AMD A10-7850K

##Guide##
1. After you download the directory, you need to unzip the input data in the input directory. Execute "cd input; tar zxvf input.tar.gz".
2. We assume the OpenCL is installed in /opt/AMDAPP/. If this is not the directory of OpenCL library, please change the Makefile in each program's directory.
3. Change the first line of "run.sh". For example, PROGRAMDIR="/home/pacman/FinePar/". PROGRAMDIR is the root of FinePar.
4. Run the programs using "bash run.sh". The results will be in "results" directory.
5. If you wants to try single application, please go into its directory and run related bash file.
6. If you want to read the results, execute "cd ${PROGRAMDIR}/results; bash show.sh" after Step 4 (bash run.sh).
7. We put a former rseult in the directory "results2016Nov18". You can step into that derectory to see the results with "bash show.sh". New results should look similar.
8. The source to source transform engine is under "codeTransEngine", which is based on Clang. This is a good example to Clang LibTooling developers.

The input tar file is in the google drive now: https://drive.google.com/file/d/0B0LKXOI68JYNVHFlcFpVbU1iWTA/view?usp=sharing

##Publication##
If you use our work, please cite our paper:

Feng Zhang, Bo Wu, Jidong Zhai, Bingsheng He, Wenguang Chen.FinePar: Irregularity-Aware Fine-Grained Workload Partitioning on Integrated Architectures
[CGO2017 URL](http://cgo.org/cgo2017/)

      @inproceedings{zhangFinePar,
       title={{FinePar: Irregularity-Aware Fine-Grained Workload Partitioning on Integrated Architectures}},
       author={Zhang, Feng and Wu, Bo and Zhai, Jidong and He, Bingsheng and Chen, Wenguang},
       booktitle={Proceedings of the 2017 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)},
       year={2017}
      }


##Acknowledgement##
*FinePar is developed by Tsinghua University, Colorado School of Mines, National University of Singapore.

Feng Zhang, Jidong Zhai and Wenguang Chen are with the Department of Computer Science and Technology, Tsinghua University, Beijing, 100084, China.

Bo Wu is with Computer Science Division, Colorado School of Mines, 1610 Illinois Street, Golden, CO 80401, USA.

Bingsheng He is with the School of Computing, National University of Singapore, 119077, Singapore.


If you have problems, please contact:
* zhangfeng.thu.hpc@gmail.com

Thanks for your interests in FinePar and hope you like it. FinePar: Irregularity-Aware Fine-Grained Workload Partitioning on Integrated Architectures

