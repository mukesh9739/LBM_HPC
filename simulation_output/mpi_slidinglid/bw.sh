##!/bin/bash
#SBATCH --nodes=41
#SBATCH --time=08:00:00
#SBATCH --partition=multiple
#SBATCH --ntasks-per-node=40
#SBATCH --output=test.out
#SBATCH --error=test.err
echo "Loading Python module and mpi module"
module load devel/python/3.8.6_gnu_10.2
module load mpi/openmpi/4.1
module list

steps=10000
echo -e "lattices\tdecomp\tsteps\tSECONDS\tMLUPS"
for lattices in $(seq 200 200 1000);
do
    for((decomp=2;decomp<=40;decomp++));
    do
        if [ $(($lattices % $decomp)) -eq 0 ]; 
        then
            SECONDS=0
            mpirun -n $(($decomp**2)) python3 ./mpi.py $lattices $steps  
            MLUPS=$(($((lattices**2)) * $steps / $SECONDS))  
            echo -e "$lattices\t\t$decomp\t$steps\t$SECONDS\t$MLUPS"
        fi
    done
done


