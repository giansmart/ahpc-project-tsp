#!/bin/bash

#SBATCH --job-name=tsp
#SBATCH --partition=standard
#SBATCH --output=logs/tsp_run_%j.out
#SBATCH --error=logs/tsp_run_%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=32        # M  ximo de procesos que se usar  n
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=robert.buleje@utec.edu.pe
runs=30
p_list=(1 4 8 12 16 20 24 28 32) # procesos
n_list=(4 6 8 10 12 14 16 18) # ciudades

script="tsp_mpi.py"

# Crear carpeta para logs si no existe
mkdir -p logs

# Ejecutar combinaciones
for n in "${n_list[@]}"; do
  for p in "${p_list[@]}"; do
    echo "--------------------------------"
    echo " ^=   Ejecutando configuraci  n: p=$p, n=$n, runs=$runs"
    for i in $(seq 1 $runs); do
      echo ""
      echo " ^v   ^o Run $i/$runs con p=$p y n=$n"
    # srun -n $p --exclusive python3 $script $n
      mpiexec -n $p python3 $script $n
    done
  done
done

echo ""
echo " ^|^e Experimentos completados."