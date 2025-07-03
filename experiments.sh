#!/bin/bash

#SBATCH --job-name=tsp_cpp_omp
#SBATCH --partition=standard
#SBATCH --output=logs/tsp_omp_run_%j.out
#SBATCH --error=logs/tsp_omp_run_%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      # Número total de hilos disponibles
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=robert.buleje@utec.edu.pe

runs=30
threads_list=(1 4 8 12 16 20 24 28 32) # Hilos OpenMP
matrices=("5nodos.txt" "11nodos.txt" "12nodos.txt" "15nodos.txt" "17nodos.txt")

exec="./tsp_bb"  # Asegúrate de compilar con: g++ -fopenmp tsp_branch_bound.cpp -o tsp_bb

mkdir -p logs

for file in "${matrices[@]}"; do
  for t in "${threads_list[@]}"; do
    echo "--------------------------------"
    echo "🧪 Ejecutando configuración: threads=$t, archivo=$file, runs=$runs"
    for i in $(seq 1 $runs); do
      echo "▶️ Run $i/$runs con threads=$t y archivo=$file"
      $exec $file $t
    done
  done
done

echo ""
echo "✅ Experimentos completados."
