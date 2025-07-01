#SBATCH --job-name=knn_batch_test
#SBATCH --output=logs/tsp_run_%j.out
#SBATCH --error=logs/tsp_run_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8         # Máximo de procesos que se usarán
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

runs=10
p_list=(1 2 4) # procesos
n_list=(4 6 8 10) # ciudades

script="tsp_mpi.py"

# Crear carpeta para logs si no existe
mkdir -p logs

# Ejecutar combinaciones
for p in "${p_list[@]}"; do
  for n in "${n_list[@]}"; do
    echo "--------------------------------"
    echo "🧪 Ejecutando configuración: p=$p, n=$n, runs=$runs"
    for i in $(seq 1 $runs); do
      echo ""
      echo "  ▶️ Run $i/$runs con p=$p y n=$n"
    # srun -n $p --exclusive python3 $script $n
    mpiexec -n $p python3 $script $n
    done
  done
done

echo ""
echo "✅ Experimentos completados."