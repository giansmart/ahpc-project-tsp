# Análisis de Rendimiento de Applied High Performance Computing (AHPC)  
## Problema del Viajero (TSP) con Branch and Bound en C++ usando OpenMP

Este repositorio contiene el análisis de rendimiento y escalabilidad de una implementación paralela del algoritmo **Branch and Bound** para el **Problema del Viajero (TSP)**, desarrollado en **C++ con OpenMP** como parte del curso de **Applied High Performance Computing**.

---

## 📌 Objetivo

El objetivo principal es analizar el comportamiento del algoritmo al aplicar paralelismo mediante hilos (OpenMP), evaluando:

- **Tiempo Teórico vs Tiempo experimental**.
- **Speedup** y **eficiencia paralela**.
- **Sobrecosto de sincronización**.
- **Impacto del número de hilos y nodos**.
- **Limitaciones teóricas (Ley de Amdahl)**.

---

## 🧠 Descripción del Algoritmo

- **Algoritmo**: Branch and Bound exacto para TSP.
- **Paralelismo**: Distribución de subramas del árbol de búsqueda entre múltiples hilos.
- **Lenguaje**: C++
- **Paralelismo**: OpenMP (`#pragma omp parallel`)

---

## 🧪 Dataset de Salida (`output.csv`)

El archivo `output.csv` contiene los siguientes campos:

| Columna                    | Descripción                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| instance                  | Nombre o ID de la instancia TSP.                                           |
| threads                   | Número de hilos usados (OpenMP).                                           |
| nodes                     | Nodos distribuidos al paralelizar el árbol de búsqueda.                    |
| total_time_seconds        | Tiempo total de ejecución.                                                 |
| computation_time_seconds  | Tiempo efectivo de cómputo útil.                                           |
| synchronization_time_seconds | Tiempo perdido en sincronización entre hilos.                        |
| nodes_created             | Nodos generados por el algoritmo.                                          |
| nodes_processed           | Nodos realmente procesados (no podados).                                   |
| parallel_efficiency_percent | Eficiencia de la paralelización respecto al ideal.                    |
| sync_overhead_percent     | Porcentaje del tiempo total consumido en sincronización.


---

## 🔧 Requisitos

### Compilar el código C++

```bash
g++ -std=c++17 -fopenmp tsp_metrics_par.cpp -o tsp_bb   
./tsp_par --ARCHIVO_MATRIZ=5nodos.txt --OMP_NUM_THREADS=4
```

---

## 📚 Créditos
Este trabajo fue realizado como parte del curso de Applied High Performance Computing.

Autores:
- Robert Junior Buleje del Carpio
- Giancarlo Poémape
- Diana Sanchez

Docente: 
- PhD. Jose Antonio Fiestas Iquira
