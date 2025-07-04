#include<bits/stdc++.h>
#include <omp.h>
#include "reader.h"
#include <ctime>
#include <chrono>

// Estructura para métricas de performance
struct PerformanceMetrics {
    double total_time;
    double computation_time;
    double synchronization_time;
    long long nodes_created;
    long long nodes_processed;
    int threads_used;
};

// Variables globales para métricas
std::atomic<long long> global_nodes_created(0);
std::atomic<long long> global_nodes_processed(0);
double total_sync_time = 0.0;
double total_comp_time = 0.0;

// Función para escribir métricas completas en CSV
void writePerformanceMetrics(const std::string& filename, const std::string& instance, const PerformanceMetrics& metrics) {
    std::ofstream file;
    bool file_exists = false;
    
    // Verificar si el archivo existe Y tiene contenido
    std::ifstream check_file(filename);
    if (check_file.good()) {
        std::string first_line;
        if (std::getline(check_file, first_line)) {
            // Si la primera línea no contiene "instance", no hay encabezados
            file_exists = (first_line.find("instance") != std::string::npos);
        }
        check_file.close();
    }
    
    file.open(filename, std::ios::app);
    
    // Escribir encabezados si el archivo no existe o está vacío
    if (!file_exists) {
        file << "instance,threads,nodes,total_time_seconds,computation_time_seconds,overhead_time_seconds,nodes_created,nodes_processed,computation_efficiency_percent,overhead_percent\n";
    }
    
    // Calcular métricas derivadas con validación
    double computation_efficiency = 0.0;
    double overhead_percent = 0.0;
    
    if (metrics.total_time > 0) {
        computation_efficiency = (metrics.computation_time / metrics.total_time) * 100.0;
        overhead_percent = (metrics.synchronization_time / metrics.total_time) * 100.0;
    }
    
    // Extraer número de nodos del nombre del archivo
    std::string nodes_str = instance;
    size_t pos = instance.find("nodos");
    if (pos != std::string::npos) {
        nodes_str = instance.substr(0, pos);
        // Eliminar caracteres no numéricos del inicio
        auto it = std::find_if(nodes_str.begin(), nodes_str.end(), ::isdigit);
        if (it != nodes_str.end()) {
            nodes_str = std::string(it, nodes_str.end());
        }
    }
    
    // Escribir datos
    file << std::fixed << std::setprecision(6);
    file << instance << ","
         << metrics.threads_used << ","
         << nodes_str << ","
         << metrics.total_time << ","
         << metrics.computation_time << ","
         << metrics.synchronization_time << ","
         << metrics.nodes_created << ","
         << metrics.nodes_processed << ","
         << computation_efficiency << ","
         << overhead_percent << "\n";
    
    file.close();
}

template <typename T>
void print(std::vector<std::vector<T>> graph) {
    for(const auto& i: graph){
        for(const auto& j: i){
            if(j == INF) std::cout << "INF        ";
            else std::cout << j << "        ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
struct Node
{
    std::vector<std::pair<T, T>> path;
    std::vector<std::vector<T>> matrix_reduced;
    T cost;
    int vertex;
    int level;
};

template <typename T>
Node<T>* newNode(std::vector<std::vector<T>> matrix_parent, std::vector<std::pair<T, T>> const &path, int level, int i, int j, int N)
{
    auto node = new Node<T>;
    node->path = path;
    if (level != 0)
        node->path.push_back(std::make_pair(i, j));
    node->matrix_reduced = matrix_parent;
    for (int k = 0; level != 0 && k < N; k++)
    {
        node->matrix_reduced[i][k] = INF;
        node->matrix_reduced[k][j] = INF;
    }

    node->matrix_reduced[j][0] = INF;
    node->level = level;
    node->vertex = j;
    
    // Incrementar contador de nodos creados
    global_nodes_created.fetch_add(1);
    
    return node;
}

template <typename T>
void reduce_row(std::vector<std::vector<T>> &matrix_reduced, std::vector<T> &row, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (matrix_reduced[i][j] < row[i])
                row[i] = matrix_reduced[i][j];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (matrix_reduced[i][j] != INF && row[i] != INF)
                matrix_reduced[i][j] -= row[i];
}

template <typename T>
void reduce_column(std::vector<std::vector<T>> &matrix_reduced, std::vector<T> &col, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (matrix_reduced[i][j] < col[j])
                col[j] = matrix_reduced[i][j];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (matrix_reduced[i][j] != INF && col[j] != INF)
                matrix_reduced[i][j] -= col[j];
}

template <typename T>
T cost_calculation(std::vector<std::vector<T>> &matrix_reduced, int N)
{
    T cost = 0;
    std::vector<T> row(N, INF);
    reduce_row(matrix_reduced, row, N);
    std::vector<T> col(N, INF);
    reduce_column(matrix_reduced, col, N);

    for (int i = 0; i < N; i++)
        cost += (row[i] != INF) ? row[i] : 0,
            cost += (col[i] != INF) ? col[i] : 0;

    return cost;
}

template <typename T>
void printPath(std::vector<std::pair<T, T>> const &list)
{
    for (int i = 0; i < list.size(); i++)
        std::cout << list[i].first + 1 << " -> " << list[i].second + 1 << std::endl;
}

template <typename T>
class comp {
public:
    bool operator()(const Node<T>* lhs, const Node<T>* rhs) const
    {
        return lhs->cost > rhs->cost;
    }
};

template <typename T>
Node<T>* TSPbranchandbound(std::vector<std::vector<T>> &adjacensyMatrix, PerformanceMetrics &metrics)
{
    std::priority_queue<Node<T>*, std::vector<Node<T>*>, comp<T>> pq;
    std::vector<std::pair<T, T>> v;
    int N = adjacensyMatrix[0].size();
    
    auto root = newNode<T>(adjacensyMatrix, v, 0, -1, 0, N);
    root->cost = cost_calculation(root->matrix_reduced, N);
    pq.push(root);
    
    double total_computation_time = 0.0;
    double total_overhead_time = 0.0;
    
    while (!pq.empty())
    {
        auto min = pq.top();
        pq.pop();
        global_nodes_processed.fetch_add(1);
        
        int i = min->vertex;
        if (min->level == N - 1)
        {
            min->path.push_back(std::make_pair(i, 0));
            
            // Actualizar métricas finales
            metrics.computation_time = total_computation_time;
            metrics.synchronization_time = total_overhead_time;
            metrics.nodes_created = global_nodes_created.load();
            metrics.nodes_processed = global_nodes_processed.load();
            
            return min;
        }

        // MEDIR TIEMPO TOTAL DE LA REGIÓN PARALELA
        auto parallel_start = std::chrono::high_resolution_clock::now();
        
        // Variable para acumular el tiempo máximo de computación entre hilos
        double max_thread_comp_time = 0.0;
        
        #pragma omp parallel shared(min, max_thread_comp_time)
        {
            double thread_comp_time = 0.0;
            
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < N; j++)
            {
                if (min->matrix_reduced[i][j] != INF)
                {
                    // Medir tiempo de computación por hilo
                    auto thread_comp_start = std::chrono::high_resolution_clock::now();
                    
                    Node<T>* child = newNode<T>(min->matrix_reduced, min->path, min->level + 1, i, j, N);
                    child->cost = min->cost + min->matrix_reduced[i][j] + cost_calculation<T>(child->matrix_reduced, N);
                    
                    auto thread_comp_end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> thread_elapsed = thread_comp_end - thread_comp_start;
                    thread_comp_time += thread_elapsed.count();
                    
                    // Medir tiempo de sincronización (critical section)
                    auto sync_start = std::chrono::high_resolution_clock::now();
                    #pragma omp critical
                    {
                        pq.push(child);
                    }
                    auto sync_end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> sync_elapsed = sync_end - sync_start;
                    
                    #pragma omp atomic
                    total_overhead_time += sync_elapsed.count();
                }
            }
            
            // Actualizar el tiempo máximo de computación entre todos los hilos
            #pragma omp critical
            {
                if (thread_comp_time > max_thread_comp_time) {
                    max_thread_comp_time = thread_comp_time;
                }
            }
        }
        
        auto parallel_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> parallel_elapsed = parallel_end - parallel_start;
        
        // El tiempo de computación es el máximo entre los hilos (tiempo del camino crítico)
        total_computation_time += max_thread_comp_time;
        
        // El overhead adicional es el tiempo total de la región paralela menos el tiempo de computación
        // Esto incluye creación/destrucción de hilos, barreras implícitas, etc.
        double region_overhead = parallel_elapsed.count() - max_thread_comp_time;
        if (region_overhead > 0) {
            total_overhead_time += region_overhead;
        }

        delete min;
    }
    
    // Actualizar métricas finales (por si no se encontró solución)
    metrics.computation_time = total_computation_time;
    metrics.synchronization_time = total_overhead_time;
    metrics.nodes_created = global_nodes_created.load();
    metrics.nodes_processed = global_nodes_processed.load();
    
    return nullptr;
}

int main(int argc, char *argv[]) {
    std::string nombreArchivo = argv[1];
    std::string outputFile = "output.csv";
    int threads = std::stoi(argv[2]);
    omp_set_num_threads(threads);
    
    auto matrix = leerArchivo<int>(nombreArchivo);
    
    // Inicializar métricas
    PerformanceMetrics metrics;
    metrics.threads_used = threads;
    
    // Medir tiempo total
    auto start = std::chrono::high_resolution_clock::now();	
    auto ans = TSPbranchandbound(matrix, metrics);
    auto finish = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = finish - start;
    metrics.total_time = elapsed.count();
    
    // Imprimir métricas
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Tiempo total: " << metrics.total_time << " segundos" << std::endl;
    std::cout << "Tiempo computación: " << metrics.computation_time << " segundos" << std::endl;
    std::cout << "Tiempo overhead: " << metrics.synchronization_time << " segundos" << std::endl;
    std::cout << "Nodos creados: " << metrics.nodes_created << std::endl;
    std::cout << "Nodos procesados: " << metrics.nodes_processed << std::endl;
    std::cout << "Eficiencia computacional: " << (metrics.computation_time / metrics.total_time) * 100 << "%" << std::endl;
    
    // Guardar métricas completas en CSV
    writePerformanceMetrics(outputFile, nombreArchivo, metrics);
    
    return 0;
}