#include <bits/stdc++.h>
#include <omp.h>
#include "reader.h"
#include <ctime>
#include <chrono>
#include <fstream>      // ⬅️ NUEVO: para escribir el CSV

/***** utilidades originales *****/
template <typename T>
void print(std::vector<std::vector<T>> graph) {
    for (const auto& i : graph) {
        for (const auto& j : i) {
            if (j == INF) std::cout << "INF        ";
            else std::cout << j << "        ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

template <typename T>
struct Node {
    std::vector<std::pair<T, T>> path;
    std::vector<std::vector<T>> matrix_reduced;
    T cost;
    int vertex;
    int level;
};

template <typename T>
Node<T>* newNode(std::vector<std::vector<T>> matrix_parent,
                 std::vector<std::pair<T, T>> const& path,
                 int level, int i, int j, int N) {
    auto node = new Node<T>;
    node->path = path;
    if (level != 0) node->path.emplace_back(i, j);
    node->matrix_reduced = matrix_parent;

    for (int k = 0; level != 0 && k < N; ++k) {
        node->matrix_reduced[i][k] = INF;
        node->matrix_reduced[k][j] = INF;
    }

    node->matrix_reduced[j][0] = INF;
    node->level  = level;
    node->vertex = j;
    return node;
}

template <typename T>
void reduce_row(std::vector<std::vector<T>>& matrix_reduced,
                std::vector<T>& row, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (matrix_reduced[i][j] < row[i])
                row[i] = matrix_reduced[i][j];

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (matrix_reduced[i][j] != INF && row[i] != INF)
                matrix_reduced[i][j] -= row[i];
}

template <typename T>
void reduce_column(std::vector<std::vector<T>>& matrix_reduced,
                   std::vector<T>& col, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (matrix_reduced[i][j] < col[j])
                col[j] = matrix_reduced[i][j];

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (matrix_reduced[i][j] != INF && col[j] != INF)
                matrix_reduced[i][j] -= col[j];
}

template <typename T>
T cost_calculation(std::vector<std::vector<T>>& matrix_reduced, int N) {
    T cost = 0;
    std::vector<T> row(N, INF);
    reduce_row(matrix_reduced, row, N);
    std::vector<T> col(N, INF);
    reduce_column(matrix_reduced, col, N);

    for (int i = 0; i < N; ++i) {
        cost += (row[i] != INF) ? row[i] : 0;
        cost += (col[i] != INF) ? col[i] : 0;
    }
    return cost;
}

template <typename T>
void printPath(std::vector<std::pair<T, T>> const& list) {
    for (auto& p : list)
        std::cout << p.first + 1 << " -> " << p.second + 1 << '\n';
}

template <typename T>
class comp {
public:
    bool operator()(const Node<T>* lhs, const Node<T>* rhs) const {
        return lhs->cost > rhs->cost;
    }
};

template <typename T>
Node<T>* TSPbranchandbound(std::vector<std::vector<T>>& adjacensyMatrix) {
    std::priority_queue<Node<T>*, std::vector<Node<T>*>, comp<T>> pq;
    std::vector<std::pair<T, T>> v;
    int N = adjacensyMatrix[0].size();

    auto root = newNode<T>(adjacensyMatrix, v, 0, -1, 0, N);
    root->cost = cost_calculation(root->matrix_reduced, N);
    pq.push(root);

    while (!pq.empty()) {
        auto min = pq.top();
        pq.pop();
        int i = min->vertex;

        if (min->level == N - 1) {
            min->path.emplace_back(i, 0);
            return min;
        }

        #pragma omp parallel for shared(min) schedule(dynamic)
        for (int j = 0; j < N; ++j) {
            if (min->matrix_reduced[i][j] != INF) {
                auto* child = newNode<T>(min->matrix_reduced, min->path,
                                         min->level + 1, i, j, N);
                child->cost = min->cost + min->matrix_reduced[i][j] +
                              cost_calculation<T>(child->matrix_reduced, N);
                #pragma omp critical
                pq.push(child);
            }
        }
        delete min;
    }
    return nullptr;
}

/***** MAIN *****/
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <archivo_matriz> <hilos>\n";
        return 1;
    }

    /* --------  ⏱️  Inicio de medición de tiempo total  -------- */
    auto program_start = std::chrono::high_resolution_clock::now();

    std::string nombreArchivo = argv[1];
    std::string outputFile    = "output.csv";
    int threads               = std::stoi(argv[2]);

    omp_set_num_threads(threads);
    auto matrix = leerArchivo<int>(nombreArchivo);   // I/O: cuenta como overhead

    /* ---------- ⏱️  Tiempo de cómputo  ---------- */
    auto compute_start = std::chrono::high_resolution_clock::now();
    auto ans           = TSPbranchandbound(matrix);
    auto compute_end   = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> compute_time = compute_end - compute_start;

    /* --------  Fin de programa (tiempo total)  -------- */
    auto program_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = program_end - program_start;

    double overhead_sec = total_time.count() - compute_time.count();
    if (overhead_sec < 0.0) overhead_sec = 0.0;   // evita negativos por redondeo

    /* ----------  Salida por pantalla  ---------- */
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Total(s):     "   << total_time.count()   << '\n';
    std::cout << "Compute(s):   "   << compute_time.count() << '\n';
    std::cout << "Overhead(s):  "   << overhead_sec         << '\n';

    /* ----------  Guardar en CSV  ---------- */
    std::ofstream csv(outputFile, std::ios::app);
    if (csv.tellp() == 0) {                // cabecera solo si el archivo estaba vacío
        csv << "archivo,threads,total,compute,overhead\n";
    }
    csv << nombreArchivo << ','
        << threads       << ','
        << total_time.count()   << ','
        << compute_time.count() << ','
        << overhead_sec         << '\n';
    csv.close();

    // writeMinPath<int>(ans->path, ans->cost);  // sigue disponible si lo necesitas
    return 0;
}

/* Dataset reference:
   https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html */