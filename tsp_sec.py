import heapq
import sys
import time
import csv
import copy

# Constante para infinito
INF = float('inf')

class Node:
    """Estructura para representar un nodo en el árbol de búsqueda"""
    def __init__(self):
        self.path = []
        self.matrix_reduced = []
        self.cost = 0
        self.vertex = 0
        self.level = 0
    
    def __lt__(self, other):
        """Comparador para la cola de prioridad (min-heap)"""
        return self.cost < other.cost

def print_matrix(matrix):
    """Imprime la matriz de adyacencia"""
    for row in matrix:
        for val in row:
            if val == INF:
                print("INF        ", end="")
            else:
                print(f"{val}        ", end="")
        print()
    print()

def new_node(matrix_parent, path, level, i, j, N):
    """Crea un nuevo nodo en el árbol de búsqueda"""
    node = Node()
    node.path = path.copy()
    
    if level != 0:
        node.path.append((i, j))
    
    # Copia profunda de la matriz
    node.matrix_reduced = [row[:] for row in matrix_parent]
    
    # Establecer infinito en fila i y columna j
    if level != 0:
        for k in range(N):
            node.matrix_reduced[i][k] = INF
            node.matrix_reduced[k][j] = INF
    
    # Prevenir subtours
    node.matrix_reduced[j][0] = INF
    node.level = level
    node.vertex = j
    
    return node

def reduce_row(matrix_reduced, N):
    """Reduce las filas de la matriz y devuelve el costo de reducción"""
    row_min = [INF] * N
    
    # Encontrar el mínimo de cada fila
    for i in range(N):
        for j in range(N):
            if matrix_reduced[i][j] < row_min[i]:
                row_min[i] = matrix_reduced[i][j]
    
    # Restar el mínimo de cada fila
    for i in range(N):
        for j in range(N):
            if matrix_reduced[i][j] != INF and row_min[i] != INF:
                matrix_reduced[i][j] -= row_min[i]
    
    return row_min

def reduce_column(matrix_reduced, N):
    """Reduce las columnas de la matriz y devuelve el costo de reducción"""
    col_min = [INF] * N
    
    # Encontrar el mínimo de cada columna
    for j in range(N):
        for i in range(N):
            if matrix_reduced[i][j] < col_min[j]:
                col_min[j] = matrix_reduced[i][j]
    
    # Restar el mínimo de cada columna
    for i in range(N):
        for j in range(N):
            if matrix_reduced[i][j] != INF and col_min[j] != INF:
                matrix_reduced[i][j] -= col_min[j]
    
    return col_min

def cost_calculation(matrix_reduced, N):
    """Calcula el costo de la reducción de matriz"""
    cost = 0
    
    row_min = reduce_row(matrix_reduced, N)
    col_min = reduce_column(matrix_reduced, N)
    
    # Sumar los costos de reducción
    for i in range(N):
        cost += row_min[i] if row_min[i] != INF else 0
        cost += col_min[i] if col_min[i] != INF else 0
    
    return cost

def print_path(path):
    """Imprime el camino encontrado"""
    for i, j in path:
        print(f"{i + 1} -> {j + 1}")

def tsp_branch_and_bound(adjacency_matrix):
    """
    Resuelve el TSP usando Branch and Bound (versión secuencial)
    """
    N = len(adjacency_matrix)
    pq = []
    heapq.heapify(pq)
    
    # Crear nodo raíz
    root = new_node(adjacency_matrix, [], 0, -1, 0, N)
    root.cost = cost_calculation(root.matrix_reduced, N)
    
    heapq.heappush(pq, root)
    
    while pq:
        current_min = heapq.heappop(pq)
        i = current_min.vertex
        
        # Si hemos visitado todos los nodos, agregar el regreso al inicio
        if current_min.level == N - 1:
            current_min.path.append((i, 0))
            return current_min
        
        # Generar todos los nodos hijos
        for j in range(N):
            if current_min.matrix_reduced[i][j] != INF:
                child = new_node(current_min.matrix_reduced, current_min.path, 
                               current_min.level + 1, i, j, N)
                child.cost = (current_min.cost + current_min.matrix_reduced[i][j] + 
                            cost_calculation(child.matrix_reduced, N))
                heapq.heappush(pq, child)
    
    return None

def read_file(filename):
    """Lee la matriz de adyacencia desde un archivo"""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            matrix = []
            for line in lines:
                row = []
                values = line.strip().split()
                for val in values:
                    if val == 'INF' or val == '-1':
                        row.append(INF)
                    else:
                        row.append(int(val))
                matrix.append(row)
            return matrix
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {filename}")
        return None
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None

def write_to_csv(filename, data):
    """Escribe los resultados al archivo CSV"""
    try:
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([data[0], data[1]])
    except Exception as e:
        print(f"Error al escribir en CSV: {e}")

def write_min_path(path, cost, filename="min_path.txt"):
    """Escribe el camino mínimo encontrado"""
    try:
        with open(filename, 'w') as file:
            file.write(f"Costo mínimo: {cost}\n")
            file.write("Camino:\n")
            for i, j in path:
                file.write(f"{i + 1} -> {j + 1}\n")
    except Exception as e:
        print(f"Error al escribir el camino: {e}")

def main():
    filename = "5nodos.txt"
    
    # Leer matriz desde archivo
    matrix = read_file(filename)
    if matrix is None:
        print("Error: No se pudo leer el archivo. Usando matriz de prueba.")
        # Usar matriz de prueba si no se puede leer el archivo
        matrix = [
            [INF, 20, 30, 10, 11],
            [15, INF, 16, 4, 2],
            [3, 5, INF, 2, 4],
            [19, 6, 18, INF, 3],
            [16, 4, 7, 16, INF]
        ]
    
    # Matrices de prueba adicionales
    """
    # Matriz de prueba 1 - Resultado esperado: 28
    ad = [
        [INF, 20, 30, 10, 11],
        [15, INF, 16, 4, 2],
        [3, 5, INF, 2, 4],
        [19, 6, 18, INF, 3],
        [16, 4, 7, 16, INF]
    ]
    
    # Matriz de prueba 2
    test = [
        [INF, 3, 4, 2, 7],
        [3, INF, 4, 6, 3],
        [4, 4, INF, 5, 8],
        [2, 6, 5, INF, 6],
        [7, 3, 8, 6, INF]
    ]
    """
    
    # Medir tiempo de ejecución
    start_time = time.time()
    result = tsp_branch_and_bound(matrix)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"{elapsed_time:.5f}")
    
    if result:
        print(result.cost)
        # Opcional: mostrar el camino completo
        # print("Camino encontrado:")
        # print_path(result.path)
    else:
        print("No se encontró solución")

if __name__ == "__main__":
    main()