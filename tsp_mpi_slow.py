from mpi4py import MPI
import heapq
import time
import sys

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

def tsp_branch_and_bound_partial(matrix, start_nodes):
    N = len(matrix)
    pq = []
    heapq.heapify(pq)
    
    for j in start_nodes:
        root = new_node(matrix, [], 1, 0, j, N)
        root.cost = matrix[0][j] + cost_calculation(root.matrix_reduced, N)
        heapq.heappush(pq, root)

    best = None
    while pq:
        current = heapq.heappop(pq)
        i = current.vertex
        if current.level == N - 1:
            current.path.append((i, 0))
            if best is None or current.cost < best.cost:
                best = current
            continue
        for j in range(N):
            if current.matrix_reduced[i][j] != INF:
                child = new_node(current.matrix_reduced, current.path, current.level + 1, i, j, N)
                child.cost = current.cost + current.matrix_reduced[i][j] + cost_calculation(child.matrix_reduced, N)
                heapq.heappush(pq, child)
    return best


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


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    matrix = None
    N = 0
    if rank == 0:
        filename = sys.argv[1]
        matrix = read_file(filename)
        N = len(matrix)

    # Medición broadcast
    t0 = MPI.Wtime()
    matrix = comm.bcast(matrix, root=0)
    N = comm.bcast(N, root=0)
    t1 = MPI.Wtime()

    # División de nodos hijos del nodo raíz (ciudad 0)
    assigned = [j for j in range(1, N) if j % size == rank]

    # Resolución local
    t2 = MPI.Wtime()
    result = tsp_branch_and_bound_partial(matrix, assigned)
    local_cost = result.cost if result else INF
    local_path = result.path if result else []
    t3 = MPI.Wtime()

    # Gather en el root
    gathered = comm.gather((local_cost, local_path), root=0)
    t4 = MPI.Wtime()

    if rank == 0:
        best_cost, best_path = min(gathered, key=lambda x: x[0])
        print(f"Mejor costo: {best_cost}")
        print("Camino:")
        for i, j in best_path:
            print(f"{i + 1} -> {j + 1}")
        print(f"T_broadcast: {t1 - t0:.5f}s")
        print(f"T_solve: {t3 - t2:.5f}s")
        print(f"T_gather: {t4 - t3:.5f}s")

if __name__ == "__main__":
    main()