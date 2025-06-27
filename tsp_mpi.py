import heapq
import sys
import time
import csv
import os
import random
from mpi4py import MPI
import pickle

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
    print("Matriz de distancias:")
    print("=" * 50)
    n = len(matrix)
    
    # Imprimir encabezado de columnas
    print("     ", end="")
    for j in range(n):
        print(f"Ciudad{j:2d} ", end="")
    print()
    
    # Imprimir filas con datos
    for i in range(n):
        print(f"Ciudad{i:2d} ", end="")
        for j in range(n):
            if matrix[i][j] == 0:
                print("      0   ", end="")
            else:
                print(f"{matrix[i][j]:8.0f} ", end="")
        print()
    print("=" * 50)
    print()

def generate_random_matrix(n, max_distance=500, seed=None):
    """Genera una matriz de distancias aleatoria para n ciudades"""
    if seed is not None:
        random.seed(seed)
    
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0  # Distancia de una ciudad a sí misma es 0
            elif i < j:
                # Generar distancia aleatoria solo para la parte superior de la matriz
                distance = random.randint(1, max_distance)
                matrix[i][j] = distance
                matrix[j][i] = distance  # Hacer la matriz simétrica
            # Para i > j, ya fue asignado cuando j < i
    
    return matrix

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

def expand_node(node, N):
    """Expande un nodo generando todos sus hijos válidos"""
    children = []
    i = node.vertex
    
    for j in range(N):
        if node.matrix_reduced[i][j] != INF:
            child = new_node(node.matrix_reduced, node.path, 
                           node.level + 1, i, j, N)
            child.cost = (node.cost + node.matrix_reduced[i][j] + 
                        cost_calculation(child.matrix_reduced, N))
            children.append(child)
    
    return children

def sequential_tsp(adjacency_matrix):
    """Resuelve TSP de forma secuencial (para cuando hay un solo proceso)"""
    N = len(adjacency_matrix)
    priority_queue = []
    heapq.heapify(priority_queue)
    
    # Crear nodo raíz
    root = new_node(adjacency_matrix, [], 0, -1, 0, N)
    root.cost = cost_calculation(root.matrix_reduced, N)
    heapq.heappush(priority_queue, root)
    
    best_solution = None
    best_cost = INF
    
    while priority_queue:
        current_node = heapq.heappop(priority_queue)
        
        # Poda por cota superior
        if current_node.cost >= best_cost:
            continue
        
        # Si es una solución completa
        if current_node.level == N - 1:
            current_node.path.append((current_node.vertex, 0))
            if current_node.cost < best_cost:
                best_cost = current_node.cost
                best_solution = current_node
        else:
            # Expandir nodo
            children = expand_node(current_node, N)
            for child in children:
                if child.cost < best_cost:
                    heapq.heappush(priority_queue, child)
    
    return best_solution

def worker_process(comm, rank):
    """Proceso trabajador que procesa nodos del árbol de búsqueda"""
    best_solution = None
    best_cost = INF
    local_queue = []
    heapq.heapify(local_queue)
    
    while True:
        # Verificar mensajes del maestro
        if comm.Iprobe(source=0, tag=MPI.ANY_TAG):
            status = MPI.Status()
            comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            
            if tag == 1:  # WORK_TAG - recibir nodo para procesar
                node = comm.recv(source=0, tag=1)
                heapq.heappush(local_queue, node)
                
            elif tag == 2:  # UPDATE_BOUND_TAG - actualizar cota superior
                new_bound = comm.recv(source=0, tag=2)
                if new_bound < best_cost:
                    best_cost = new_bound
                    
            elif tag == 3:  # TERMINATE_TAG - terminar
                break
        
        # Procesar nodos locales si los hay
        if local_queue:
            current_node = heapq.heappop(local_queue)
            
            # Poda por cota superior
            if current_node.cost >= best_cost:
                continue
            
            N = len(current_node.matrix_reduced)
            
            # Si es una solución completa
            if current_node.level == N - 1:
                current_node.path.append((current_node.vertex, 0))
                if current_node.cost < best_cost:
                    best_cost = current_node.cost
                    best_solution = current_node
                    # Reportar nueva mejor solución al maestro
                    comm.send(best_solution, dest=0, tag=4)
            else:
                # Expandir nodo
                children = expand_node(current_node, N)
                for child in children:
                    if child.cost < best_cost:
                        heapq.heappush(local_queue, child)
        
        # Si no hay trabajo local, solicitar más al maestro
        if not local_queue:
            comm.send(None, dest=0, tag=5)  # REQUEST_WORK_TAG
    
    return best_solution, best_cost

def master_process(comm, size, adjacency_matrix):
    """Proceso maestro que coordina la búsqueda"""
    N = len(adjacency_matrix)
    
    # Si solo hay un proceso, resolver secuencialmente
    if size == 1:
        return sequential_tsp(adjacency_matrix)
    
    global_queue = []
    heapq.heapify(global_queue)
    
    # Crear nodo raíz
    root = new_node(adjacency_matrix, [], 0, -1, 0, N)
    root.cost = cost_calculation(root.matrix_reduced, N)
    heapq.heappush(global_queue, root)
    
    best_solution = None
    best_cost = INF
    active_workers = size - 1
    pending_requests = 0
    
    # Tags para comunicación
    WORK_TAG = 1
    UPDATE_BOUND_TAG = 2
    TERMINATE_TAG = 3
    SOLUTION_TAG = 4
    REQUEST_WORK_TAG = 5
    
    # Distribuir trabajo inicial
    initial_work_distributed = 0
    while global_queue and initial_work_distributed < active_workers:
        if global_queue:
            node = heapq.heappop(global_queue)
            children = expand_node(node, N)
            for child in children:
                heapq.heappush(global_queue, child)
        initial_work_distributed += 1
    
    # Distribuir nodos iniciales a los trabajadores
    for worker_id in range(1, min(size, len(global_queue) + 1)):
        if global_queue:
            node = heapq.heappop(global_queue)
            comm.send(node, dest=worker_id, tag=WORK_TAG)
    
    while active_workers > 0:
        # Verificar mensajes de los trabajadores
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
            status = MPI.Status()
            comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            
            if tag == SOLUTION_TAG:  # Nueva mejor solución
                solution = comm.recv(source=source, tag=SOLUTION_TAG)
                if solution.cost < best_cost:
                    best_cost = solution.cost
                    best_solution = solution
                    
                    # Actualizar cota en todos los trabajadores
                    for worker_id in range(1, size):
                        comm.send(best_cost, dest=worker_id, tag=UPDATE_BOUND_TAG)
                        
            elif tag == REQUEST_WORK_TAG:  # Solicitud de trabajo
                comm.recv(source=source, tag=REQUEST_WORK_TAG)  # Consumir mensaje
                
                if global_queue:
                    # Enviar trabajo disponible
                    node = heapq.heappop(global_queue)
                    if node.cost < best_cost:
                        comm.send(node, dest=source, tag=WORK_TAG)
                    else:
                        pending_requests += 1
                else:
                    pending_requests += 1
                
                # Si todos los trabajadores están esperando trabajo y no hay más nodos
                if pending_requests >= active_workers and not global_queue:
                    break
        
        # Procesar algunos nodos localmente si la cola está muy llena
        nodes_processed = 0
        while global_queue and nodes_processed < 10:
            current_node = heapq.heappop(global_queue)
            
            if current_node.cost >= best_cost:
                nodes_processed += 1
                continue
            
            if current_node.level == N - 1:
                current_node.path.append((current_node.vertex, 0))
                if current_node.cost < best_cost:
                    best_cost = current_node.cost
                    best_solution = current_node
                    
                    # Actualizar cota en todos los trabajadores
                    for worker_id in range(1, size):
                        comm.send(best_cost, dest=worker_id, tag=UPDATE_BOUND_TAG)
            else:
                children = expand_node(current_node, N)
                for child in children:
                    if child.cost < best_cost:
                        heapq.heappush(global_queue, child)
            
            nodes_processed += 1
    
    # Terminar todos los trabajadores
    for worker_id in range(1, size):
        comm.send(None, dest=worker_id, tag=TERMINATE_TAG)
    
    return best_solution

def print_solution(solution):
    """Imprime la solución encontrada de forma legible"""
    if solution is None:
        print("No se encontró solución")
        return
    
    print(f"\nSolución encontrada:")
    print(f"Costo total del recorrido: {solution.cost}")
    print("Ruta del vendedor:")
    
    path_str = "Ciudad 0"
    for i, (from_city, to_city) in enumerate(solution.path):
        path_str += f" -> Ciudad {to_city}"
    
    print(path_str)
    print()

def write_to_csv(data: dict, headers: list[str], filename: str):
    """Escribe los resultados al archivo CSV"""
    file_exists = os.path.isfile(filename)

    with open(filename, "a") as f:
        if not file_exists:
            f.write(",".join(headers) + "\n")
            print(f"\nArchivo creado: {filename}\n")
        valores = [f"{data[h]:.4f}" if isinstance(data[h], float) else str(data[h]) for h in headers]
        f.write(",".join(valores) + "\n")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if len(sys.argv) < 2:
        if rank == 0:
            print("Uso: mpirun -np <num_procesos> python tsp_parallel.py <num_ciudades> [semilla_aleatoria] [distancia_maxima]")
            print("Ejemplo: mpirun -np 4 python tsp_parallel.py 5 42 300")
        return
    
    try:
        num_cities = int(sys.argv[1])
        if num_cities < 3:
            if rank == 0:
                print("Error: El número de ciudades debe ser al menos 3")
            return
    except ValueError:
        if rank == 0:
            print("Error: El número de ciudades debe ser un entero válido")
        return
    
    # Parámetros opcionales
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    max_distance = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    
    # Si solo hay un proceso, ejecutar todo localmente
    if size == 1:
        matrix = generate_random_matrix(num_cities, max_distance, seed)
        
        if rank == 0:
            print(f"Resolviendo TSP para {num_cities} ciudades")
            print(f"Distancia máxima entre ciudades: {max_distance}")
            if seed is not None:
                print(f"Semilla aleatoria: {seed}")
            print()
            print_matrix(matrix)
        
        start_time = time.time()
        result = sequential_tsp(matrix)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        if result:
            print_solution(result)
            print(f"Tiempo de ejecución: {elapsed_time:.5f} segundos")
            data = {
                "elapsed_time": elapsed_time,
                "num_processes": size,
                "num_cities": num_cities,
                "cost": result.cost
            }
            print(f"Datos de rendimiento: {data}")
        else:
            print("No se encontró solución")
        return
    
    # Código para múltiples procesos
    if rank == 0:
        matrix = generate_random_matrix(num_cities, max_distance, seed)
        
        print(f"Resolviendo TSP para {num_cities} ciudades con {size} procesos")
        print(f"Distancia máxima entre ciudades: {max_distance}")
        if seed is not None:
            print(f"Semilla aleatoria: {seed}")
        print()
        print_matrix(matrix)
        
        # Enviar matriz a todos los procesos trabajadores
        for i in range(1, size):
            comm.send(matrix, dest=i, tag=0)
    else:
        # Recibir matriz del proceso maestro
        matrix = comm.recv(source=0, tag=0)
    
    # Sincronizar todos los procesos antes de comenzar
    comm.Barrier()
    start_time = time.time()
    
    if rank == 0:
        # Proceso maestro
        result = master_process(comm, size, matrix)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        if result:
            print_solution(result)
            print(f"Tiempo de ejecución: {elapsed_time:.5f} segundos")
            data = {
                "elapsed_time": elapsed_time,
                "num_processes": size,
                "num_cities": num_cities,
                "cost": result.cost
            }
            print(f"Datos de rendimiento: {data}")
        else:
            print("No se encontró solución")
    else:
        # Procesos trabajadores
        worker_process(comm, rank)

if __name__ == "__main__":
    main()