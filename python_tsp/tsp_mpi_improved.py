import heapq
import sys
import time
import csv
import os
import random
from mpi4py import MPI
import numpy as np
from numba import jit

INF = float('inf')

class Node:
    def __init__(self, matrix=None, path=None, cost=0, vertex=0, level=0):
        self.path = path if path is not None else []
        self.matrix_reduced = matrix.copy() if matrix is not None else None
        self.cost = cost
        self.vertex = vertex
        self.level = level

    def __lt__(self, other):
        return self.cost < other.cost

@jit(nopython=True, cache=True)
def generate_random_matrix_numba(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    matrix = np.full((n, n), INF, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.random.randint(1, 1001)
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix

@jit(nopython=True, cache=True)
def reduce_row_numba(matrix_reduced):
    n = matrix_reduced.shape[0]
    total_reduction = 0.0
    for i in range(n):
        min_val = np.min(matrix_reduced[i])
        if min_val != INF and min_val > 0:
            total_reduction += min_val
            for j in range(n):
                if matrix_reduced[i, j] != INF:
                    matrix_reduced[i, j] -= min_val
    return total_reduction

@jit(nopython=True, cache=True)
def reduce_column_numba(matrix_reduced):
    n = matrix_reduced.shape[0]
    total_reduction = 0.0
    for j in range(n):
        min_val = np.min(matrix_reduced[:, j])
        if min_val != INF and min_val > 0:
            total_reduction += min_val
            for i in range(n):
                if matrix_reduced[i, j] != INF:
                    matrix_reduced[i, j] -= min_val
    return total_reduction

@jit(nopython=True, cache=True)
def cost_calculation_numba(matrix_reduced):
    matrix_copy = matrix_reduced.copy()
    return reduce_row_numba(matrix_copy) + reduce_column_numba(matrix_copy)

def generate_random_matrix(n, seed=42):
    """Genera matriz con seed fija para reproducibilidad"""
    return generate_random_matrix_numba(n, seed)

def new_node(matrix_parent, path, level, i, j, N):
    matrix_np = matrix_parent.copy()
    
    if level == 0:
        # Nodo raíz
        cost = cost_calculation_numba(matrix_np)
        return Node(matrix=matrix_np, path=[], cost=cost, vertex=0, level=0)
    else:
        # Eliminar fila i y columna j
        matrix_np[i, :] = INF
        matrix_np[:, j] = INF
        
        # Evitar subtours prematuros
        if j != 0 and level < N - 1:
            matrix_np[j, 0] = INF
            
        new_path = path.copy()
        new_path.append((i, j))
        
        # Calcular costo reducido
        reduction_cost = cost_calculation_numba(matrix_np)
        
        return Node(matrix=matrix_np, path=new_path, vertex=j, level=level, cost=reduction_cost)

def expand_node(node, N, original_matrix):
    """Expande un nodo generando todos sus hijos válidos"""
    children = []
    i = node.vertex
    
    # Conjunto de ciudades ya visitadas
    visited = {0}  # Ciudad 0 siempre está visitada (inicio)
    for _, dest in node.path:
        visited.add(dest)
    
    for j in range(N):
        if j not in visited and node.matrix_reduced[i, j] != INF:
            # Crear nodo hijo
            child = new_node(node.matrix_reduced, node.path, node.level + 1, i, j, N)
            
            # Calcular costo total = costo acumulado + costo de la arista + costo reducido
            edge_cost = original_matrix[i, j]
            child.cost = node.cost + edge_cost + child.cost
            
            children.append(child)
    
    return children

def print_matrix(matrix):
    """Imprime la matriz de adyacencia"""
    print("\nMatriz de distancias:")
    print("=" * 80)
    n = len(matrix)
    
    # Imprimir encabezado de columnas
    print("        ", end="")
    for j in range(n):
        print(f"Ciudad{j:2d} ", end="")
    print()
    print("        " + "-" * (n * 9))
    
    # Imprimir filas con datos
    for i in range(n):
        print(f"Ciudad{i:2d} ", end="")
        for j in range(n):
            if i == j:  # Diagonal principal
                print("       0 ", end="")
            elif matrix[i][j] == INF:
                print("     INF ", end="")
            else:
                print(f"{matrix[i][j]:8.0f} ", end="")
        print()
    print("=" * 80)
    print()

def print_solution(cost, path):
    """Imprime la solución encontrada de forma legible"""
    if path is None:
        print("No se encontró solución")
        return
    
    print(f"\n=== SOLUCIÓN ENCONTRADA ===")
    print(f"Costo total del recorrido: {cost:.0f}")
    print("Ruta del vendedor:")
    
    # Construir string de la ruta
    path_str = "Ciudad 0"
    for from_city, to_city in path:
        path_str += f" -> Ciudad {to_city}"
    
    print(path_str)
    print("=" * 40)
    print()

def save_result(data: dict, filename: str = "resultados.txt"):
    file_exists = os.path.isfile(filename)
    headers = list(data.keys())

    with open(filename, "a") as f:
        if not file_exists:
            f.write(",".join(headers) + "\n")
            print(f"\nArchivo creado: {filename}\n")
        valores = [f"{data[h]:.4f}" if isinstance(data[h], float) else str(data[h]) for h in headers]
        f.write(",".join(valores) + "\n")
        f.flush()
        os.fsync(f.fileno())

def parallel_tsp_worker(matrix, comm, rank):
    """Worker process para explorar nodos"""
    N = len(matrix)
    best_cost = INF
    nodes_explored = 0
    compute_time = 0.0  # Tiempo de cómputo puro
    
    while True:
        # Recibir trabajo del master (no contamos como tiempo de cómputo)
        data = comm.recv(source=0, tag=MPI.ANY_TAG)
        
        if data == "TERMINATE":
            # Enviar estadísticas finales antes de terminar
            comm.send(("STATS", nodes_explored, compute_time), dest=0)
            break
            
        work_nodes, current_best = data
        
        # Actualizar mejor costo conocido
        if current_best < best_cost:
            best_cost = current_best
        
        # Listas para resultados
        new_nodes = []
        solutions = []
        
        # AQUÍ MEDIMOS EL TIEMPO DE CÓMPUTO REAL
        comp_start = time.time()
        
        for node in work_nodes:
            nodes_explored += 1
            
            # Poda: si el costo del nodo es mayor o igual al mejor conocido, descartarlo
            if node.cost >= best_cost:
                continue
            
            # Si es el último nivel, verificar si es una solución completa
            if node.level == N - 1:
                # Costo de regresar al origen
                return_cost = matrix[node.vertex, 0]
                if return_cost != INF:
                    total_cost = node.cost + return_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        complete_path = node.path + [(node.vertex, 0)]
                        solutions.append((total_cost, complete_path))
            else:
                # Expandir el nodo
                children = expand_node(node, N, matrix)
                for child in children:
                    # Solo agregar hijos prometedores
                    if child.cost < best_cost:
                        new_nodes.append(child)
        
        compute_time += time.time() - comp_start
        
        # Enviar resultados al master (no contamos como tiempo de cómputo)
        comm.send(("RESULT", solutions, best_cost, new_nodes, nodes_explored), dest=0)

def sequential_tsp(matrix):
    """Versión secuencial del TSP para cuando solo hay un proceso"""
    N = len(matrix)
    best_cost = INF
    best_path = None
    nodes_explored = 0
    
    # Medir tiempo de cómputo
    compute_start = time.time()
    
    # Inicializar con nodo raíz
    root = new_node(matrix, [], 0, 0, 0, N)
    queue = []
    heapq.heappush(queue, root)
    
    while queue:
        node = heapq.heappop(queue)
        nodes_explored += 1
        
        # Poda
        if node.cost >= best_cost:
            continue
        
        # Solución completa
        if node.level == N - 1:
            return_cost = matrix[node.vertex, 0]
            if return_cost != INF:
                total_cost = node.cost + return_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = node.path + [(node.vertex, 0)]
        else:
            # Expandir nodo
            children = expand_node(node, N, matrix)
            for child in children:
                if child.cost < best_cost:
                    heapq.heappush(queue, child)
    
    compute_time = time.time() - compute_start
    
    # Retornar tupla con solución, tiempo de cómputo y nodos explorados
    if best_path:
        return (best_cost, best_path), compute_time, nodes_explored
    else:
        return None, compute_time, nodes_explored

def parallel_tsp_master(matrix, comm, size, num_cities):
    """Master process que coordina la búsqueda"""
    N = len(matrix)
    
    # Si solo hay un proceso, ejecutar versión secuencial
    if size == 1:
        solution, compute_time, nodes_explored = sequential_tsp(matrix)
        return solution, compute_time, nodes_explored
    
    # Tiempo de cómputo del master
    master_compute_time = 0.0
    
    # Inicializar nodo raíz y expandir (esto es cómputo)
    comp_start = time.time()
    root = new_node(matrix, [], 0, 0, 0, N)
    initial_nodes = expand_node(root, N, matrix)
    
    # Cola de trabajo con heap para priorizar nodos prometedores
    work_queue = []
    for node in initial_nodes:
        heapq.heappush(work_queue, node)
    master_compute_time += time.time() - comp_start
    
    best_solution = None
    best_cost = INF
    
    # Estadísticas
    total_nodes_explored = 0
    total_worker_compute_time = 0.0
    
    # Distribuir trabajo inicial
    num_workers = size - 1
    batch_size = max(10, len(work_queue) // num_workers) if num_workers > 0 else len(work_queue)
    
    for worker_id in range(1, size):
        if work_queue:
            batch = []
            for _ in range(min(batch_size, len(work_queue))):
                batch.append(heapq.heappop(work_queue))
            comm.send((batch, best_cost), dest=worker_id)
    
    active_workers = size - 1
    workers_with_work = set(range(1, size))
    
    # Loop principal del master
    while active_workers > 0:
        # Recibir resultado de cualquier worker
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker_id = status.Get_source()
        
        if result[0] == "STATS":
            # Worker terminando, actualizar estadísticas
            _, nodes, worker_comp_time = result
            total_nodes_explored += nodes
            total_worker_compute_time += worker_comp_time
            continue
            
        _, solutions, worker_best, new_nodes, nodes_explored = result
        total_nodes_explored += nodes_explored
        
        # Procesar resultados (esto es cómputo del master)
        comp_start = time.time()
        
        # Actualizar mejor solución
        for cost, path in solutions:
            if cost < best_cost:
                best_cost = cost
                best_solution = (cost, path)
                if num_cities <= 20:  # Solo imprimir para problemas pequeños
                    print(f"Nueva mejor solución encontrada: {cost}")
                
                # Broadcast del nuevo mejor costo a todos los workers
                for w in range(1, size):
                    if w != worker_id and w in workers_with_work:
                        comm.send(([], best_cost), dest=w, tag=1)
        
        # Agregar nuevos nodos a la cola
        for node in new_nodes:
            if node.cost < best_cost:
                heapq.heappush(work_queue, node)
        
        master_compute_time += time.time() - comp_start
        
        # Enviar más trabajo o terminar worker
        if work_queue:
            # Enviar batch de trabajo
            batch = []
            batch_size = min(50, max(10, len(work_queue) // active_workers)) if active_workers > 0 else 50
            for _ in range(min(batch_size, len(work_queue))):
                batch.append(heapq.heappop(work_queue))
            comm.send((batch, best_cost), dest=worker_id)
        else:
            # No hay más trabajo para este worker
            workers_with_work.discard(worker_id)
            
            # Si nadie tiene trabajo, terminar
            if not workers_with_work:
                for w in range(1, size):
                    comm.send("TERMINATE", dest=w)
                active_workers = 0
            else:
                # Enviar lista vacía para que el worker espere
                comm.send(([], best_cost), dest=worker_id)
    
    # Tiempo total de cómputo = master + todos los workers
    total_compute_time = master_compute_time + total_worker_compute_time
    
    return best_solution, total_compute_time, total_nodes_explored

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if len(sys.argv) < 2:
        if rank == 0:
            print("Uso: mpirun -n <num_procesos> python tsp_mpi_branch_bound.py <num_ciudades>")
            print("Nota: Se requieren al menos 2 procesos (1 master + 1 worker)")
        sys.exit(1)
    
    num_cities = int(sys.argv[1])
    
    # Validar número de procesos
    if size < 2 and rank == 0:
        print("ADVERTENCIA: Ejecutando con un solo proceso. Se recomienda usar al menos 2 procesos.")
        print("Ejemplo: mpirun -n 4 python tsp_mpi_branch_bound.py 10")
    
    if rank == 0:
        print(f"Iniciando TSP con {num_cities} ciudades y {size} procesos")
        if size == 1:
            print("Ejecutando versión secuencial (no paralela)")
        
        # INICIO DE LA MEDICIÓN DEL TIEMPO TOTAL
        total_start_time = time.time()
        
        # Generar matriz con seed fija para reproducibilidad
        matrix = generate_random_matrix(num_cities, seed=42)
        
        # Imprimir la matriz si tiene 10 ciudades o menos
        # if num_cities <= 10:
        print_matrix(matrix)
        
        # Enviar matriz a todos los workers (si hay)
        for i in range(1, size):
            comm.send(matrix, dest=i)
        
        # Resolver el problema TSP
        solution, compute_time, nodes_explored = parallel_tsp_master(matrix, comm, size, num_cities)
        
        # FIN DE LA MEDICIÓN DEL TIEMPO TOTAL
        total_time = time.time() - total_start_time
        
        # Tiempo de comunicación = tiempo total - tiempo de cómputo
        comm_time = total_time - compute_time
        
        if solution:
            print_solution(solution[0], solution[1])
            print(f"Estadísticas de tiempo:")
            print(f"  - Tiempo total: {total_time:.4f}s")
            print(f"  - Tiempo de cómputo: {compute_time:.4f}s ({(compute_time/total_time*100):.2f}%)")
            print(f"  - Tiempo de comunicación: {comm_time:.4f}s ({(comm_time/total_time*100):.2f}%)")
            print(f"\nOtras estadísticas:")
            print(f"  - Nodos explorados: {nodes_explored}")
            cost = solution[0]
            status = "completed"

        else:
            print("No se encontró solución")
            cost = INF
            status = "failed"
        
        result_data = {
            'n': num_cities,
            'p': size,
            'total_time': total_time,
            'comp_time': compute_time,
            'comm_time': comm_time,
            'cost': cost,
            'status': status,
            'nodes_explored': nodes_explored
        }
        save_result(result_data)
    
    else:
        # Worker process
        matrix = comm.recv(source=0)
        parallel_tsp_worker(matrix, comm, rank)

if __name__ == "__main__":
    main()