import os

def print_matrix(matrix):
    """Imprime la matriz de adyacencia"""
    print("Matriz de distancias:")
    print("=" * 70)
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