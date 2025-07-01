import os

def save_result(data: dict, filename: str = "resultados.txt"):
    """
    Guarda un registro en un archivo de texto como CSV.
    
    Args:
        data (dict): Diccionario con claves como columnas y valores como fila.
        headers (list): Lista de columnas en orden.
        filename (str): Nombre del archivo a guardar.
    """
    file_exists = os.path.isfile(filename)
    headers = list(data.keys())

    with open(filename, "a") as f:
        if not file_exists:
            f.write(",".join(headers) + "\n")
            print(f"\nArchivo creado: {filename}\n")
        valores = [f"{data[h]:.4f}" if isinstance(data[h], float) else str(data[h]) for h in headers]
        f.write(",".join(valores) + "\n")

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