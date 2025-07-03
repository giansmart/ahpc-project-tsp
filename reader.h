#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <limits> // for std::numeric_limits
#include <queue>
#include <iomanip>

const int INF = std::numeric_limits<int>::max();

template <typename T>
std::vector<std::vector<T>> leerArchivo(const std::string& nombreArchivo) {
    std::vector<std::vector<T>> matriz;
    std::ifstream archivo(nombreArchivo);

    if (!archivo) {
        std::cout << "No se pudo abrir el archivo: " << nombreArchivo << std::endl;
        return matriz;
    }

    std::string linea;
    while (getline(archivo, linea)) {
        std::vector<T> fila;
        std::istringstream iss(linea);
        T valor;

        while (iss >> valor) {
            if(valor == 0)
                fila.push_back(INF);
            else
                fila.push_back(valor);
        }

        matriz.push_back(fila);
    }

    archivo.close();
    return matriz;
}

template <typename T>
void writeToCSV(const std::string& filename, const std::string& inputFile, std::pair<T, double>& data) {
    std::ofstream file("output.csv", std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write data
    // file << data.first << "," << data.second << std::endl;
    file << inputFile << "," << data.first << "," << data.second << std::endl;
    

    file.close();
    std::cout << "Data written to " << filename << std::endl;
}

template <typename T>
void writeMinPath(std::vector<std::pair<T, T>> const &list, T cost) {
    std::ofstream file("minPath.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << std::endl;
        return;
    }

    // Write data
    for (int i = 0; i < list.size(); i++)
        file << list[i].first + 1 << " -> " << list[i].second + 1 << std::endl;
    
    file << "cost: " << cost << std::endl;

    file.close();
    std::cout << "Data written " << std::endl;
}