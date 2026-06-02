#include "dataset.h"
#include <chrono>
#include <iomanip>
#include <iostream>

long long distance_count = 0;

std::vector<Object> generate_dataset(int N, int D, int seed = 42);

int main() {
    std::vector<int> sizes = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};
    std::vector<int> dimensions = {2, 5, 10, 25, 50, 100, 200, 400, 800};
    double query_radius = 0.15;

    std::cout << std::setw(8) << "Size(N)" 
              << std::setw(8) << "Dim(D)" 
              << std::setw(15) << "Build Time(ms)" 
              << std::setw(15) << "Build Dists" 
              << std::setw(16) << "Search Time(ms)" 
              << std::setw(15) << "Search Dists" 
              << std::setw(12) << "Results" << "\n";
    std::cout << std::string(89, '-') << "\n";

    for (int N : sizes) {
        for (int D : dimensions) {
            // 1. Dataset Initialization
            std::vector<Object> dataset = generate_dataset(N, D, 42);
            
            // 2. Benchmarking Construction
            distance_count = 0;
            MTreeNode* root = nullptr;
            
            auto start_build = std::chrono::high_resolution_clock::now();
            for (const auto& obj : dataset) {
                insert_m_tree(root, obj);
            }
            auto end_build = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double, std::milli>(end_build - start_build).count();
            long long build_dists = distance_count;

            // 3. Setup Spatial Range Query
            Object query(D, 0.5); // Center point query vector
            std::vector<Object> search_results;

            // 4. Benchmarking Range Search Execution
            distance_count = 0;
            auto start_search = std::chrono::high_resolution_clock::now();
            range_search(root, query, query_radius, 0.0, true, search_results);
            auto end_search = std::chrono::high_resolution_clock::now();
            double search_time = std::chrono::duration<double, std::milli>(end_search - start_search).count();
            long long search_dists = distance_count;

            // 5. Output Row Elements
            std::cout << std::setw(8) << N 
                      << std::setw(8) << D 
                      << std::setw(15) << std::fixed << std::setprecision(2) << build_time 
                      << std::setw(15) << build_dists 
                      << std::setw(16) << search_time 
                      << std::setw(15) << search_dists 
                      << std::setw(12) << search_results.size() << "\n";

            delete root;
        }
    }
    return 0;
}