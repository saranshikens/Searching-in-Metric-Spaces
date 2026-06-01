#include "dataset.h"
#include <chrono>
#include <iostream>
#include <iomanip>

// Instantiate global distance tracker
long long distance_count = 0;

// External module link interfaces
std::vector<Object> generate_dataset(int num_objects, int dimensions);
LAESAIndex build_laesa(const std::vector<Object>& dataset, int k);
std::vector<Object> range_search_laesa(const LAESAIndex& index, const std::vector<Object>& dataset, const Object& query, double radius);

int main() {
    // Array testing sweeps requested for deep investigation
    std::vector<int> N_sizes = {1000, 2000, 5000};
    std::vector<int> D_dims = {2, 5, 10};
    int k_fixed_pivots = 16; 
    double query_radius = 0.15;

    std::cout << "=================================================================================\n";
    std::cout << "               LAESA METRIC SPACE STRUCTURAL PERFORMANCE PROFILE                 \n";
    std::cout << "=================================================================================\n";
    std::cout << std::left << std::setw(8)  << "Size(N)" 
              << std::setw(8)  << "Dim(D)" 
              << std::setw(15) << "Build Time(us)" 
              << std::setw(15) << "Build Dists" 
              << std::setw(16) << "Avg Srch(us)" 
              << std::setw(15) << "Avg Srch Dists" << "\n";
    std::cout << "---------------------------------------------------------------------------------\n";

    for (int N : N_sizes) {
        for (int D : D_dims) {
            // 1. Data Generation Phase
            std::vector<Object> dataset = generate_dataset(N, D);

            // 2. Build Pipeline Profile
            distance_count = 0;
            auto start_build = std::chrono::high_resolution_clock::now();
            LAESAIndex index = build_laesa(dataset, k_fixed_pivots);
            auto end_build = std::chrono::high_resolution_clock::now();
            auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(end_build - start_build).count();
            long long build_dists = distance_count;

            // 3. Search Pipeline Profile (Averaged across 50 independent query objects)
            int query_count = 50;
            long long total_search_time = 0;
            long long total_search_dists = 0;

            std::vector<Object> queries = generate_dataset(query_count, D);

            for (int q = 0; q < query_count; ++q) {
                distance_count = 0;
                auto start_search = std::chrono::high_resolution_clock::now();
                auto results = range_search_laesa(index, dataset, queries[q], query_radius);
                auto end_search = std::chrono::high_resolution_clock::now();
                
                total_search_time += std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();
                total_search_dists += distance_count;
            }

            double avg_search_time = static_cast<double>(total_search_time) / query_count;
            double avg_search_dists = static_cast<double>(total_search_dists) / query_count;

            std::cout << std::left << std::setw(8)  << N 
                      << std::setw(8)  << D 
                      << std::setw(15) << build_time 
                      << std::setw(15) << build_dists 
                      << std::setw(16) << std::fixed << std::setprecision(2) << avg_search_time 
                      << std::setw(15) << std::setprecision(1) << avg_search_dists << "\n";
        }
    }
    std::cout << "=================================================================================\n";
    return 0;
}