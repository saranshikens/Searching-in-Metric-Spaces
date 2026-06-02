#include "dataset.h"
#include "SAT.h"
#include <iostream>
#include <chrono>
#include <iomanip>

// Instantiate tracking flag
long long distance_count = 0;

void run_profile_test(int N, int D) {
    // Generate synthetic hypercube space
    std::vector<Point> data = generate_dataset(N, D, 101);
    
    // Generate 50 distinct query instances
    std::vector<Point> queries = generate_dataset(50, D, 202);
    double query_radius = 0.15;

    // --- Benchmark Construction ---
    distance_count = 0;
    auto start_build = std::chrono::high_resolution_clock::now();
    SATNode* root = buildSAT(data);
    auto end_build = std::chrono::high_resolution_clock::now();
    
    auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(end_build - start_build).count();
    long long build_dists = distance_count;

    // --- Benchmark Range Query Execution ---
    distance_count = 0;
    auto start_search = std::chrono::high_resolution_clock::now();
    
    size_t total_found = 0;
    for (const auto& q : queries) {
        std::vector<Point> results;
        double d_root = euclidean_distance(q, root->pivot);
        rangeSearchSAT(root, q, query_radius, d_root, d_root, results);
        total_found += results.size();
    }
    
    auto end_search = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();
    
    // Compute averages per individual query
    double avg_search_time = static_cast<double>(search_time) / 50.0;
    double avg_search_dists = static_cast<double>(distance_count) / 50.0;

    // Print standardized profiling rows
    std::cout << std::setw(8) << N << " | "
              << std::setw(5) << D << " | "
              << std::setw(11) << build_time << " | "
              << std::setw(12) << build_dists << " | "
              << std::setw(13) << std::fixed << std::setprecision(1) << avg_search_time << " | "
              << std::setw(13) << avg_search_dists << "\n";

    // Reclaim dynamically allocated graph memory
    delete root;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "                     SAT PROFILER INVESTIGATION INPUTS                  \n";
    std::cout << "========================================================================\n";
    std::cout << "   Size  |  Dim  | Build Time  | Build Dists  | Avg Query Time | Avg Query Dists\n";
    std::cout << "    (N)  |  (D)  |    (us)     |   (Counts)   |      (us)      |    (Counts)   \n";
    std::cout << "------------------------------------------------------------------------\n";

    // Execution grid sweeps
    std::vector<int> sizes = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};
    std::vector<int> dimensions = {2, 5, 10, 25, 50, 100, 200, 400, 800};

    for (int N : sizes) {
        for (int D : dimensions) {
            run_profile_test(N, D);
        }
        std::cout << "------------------------------------------------------------------------\n";
    }
    return 0;
}