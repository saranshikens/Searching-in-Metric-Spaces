#include "dataset.h"
#include <chrono>
#include <iostream>

long long distance_count = 0;

// Declarations linking modules
std::vector<Object> generate_dataset(int N, int D);
VPTNode* build_vpt(const std::vector<Object>& dataset, int leaf_capacity);
std::vector<Object> range_search_vpt(VPTNode* root, const Object& q, double r);

void run_experiment(int N, int D, double r) {
    std::cout << "========================================================\n";
    std::cout << "Profile Target: Size (N) = " << N << " | Dimensions (D) = " << D << " | Range (r) = " << r << "\n";
    std::cout << "========================================================\n";

    // 1. Data Generation
    std::vector<Object> dataset = generate_dataset(N, D);

    // 2. Build Benchmark
    distance_count = 0;
    auto start_build = std::chrono::high_resolution_clock::now();
    VPTNode* root = build_vpt(dataset, 5); // Leaf node capacity set to 5
    auto end_build = std::chrono::high_resolution_clock::now();
    
    auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(end_build - start_build).count();
    long long build_ops = distance_count;

    std::cout << "  [BUILD] Time Elasped : " << build_time << " us\n";
    std::cout << "  [BUILD] Dist Computes: " << build_ops << "\n";

    // 3. Search Benchmark
    Object query;
    query.id = -1;
    query.coords.resize(D, 50.0); // Target centrally located coordinates

    distance_count = 0;
    auto start_search = std::chrono::high_resolution_clock::now();
    std::vector<Object> results = range_search_vpt(root, query, r);
    auto end_search = std::chrono::high_resolution_clock::now();
    
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();
    long long search_ops = distance_count;

    std::cout << "  [SEARCH] Time Elapsed : " << search_time << " us\n";
    std::cout << "  [SEARCH] Dist Computes: " << search_ops << "\n";
    std::cout << "  [SEARCH] Items Found  : " << results.size() << "\n\n";

    delete root; // Clean heap allocation memory
}

int main() {
    std::vector<int> number = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};
    std::vector<int> dimensions = {2, 5, 10, 25, 50, 100, 200, 400, 800};

    for(int& n: number){
        for(int& d: dimensions){
            run_experiment(n, d, 12.0);
        }
    }
    return 0;
}