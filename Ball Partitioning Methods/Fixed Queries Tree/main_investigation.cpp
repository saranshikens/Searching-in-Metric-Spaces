#include "dataset.h"
#include <chrono>
#include <iostream>
#include <iomanip>

// Instantiate global calculation counter
long long distance_count = 0;

// Declarations linking modular components
std::vector<Object> generate_dataset(int num_objects, int dimensions, int max_val = 100);
FQTNode* build_fqt(std::vector<Object>& dataset);
void range_search_fqt(FQTNode* node, const Object& q, int r, std::vector<Object>& results, std::unordered_map<int, int>& level_dist_cache);

void free_memory(FQTNode* node) {
    if (!node) return;
    for (auto& pair : node->children) {
        free_memory(pair.second);
    }
    delete node;
}

void run_investigation_node(int N, int D, int R) {
    std::cout << "=========================================================\n";
    std::cout << "TEST SETUP: N = " << N << " | Dimensions = " << D << " | Range Radius = " << R << "\n";
    std::cout << "=========================================================\n";

    std::vector<Object> dataset = generate_dataset(N, D);
    
    // Evaluate Tree Building
    distance_count = 0;
    auto start_build = std::chrono::high_resolution_clock::now();
    FQTNode* root = build_fqt(dataset);
    auto end_build = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> build_time = end_build - start_build;
    long long build_comps = distance_count;

    // Target query node initialized near spatial midpoint
    Object query;
    query.features.resize(D, 50);

    // Evaluate Range Search
    distance_count = 0;
    std::vector<Object> search_results;
    std::unordered_map<int, int> level_dist_cache;

    auto start_search = std::chrono::high_resolution_clock::now();
    range_search_fqt(root, query, R, search_results, level_dist_cache);
    auto end_search = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> search_time = end_search - start_search;
    long long search_comps = distance_count;

    // Print Performance Report Table
    std::cout << std::left << std::setw(20) << "Phase" 
              << std::setw(20) << "Time (ms)" 
              << "Distance Computations\n";
    std::cout << std::left << std::setw(20) << "Build (FQT)" 
              << std::setw(20) << build_time.count() 
              << build_comps << "\n";
    std::cout << std::left << std::setw(20) << "Range Search" 
              << std::setw(20) << search_time.count() 
              << search_comps << "\n";
    std::cout << "Matches Found: " << search_results.size() << "\n\n";

    free_memory(root);
}

int main() {

    std::vector<int> dimensions = {2, 5, 10, 25, 50, 100, 200, 400, 800};
    std::vector<int> num = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};

    std::cout << ">>> CRITERIA 1: EVALUATING DATASET CAPACITIES (N) <<<\n";
    for(auto& n: num){
        for(auto& dim: dimensions){
            run_investigation_node(n, dim, 20);
        }
    }

    return 0;
}