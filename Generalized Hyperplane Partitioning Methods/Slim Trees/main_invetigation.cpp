#include "dataset.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

// Declare the global tracker instance
long long distance_count = 0;

// External linkages to isolated operational modules
std::vector<Object> generate_dataset(int num_objects, int dimensions, unsigned int seed = 42);
STNode* build_slim_tree(const std::vector<Object>& dataset, int max_leaf_capacity = 4);
std::vector<Object> slim_tree_range_search(STNode* root, const Object& query, double r);

int main() {
    std::vector<int> sizes = {200, 500, 1000};
    std::vector<int> dimensions = {2, 5, 10};
    double query_radius = 0.15;
    
    std::cout << "=========================================================================\n";
    std::cout << "               Slim Tree (ST) Performance Profile Grid                   \n";
    std::cout << "=========================================================================\n";
    std::cout << std::left << std::setw(8) << "N" 
              << std::setw(6) << "Dim" 
              << std::setw(16) << "Build Time(ms)" 
              << std::setw(14) << "Build Dists" 
              << std::setw(16) << "Search Time(ms)" 
              << std::setw(14) << "Search Dists" 
              << "Results Found\n";
    std::cout << "-------------------------------------------------------------------------\n";

    for (int N : sizes) {
        for (int D : dimensions) {
            std::vector<Object> data = generate_dataset(N, D, 54321);
            
            Object query;
            query.id = -1;
            query.coordinates.assign(D, 0.5); // Center point query position
            
            // Profiling Build Step
            distance_count = 0;
            auto start_build = std::chrono::high_resolution_clock::now();
            STNode* root = build_slim_tree(data, 4);
            auto end_build = std::chrono::high_resolution_clock::now();
            
            double build_time = std::chrono::duration<double, std::milli>(end_build - start_build).count();
            long long build_dists = distance_count;
            
            // Profiling Search Step
            distance_count = 0;
            auto start_search = std::chrono::high_resolution_clock::now();
            std::vector<Object> results = slim_tree_range_search(root, query, query_radius);
            auto end_search = std::chrono::high_resolution_clock::now();
            
            double search_time = std::chrono::duration<double, std::milli>(end_search - start_search).count();
            long long search_dists = distance_count;
            
            std::cout << std::left << std::setw(8) << N 
                      << std::setw(6) << D 
                      << std::fixed << std::setprecision(3) << std::setw(16) << build_time 
                      << std::setw(14) << build_dists 
                      << std::setw(16) << search_time 
                      << std::setw(14) << search_dists 
                      << results.size() << "\n";
                      
            delete root; // Clean heap resources
        }
    }
    std::cout << "=========================================================================\n";
    return 0;
}