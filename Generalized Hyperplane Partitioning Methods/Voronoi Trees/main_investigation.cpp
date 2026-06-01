#include "dataset.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// Component Linking Declarations
std::vector<std::vector<double>> generate_synthetic_dataset(int N, int D);
VTNode* build_vt_tree(std::vector<std::vector<double>> objects);
void range_search_vt(VTNode* node, const std::vector<double>& query, double r, std::vector<std::vector<double>>& results);

// Define global distance counter instance
long long distance_count = 0;

void run_profile_instance(int N, int D, double query_radius) {
    // 1. Data Generation
    auto dataset = generate_synthetic_dataset(N, D);

    // 2. Measure Tree Generation Metrics
    distance_count = 0;
    auto start_build = std::chrono::high_resolution_clock::now();
    VTNode* root = build_vt_tree(dataset);
    auto end_build = std::chrono::high_resolution_clock::now();

    long long build_time = std::chrono::duration_cast<std::chrono::microseconds>(end_build - start_build).count();
    long long build_dists = distance_count;

    // 3. Measure Range Query Search Metrics
    std::vector<double> query = dataset[0]; // Anchor query to a deterministic point in space
    std::vector<std::vector<double>> results;

    distance_count = 0;
    auto start_search = std::chrono::high_resolution_clock::now();
    range_search_vt(root, query, query_radius, results);
    auto end_search = std::chrono::high_resolution_clock::now();

    long long search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();
    long long search_dists = distance_count;

    // Output formatted metrics row
    std::cout << std::setw(8) << N 
              << std::setw(6) << D 
              << std::setw(16) << build_time 
              << std::setw(14) << build_dists 
              << std::setw(17) << search_time 
              << std::setw(14) << search_dists 
              << std::setw(10) << results.size() << "\n";

    delete root; // Cleanup tree allocations
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "                       VORONOI TREE (VT) BENCHMARK PROFILER                     \n";
    std::cout << "================================================================================\n";
    std::cout << std::setw(8) << "Size(N)" 
              << std::setw(6) << "Dim(D)" 
              << std::setw(16) << "Build Time(us)" 
              << std::setw(14) << "Build Dists" 
              << std::setw(17) << "Search Time(us)" 
              << std::setw(14) << "Search Dists" 
              << std::setw(10) << "Matches" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Investigative profiles across altering sizes and spatial dimensions
    std::vector<int> test_sizes = {500, 1000, 2500};
    std::vector<int> test_dimensions = {2, 5, 10};
    double target_radius = 20.0;

    for (int N : test_sizes) {
        for (int D : test_dimensions) {
            run_profile_instance(N, D, target_radius);
        }
    }

    std::cout << "================================================================================\n";
    return 0;
}