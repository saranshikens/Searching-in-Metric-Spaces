#include <iostream>
#include <chrono>
#include <iomanip>
#include "dataset.h"

// External declarations linking our separate pipeline steps
std::vector<Object> generate_dataset(int size, int dimensions, int coordinate_range = 50);
std::unique_ptr<BKTNode> build_bkt(std::vector<Object>& objects);
void range_search_bkt(const std::unique_ptr<BKTNode>& root, const Object& q, int r, std::vector<Object>& results);

void run_investigation() {
    // Configurations to investigate performance
    std::vector<int> target_sizes = {1000, 5000, 10000, 20000};
    std::vector<int> target_dimensions = {2, 5, 10, 15};
    int search_radius = 12;

    std::cout << "========================================================================\n";
    std::cout << std::setw(10) << "Size (N)" << std::setw(15) << "Dimensions (D)" 
              << std::setw(20) << "Build Time (ms)" << std::setw(20) << "Search Time (ms)" << "\n";
    std::cout << "========================================================================\n";

    for (int size : target_sizes) {
        for (int dim : target_dimensions) {
            
            // 1. Dataset Generation
            auto dataset = generate_dataset(size, dim);
            
            // Define a random target Query object
            Object query = generate_dataset(1, dim)[0];

            // 2. Profile Build Action
            auto start_build = std::chrono::high_resolution_clock::now();
            auto bkt_root = build_bkt(dataset);
            auto end_build = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> build_duration = end_build - start_build;

            // 3. Profile Search Action
            std::vector<Object> query_results;
            auto start_search = std::chrono::high_resolution_clock::now();
            range_search_bkt(bkt_root, query, search_radius, query_results);
            auto end_search = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> search_duration = end_search - start_search;

            // Display logged parameters
            std::cout << std::setw(10) << size 
                      << std::setw(15) << dim 
                      << std::setw(20) << std::fixed << std::setprecision(3) << build_duration.count()
                      << std::setw(20) << std::fixed << std::setprecision(3) << search_duration.count() 
                      << "\n";
        }
        std::cout << "------------------------------------------------------------------------\n";
    }
}

int main() {
    run_investigation();
    return 0;
}