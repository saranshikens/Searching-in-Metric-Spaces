#include <iostream>
#include <chrono>
#include <iomanip>
#include "dataset.h"

// Define the global counter allocation instance
long long distance_count = 0;

std::vector<Object> generate_dataset(int size, int dimensions, int coordinate_range = 50);
std::unique_ptr<BKTNode> build_bkt(std::vector<Object>& objects);
void range_search_bkt(const std::unique_ptr<BKTNode>& root, const Object& q, int r, std::vector<Object>& results);

void run_investigation() {
    std::vector<int> target_sizes = {1000, 5000, 10000, 20000};
    std::vector<int> target_dimensions = {2, 5, 10, 15};
    int search_radius = 12;

    std::cout << "========================================================================================================\n";
    std::cout << std::setw(10) << "Size (N)" << std::setw(12) << "Dims (D)" 
              << std::setw(18) << "Build Time(ms)" << std::setw(16) << "Build Comps"
              << std::setw(18) << "Search Time(ms)" << std::setw(16) << "Search Comps" << "\n";
    std::cout << "========================================================================================================\n";

    for (int size : target_sizes) {
        for (int dim : target_dimensions) {
            
            auto dataset = generate_dataset(size, dim);
            Object query = generate_dataset(1, dim)[0];

            // --- Profile Tree Building ---
            distance_count = 0; // Reset tracking counter
            auto start_build = std::chrono::high_resolution_clock::now();
            auto bkt_root = build_bkt(dataset);
            auto end_build = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> build_duration = end_build - start_build;
            long long build_comps = distance_count;

            // --- Profile Range Query Searching ---
            std::vector<Object> query_results;
            distance_count = 0; // Reset tracking counter
            auto start_search = std::chrono::high_resolution_clock::now();
            range_search_bkt(bkt_root, query, search_radius, query_results);
            auto end_search = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> search_duration = end_search - start_search;
            long long search_comps = distance_count;

            // Print the combined empirical data row
            std::cout << std::setw(10) << size 
                      << std::setw(12) << dim 
                      << std::setw(18) << std::fixed << std::setprecision(3) << build_duration.count()
                      << std::setw(16) << build_comps
                      << std::setw(18) << std::fixed << std::setprecision(3) << search_duration.count()
                      << std::setw(16) << search_comps << "\n";
        }
        std::cout << "--------------------------------------------------------------------------------------------------------\n";
    }
}

int main() {
    run_investigation();
    return 0;
}