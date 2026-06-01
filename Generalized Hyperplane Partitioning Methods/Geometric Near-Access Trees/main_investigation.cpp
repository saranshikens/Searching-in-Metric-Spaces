#include "dataset.h"
#include <chrono>
#include <iomanip>

// Instantiate global tracker
long long distance_count = 0;

// Declarations from external compilation units
std::vector<std::vector<double>> generate_dataset(int num_points, int dimensions);
GNATNode* build_gnat(std::vector<std::vector<double>> points, size_t m);
void range_search_gnat(GNATNode* node, const std::vector<double>& q, double r, std::vector<std::vector<double>>& results);

int main() {
    std::vector<int> sizes = {500, 1000, 2000};
    std::vector<int> dimensions = {2, 5, 10};
    double query_radius = 0.15;
    size_t m_pivots = 4; // Node branching factor

    std::cout << "=========================================================================\n";
    std::cout << "               GNAT METRIC TREE PERFORMANCE ANALYSIS                     \n";
    std::cout << "=========================================================================\n";
    std::cout << std::left << std::setw(8)  << "Size(N)" 
              << std::setw(8)  << "Dim(D)" 
              << std::setw(16) << "Build Time(us)" 
              << std::setw(14) << "Build Dists" 
              << std::setw(16) << "Search Time(us)" 
              << std::setw(14) << "Search Dists" << "\n";
    std::cout << "-------------------------------------------------------------------------\n";

    for (int N : sizes) {
        for (int D : dimensions) {
            // 1. Data Setup
            auto dataset = generate_dataset(N, D);
            std::vector<double> query = dataset[0]; // Self-target query test

            // 2. Track Build Phase
            distance_count = 0;
            auto start_build = std::chrono::high_resolution_clock::now();
            GNATNode* root = build_gnat(dataset, m_pivots);
            auto end_build = std::chrono::high_resolution_clock::now();
            auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(end_build - start_build).count();
            long long build_dists = distance_count;

            // 3. Track Search Phase
            distance_count = 0;
            std::vector<std::vector<double>> query_results;
            auto start_search = std::chrono::high_resolution_clock::now();
            range_search_gnat(root, query, query_radius, query_results);
            auto end_search = std::chrono::high_resolution_clock::now();
            auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();
            long long search_dists = distance_count;

            // 4. Output Row Metrics
            std::cout << std::left << std::setw(8)  << N 
                      << std::setw(8)  << D 
                      << std::setw(16) << build_time 
                      << std::setw(14) << build_dists 
                      << std::setw(16) << search_time 
                      << std::setw(14) << search_dists << "\n";

            delete root; // Prevent memory leaks
        }
    }
    std::cout << "=========================================================================\n";
    return 0;
}