#include "dataset.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

// Define global distance tracker instance
long long distance_count = 0;

// Module Declarations
std::vector<Point> generate_dataset(int num_points, int dimensions, unsigned int seed);
GHTNode* build_ght(std::vector<Point>& points, int leaf_capacity);
void range_search_ght(GHTNode* node, const Point& q, double r, std::vector<Point>& results);

int main() {
    std::vector<int> sizes = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};
    std::vector<int> dimensions = {2, 5, 10, 25, 50, 100, 200, 400, 800};
    
    double query_radius = 0.15;
    int num_queries = 10;

    std::cout << "=========================================================================================\n";
    std::cout << "                       GHT METRIC SPACE STRUCTURAL PROFILE REPORT                        \n";
    std::cout << "=========================================================================================\n";
    std::cout << std::left << std::setw(10) << "Size (N)"
              << std::setw(10) << "Dim (D)"
              << std::setw(16) << "Build Time(ms)"
              << std::setw(14) << "Build Dists"
              << std::setw(18) << "Avg Search(ms)"
              << std::setw(14) << "Avg Search Dists" << "\n";
    std::cout << "-----------------------------------------------------------------------------------------\n";

    for (int N : sizes) {
        for (int D : dimensions) {
            // Generate synthetic testing space
            std::vector<Point> dataset = generate_dataset(N, D, 54321);

            // Measure Tree Building Metrics
            distance_count = 0;
            auto start_build = std::chrono::high_resolution_clock::now();
            GHTNode* root = build_ght(dataset, 4);
            auto end_build = std::chrono::high_resolution_clock::now();
            
            double build_time = std::chrono::duration<double, std::milli>(end_build - start_build).count();
            long long build_dists = distance_count;

            // Generate isolated target query points
            std::vector<Point> queries = generate_dataset(num_queries, D, 98765);

            // Measure Tree Searching Metrics
            distance_count = 0;
            auto start_search = std::chrono::high_resolution_clock::now();
            for (const auto& q : queries) {
                std::vector<Point> results;
                range_search_ght(root, q, query_radius, results);
            }
            auto end_search = std::chrono::high_resolution_clock::now();

            double total_search_time = std::chrono::duration<double, std::milli>(end_search - start_search).count();
            double avg_search_time = total_search_time / num_queries;
            double avg_search_dists = static_cast<double>(distance_count) / num_queries;

            // Display Profiler Row
            std::cout << std::left << std::setw(10) << N
                      << std::setw(10) << D
                      << std::fixed << std::setprecision(2)
                      << std::setw(16) << build_time
                      << std::setw(14) << build_dists
                      << std::setw(18) << avg_search_time
                      << std::setw(14) << static_cast<long long>(avg_search_dists) << "\n";

            delete root; // Cascade cleanup
        }
    }
    std::cout << "=========================================================================================\n";
    return 0;
}