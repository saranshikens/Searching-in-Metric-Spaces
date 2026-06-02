#include "dataset.h"
#include <iostream>
#include <chrono>
#include <iomanip>

// Definition of the tracking instance
long long distance_count = 0;

// Link declarations
std::vector<Point> generate_dataset(int n, int d, unsigned int seed = 42);
AESAMatrix build_AESA(const std::vector<Point>& dataset);
std::vector<int> search_AESA(const std::vector<Point>& dataset, const AESAMatrix& matrix, const Point& q, double r);

int main() {
    std::vector<int> sizes = {1000, 5000, 10000};
    std::vector<int> dimensions = {2, 5, 10, 25, 50, 100, 200, 400, 800};
    double query_radius = 0.15;
    int num_queries = 20;

    std::cout << "========================================================================\n";
    std::cout << "               AESA SYSTEM PROFILE EMPIRICAL RUNNER                     \n";
    std::cout << "========================================================================\n";
    std::cout << std::left << std::setw(8)  << "N"
              << std::setw(8)  << "Dim"
              << std::setw(15) << "Build Time(us)"
              << std::setw(15) << "Build Dists"
              << std::setw(15) << "Avg Search(us)"
              << std::setw(15) << "Avg Search Dists" << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    for (int n : sizes) {
        for (int d : dimensions) {
            // Data generation
            std::vector<Point> dataset = generate_dataset(n, d);

            // Profile indexing build phase
            distance_count = 0;
            auto build_start = std::chrono::high_resolution_clock::now();
            AESAMatrix matrix = build_AESA(dataset);
            auto build_end = std::chrono::high_resolution_clock::now();
            
            long long build_time = std::chrono::duration_cast<std::chrono::microseconds>(build_end - build_start).count();
            long long build_dists = distance_count;

            // Profile query search phase (Averaged to ensure consistency)
            std::vector<Point> queries = generate_dataset(num_queries, d, 999);
            long long total_search_time = 0;
            long long total_search_dists = 0;

            for (const auto& q : queries) {
                distance_count = 0;
                auto search_start = std::chrono::high_resolution_clock::now();
                auto results = search_AESA(dataset, matrix, q, query_radius);
                auto search_end = std::chrono::high_resolution_clock::now();

                total_search_time += std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count();
                total_search_dists += distance_count;
            }

            double avg_search_time = (double)total_search_time / num_queries;
            double avg_search_dists = (double)total_search_dists / num_queries;

            std::cout << std::left << std::setw(8)  << n
                      << std::setw(8)  << d
                      << std::setw(15) << build_time
                      << std::setw(15) << build_dists
                      << std::setw(15) << std::fixed << std::setprecision(2) << avg_search_time
                      << std::setw(15) << std::fixed << std::setprecision(2) << avg_search_dists << "\n";
        }
        std::cout << "------------------------------------------------------------------------\n";
    }

    return 0;
}