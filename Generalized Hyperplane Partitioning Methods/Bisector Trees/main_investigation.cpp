#include "dataset.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>

// Instantiate global metric counter
long long distance_count = 0;

// Internal external linking symbols
std::vector<Point> generate_dataset(int num_points, int dimensions, int seed = 42);
BSTNode* build_bst_recursive(std::vector<Point>& points);
void bst_range_search(BSTNode* node, const Point& query, double r, std::vector<Point>& results);
void free_bst(BSTNode* node);

int main() {
    std::vector<int> sizes = {1000, 5000, 10000};
    std::vector<int> dimensions = {2, 5, 10};
    double query_radius = 12.5;

    std::cout << std::left 
              << std::setw(8)  << "Size(N)" 
              << std::setw(8)  << "Dim(D)" 
              << std::setw(16) << "Build Time(ms)" 
              << std::setw(16) << "Build Dists" 
              << std::setw(16) << "Search Time(ms)" 
              << std::setw(16) << "Search Dists" 
              << std::setw(12) << "Matches" << "\n";
    std::cout << std::string(95, '-') << "\n";

    for (int N : sizes) {
        for (int D : dimensions) {
            std::vector<Point> dataset = generate_dataset(N, D);
            
            // Profile Build Step
            distance_count = 0;
            auto start_build = std::chrono::high_resolution_clock::now();
            BSTNode* root = build_bst_recursive(dataset);
            auto end_build = std::chrono::high_resolution_clock::now();
            
            double build_time = std::chrono::duration<double, std::milli>(end_build - start_build).count();
            long long build_dists = distance_count;

            // Anchor a query location in middle-ground space
            Point query;
            query.id = -1;
            query.coords.resize(D, 50.0);

            // Profile Search Step
            distance_count = 0;
            std::vector<Point> results;
            auto start_search = std::chrono::high_resolution_clock::now();
            bst_range_search(root, query, query_radius, results);
            auto end_search = std::chrono::high_resolution_clock::now();

            double search_time = std::chrono::duration<double, std::milli>(end_search - start_search).count();
            long long search_dists = distance_count;

            std::cout << std::left 
                      << std::setw(8)  << N 
                      << std::setw(8)  << D 
                      << std::setw(16) << build_time 
                      << std::setw(16) << build_dists 
                      << std::setw(16) << search_time 
                      << std::setw(16) << search_dists 
                      << std::setw(12) << results.size() << "\n";

            free_bst(root);
        }
    }
    return 0;
}