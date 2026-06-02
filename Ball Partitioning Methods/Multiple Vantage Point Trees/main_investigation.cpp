#include "dataset.h"
#include <chrono>
#include <iostream>
#include <vector>

std::vector<Point> generate_dataset(size_t num_points, size_t dimensions);
MVPTNode* build_mvpt(std::vector<Point>& points, std::vector<Point>& path_pivots, size_t leaf_capacity = 4);
void search_mvpt(MVPTNode* node, const Point& q, double r, 
                 std::vector<double>& path_q_distances, std::vector<Point>& results);

size_t distance_count = 0;

void run_profile(size_t N, size_t D, double radius) {
    std::cout << "==================================================\n";
    std::cout << "Target Parameters: N = " << N << " | Dimensions = " << D << "\n";
    std::cout << "==================================================\n";

    std::vector<Point> dataset = generate_dataset(N, D);
    
    // Build Profiling
    distance_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Point> path_pivots;
    MVPTNode* root = build_mvpt(dataset, path_pivots, 4);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    size_t build_dist = distance_count;

    std::cout << " -> Build Duration          : " << build_time << " us\n";
    std::cout << " -> Build Distance Metrics  : " << build_dist << "\n";

    // Search Profiling
    Point query;
    query.id = -1;
    query.coords.resize(D, 50.0); // Constant focal center query

    distance_count = 0;
    std::vector<Point> results;
    std::vector<double> path_q_distances;
    
    auto t2 = std::chrono::high_resolution_clock::now();
    search_mvpt(root, query, radius, path_q_distances, results);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    size_t search_dist = distance_count;

    std::cout << " -> Search Duration         : " << search_time << " us\n";
    std::cout << " -> Search Distance Metrics : " << search_dist << "\n";
    std::cout << " -> Matches Verified        : " << results.size() << "\n\n";

    delete root;
}

int main() {
    std::vector<size_t> test_sizes = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};
    std::vector<size_t> test_dims = {2, 5, 10, 25, 50, 100, 200, 400, 800};
    double query_radius = 12.5;

    for (size_t n : test_sizes) {
        for (size_t d : test_dims) {
            run_profile(n, d, query_radius);
        }
    }
    return 0;
}