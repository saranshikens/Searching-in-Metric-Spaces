#include "dataset.h"
#include <iostream>
#include <iomanip>

// External function linkers
std::vector<Vector> generate_dataset(int N, int D, int max_val = 100);
std::vector<FQAElement> build_FQA(const std::vector<Vector>& dataset, const std::vector<Vector>& pivots, int h);
std::vector<Vector> range_search_FQA(const std::vector<FQAElement>& fqa, const Vector& q, int r,
                                     const std::vector<Vector>& pivots, int h);

long long distance_count = 0;

void run_experiment(int N, int D, int h, int r) {
    std::cout << "---------------------------------------------------------\n";
    std::cout << "CONFIG: Size (N)=" << N << " | Dim (D)=" << D << " | Pivots (h)=" << h << " | Radius (r)=" << r << "\n";
    std::cout << "---------------------------------------------------------\n";

    std::vector<Vector> dataset = generate_dataset(N, D);

    // Pick first h unique items from dataset to behave as level pivots
    std::vector<Vector> pivots(h);
    for (int i = 0; i < h; ++i) pivots[i] = dataset[i % N];

    // --- BUILD PROCESS ---
    distance_count = 0;
    auto start_build = std::chrono::high_resolution_clock::now();
    std::vector<FQAElement> fqa = build_FQA(dataset, pivots, h);
    auto end_build = std::chrono::high_resolution_clock::now();
    
    auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(end_build - start_build).count();
    long long build_comps = distance_count;

    std::cout << "  [BUILD]  Time: " << std::setw(6) << build_time << " us | Distance Comps: " << build_comps << "\n";

    // --- SEARCH PROCESS ---
    int test_queries = 10;
    std::vector<Vector> queries = generate_dataset(test_queries, D, 120); // offset query coordinates
    
    long long total_search_time = 0;
    long long total_search_comps = 0;

    for (int i = 0; i < test_queries; ++i) {
        distance_count = 0;
        auto start_search = std::chrono::high_resolution_clock::now();
        auto results = range_search_FQA(fqa, queries[i], r, pivots, h);
        auto end_search = std::chrono::high_resolution_clock::now();

        total_search_time += std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();
        total_search_comps += distance_count;
    }

    std::cout << "  [SEARCH] Avg Time: " << std::setw(6) << (total_search_time / test_queries) 
              << " us | Avg Distance Comps: " << (total_search_comps / test_queries) << "\n\n";
}

int main() {
    std::vector<int> sample_sizes = {1000, 5000};
    std::vector<int> dimensional_steps = {2, 5, 10};
    int total_pivots = 6; 
    int search_radius = 20;

    std::cout << "=== INVESTIGATING FIXED QUERIES ARRAY (FQA) PERFORMANCE ===\n\n";
    for (int N : sample_sizes) {
        for (int D : dimensional_steps) {
            run_experiment(N, D, total_pivots, search_radius);
        }
    }
    return 0;
}