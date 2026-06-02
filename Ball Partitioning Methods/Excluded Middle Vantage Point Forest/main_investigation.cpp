#include "dataset.h"
#include <iostream>
#include <chrono>

// Forward linking declarations
std::vector<Point> generate_dataset(int num_points, int dimensions);
std::vector<VPFTree> build_vpf_forest(std::vector<Point> dataset, double rho);
std::vector<Point> search_vpf_forest(const std::vector<VPFTree>& forest, const Point& q, double r, double rho);

long long distance_count = 0;

void execute_profile_run(int N, int D, double rho, double r) {
    std::cout << "--------------------------------------------------------\n";
    std::cout << "DATASET: N = " << N << " elements | Dimensions = " << D << "D\n";
    std::cout << "CONFIG : Exclusion Margin (rho) = " << rho << " | Query Rad (r) = " << r << "\n";

    // 1. Generate Dataset
    auto dataset = generate_dataset(N, D);

    // 2. Profile Build Phase
    distance_count = 0;
    auto start_build = std::chrono::high_resolution_clock::now();
    auto forest = build_vpf_forest(dataset, rho);
    auto end_build = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_build - start_build).count();

    std::cout << " -> Forest Depth Created      : " << forest.size() << " separate trees\n";
    std::cout << " -> Construction Duration     : " << build_duration << " microseconds\n";
    std::cout << " -> Build Distance Metrics    : " << distance_count << " calls\n";

    // 3. Profile Search Phase (Create a test query point)
    Point query_point = dataset[dataset.size() / 2];
    for (double& coord : query_point.coords) {
        coord += 12.5; // Offset slightly to guarantee a dynamic search scenario
    }

    distance_count = 0;
    auto start_search = std::chrono::high_resolution_clock::now();
    auto matches = search_vpf_forest(forest, query_point, r, rho);
    auto end_search = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search).count();

    std::cout << " -> Query Search Duration     : " << search_duration << " microseconds\n";
    std::cout << " -> Search Distance Metrics   : " << distance_count << " calls\n";
    std::cout << " -> Output Set Size           : " << matches.size() << " elements\n";
}

int main() {
    std::vector<int> sample_sizes = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};
    std::vector<int> dimensions_to_test = {2, 5, 10, 25, 50, 100, 200, 400, 800};
    
    double rho = 25.0; // Fixed exclusion width
    double r = 15.0;   // Query radius (kept less than or equal to rho to demonstrate sublinear behavior)

    std::cout << "STARTING EXCLUDED MIDDLE VANTAGE POINT FOREST INVESTIGATION\n";
    for (int N : sample_sizes) {
        for (int D : dimensions_to_test) {
            execute_profile_run(N, D, rho, r);
        }
    }
    return 0;
}