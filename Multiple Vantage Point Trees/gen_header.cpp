#include "dataset.h"
#include <random>

std::vector<Point> generate_dataset(size_t num_points, size_t dimensions) {
    std::vector<Point> dataset;
    std::mt19937 gen(42); // Deterministic seed for benchmarking
    std::uniform_real_distribution<double> dis(0.0, 100.0);

    for (size_t i = 0; i < num_points; ++i) {
        Point p;
        p.id = static_cast<int>(i);
        p.coords.resize(dimensions);
        for (size_t d = 0; d < dimensions; ++d) {
            p.coords[d] = dis(gen);
        }
        dataset.push_back(p);
    }
    return dataset;
}