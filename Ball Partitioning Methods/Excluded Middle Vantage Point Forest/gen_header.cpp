#include "dataset.h"
#include <random>

std::vector<Point> generate_dataset(int num_points, int dimensions) {
    std::vector<Point> dataset;
    std::mt19937 rng(1337); // Fixed seed for profiling consistency
    std::uniform_real_distribution<double> dist(0.0, 500.0);

    for (int i = 0; i < num_points; ++i) {
        Point p;
        p.id = i;
        p.coords.resize(dimensions);
        for (int d = 0; d < dimensions; ++d) {
            p.coords[d] = dist(rng);
        }
        dataset.push_back(p);
    }
    return dataset;
}