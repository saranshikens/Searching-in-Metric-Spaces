#include "dataset.h"
#include <random>

std::vector<Point> generate_dataset(int num_points, int dimensions, int seed = 42) {
    std::vector<Point> dataset;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 100.0);

    for (int i = 0; i < num_points; ++i) {
        Point p;
        p.id = i;
        p.coords.resize(dimensions);
        for (int d = 0; d < dimensions; ++d) {
            p.coords[d] = dis(gen);
        }
        dataset.push_back(p);
    }
    return dataset;
}