#include "dataset.h"
#include <random>

// Generates continuous point vector instances uniformly distributed within [0, 1]^D
std::vector<Point> generate_dataset(int num_points, int dimensions, unsigned int seed = 42) {
    std::vector<Point> dataset;
    dataset.reserve(num_points);
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    for (int i = 0; i < num_points; ++i) {
        Point p;
        p.coords.reserve(dimensions);
        for (int d = 0; d < dimensions; ++d) {
            p.coords.push_back(dis(gen));
        }
        dataset.push_back(p);
    }
    return dataset;
}