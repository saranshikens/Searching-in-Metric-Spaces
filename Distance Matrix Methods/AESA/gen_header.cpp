#include "dataset.h"
#include <random>

// Synthesizes continuous multi-dimensional point configurations
std::vector<Point> generate_dataset(int n, int d, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    std::vector<Point> dataset(n);
    for (int i = 0; i < n; ++i) {
        dataset[i].coords.resize(d);
        for (int j = 0; j < d; ++j) {
            dataset[i].coords[j] = dis(gen);
        }
    }
    return dataset;
}