#include "dataset.h"
#include <random>

std::vector<Point> generate_dataset(int N, int D, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    std::vector<Point> dataset(N);
    for (int i = 0; i < N; ++i) {
        dataset[i].coords.resize(D);
        for (int j = 0; j < D; ++j) {
            dataset[i].coords[j] = dis(gen);
        }
        dataset[i].id = i;
    }
    return dataset;
}