#include "dataset.h"
#include <random>

std::vector<Vector> generate_dataset(int N, int D, int max_val = 100) {
    std::vector<Vector> dataset(N);
    std::mt19937 rng(42); // Fixed seed for reproducible investigation
    std::uniform_int_distribution<int> dist(0, max_val);

    for (int i = 0; i < N; ++i) {
        dataset[i].coords.resize(D);
        for (int j = 0; j < D; ++j) {
            dataset[i].coords[j] = dist(rng);
        }
    }
    return dataset;
}