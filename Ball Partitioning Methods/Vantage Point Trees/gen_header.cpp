#include "dataset.h"
#include <random>

std::vector<Object> generate_dataset(int N, int D) {
    std::vector<Object> dataset;
    std::mt19937 rng(42); // Fixed seed for reproducible benchmarks
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int i = 0; i < N; ++i) {
        Object obj;
        obj.id = i;
        obj.coords.resize(D);
        for (int j = 0; j < D; ++j) {
            obj.coords[j] = dist(rng);
        }
        dataset.push_back(obj);
    }
    return dataset;
}