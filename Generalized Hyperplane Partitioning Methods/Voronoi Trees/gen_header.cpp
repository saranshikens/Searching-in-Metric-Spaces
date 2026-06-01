#include "dataset.h"
#include <random>

std::vector<std::vector<double>> generate_synthetic_dataset(int N, int D) {
    std::vector<std::vector<double>> dataset(N, std::vector<double>(D));
    std::mt19937 gen(42); // Deterministic seed for benchmark alignment
    std::uniform_real_distribution<double> dis(0.0, 100.0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            dataset[i][j] = dis(gen);
        }
    }
    return dataset;
}