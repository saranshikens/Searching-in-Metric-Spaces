#include "dataset.h"
#include <random>

std::vector<Object> generate_dataset(int N, int D, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    std::vector<Object> data(N, Object(D));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            data[i][j] = dis(gen);
        }
    }
    return data;
}