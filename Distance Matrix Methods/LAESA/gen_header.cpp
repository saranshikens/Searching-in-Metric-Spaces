#include "dataset.h"
#include <random>

std::vector<Object> generate_dataset(int num_objects, int dimensions) {
    std::vector<Object> dataset(num_objects);
    std::mt19937 gen(42); // Fixed seed for reproducible comparative benchmarks
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < num_objects; ++i) {
        dataset[i].coordinates.resize(dimensions);
        for (int d = 0; d < dimensions; ++d) {
            dataset[i].coordinates[d] = dis(gen);
        }
    }
    return dataset;
}