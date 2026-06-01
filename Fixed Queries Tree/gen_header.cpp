#include "dataset.h"
#include <random>

std::vector<Object> generate_dataset(int num_objects, int dimensions, int max_val = 100) {
    std::vector<Object> dataset;
    std::mt19937 rng(42); // Seeded for consistent baseline evaluation across tests
    std::uniform_int_distribution<int> dist(0, max_val);

    for (int i = 0; i < num_objects; ++i) {
        Object obj;
        obj.features.resize(dimensions);
        for (int d = 0; d < dimensions; ++d) {
            obj.features[d] = dist(rng);
        }
        dataset.push_back(obj);
    }
    return dataset;
}