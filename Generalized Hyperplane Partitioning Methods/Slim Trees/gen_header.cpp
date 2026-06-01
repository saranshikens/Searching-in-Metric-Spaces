#include "dataset.h"
#include <random>

std::vector<Object> generate_dataset(int num_objects, int dimensions, unsigned int seed = 42) {
    std::vector<Object> dataset;
    dataset.reserve(num_objects);
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    for (int i = 0; i < num_objects; ++i) {
        Object obj;
        obj.id = i;
        obj.coordinates.resize(dimensions);
        for (int d = 0; d < dimensions; ++d) {
            obj.coordinates[d] = dis(gen);
        }
        dataset.push_back(obj);
    }
    return dataset;
}