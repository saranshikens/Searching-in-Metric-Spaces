#include "dataset.h"
#include <random>

// Generates an array of multi-dimensional objects with random coordinates
std::vector<Object> generate_dataset(int size, int dimensions, int coordinate_range = 50) {
    std::vector<Object> dataset;
    dataset.reserve(size);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(0, coordinate_range);

    for (int i = 0; i < size; ++i) {
        Object obj;
        obj.id = i;
        obj.coordinates.resize(dimensions);
        for (int d = 0; d < dimensions; ++d) {
            obj.coordinates[d] = distribution(generator);
        }
        dataset.push_back(obj);
    }
    return dataset;
}