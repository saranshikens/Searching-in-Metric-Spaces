#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>
#include <iostream>

// Global counter tracking precise metric distance evaluations
extern long long distance_count;

struct Object {
    std::vector<double> coordinates;
};

// Euclidean distance implementation tracking total operations
inline double continuous_distance(const Object& a, const Object& b) {
    distance_count++;
    double sum = 0.0;
    size_t dim = a.coordinates.size();
    for (size_t i = 0; i < dim; ++i) {
        double diff = a.coordinates[i] - b.coordinates[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// LAESA index layout structure
struct LAESAIndex {
    int k_pivots;
    std::vector<int> pivot_indices;               // Maps index of pivot in full dataset
    std::vector<Object> pivots;                   // Deep copy of pivot coordinates
    std::vector<std::vector<double>> dist_matrix; // Matrix size: N x k
};

#endif // DATASET_H