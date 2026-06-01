#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

// Global distance computation counter
extern long long distance_count;

struct Vector {
    std::vector<int> coords;
};

struct FQAElement {
    Vector obj;
    std::vector<int> dists; // Sequence of h distances to pivots
};

// Discrete Manhattan distance metric
inline int discrete_distance(const Vector& a, const Vector& b) {
    distance_count++;
    int dist = 0;
    for (size_t i = 0; i < a.coords.size(); ++i) {
        dist += std::abs(a.coords[i] - b.coords[i]);
    }
    return dist;
}

#endif