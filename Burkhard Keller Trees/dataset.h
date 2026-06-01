#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>
#include <unordered_map>
#include <memory>

// Global counter to track distance computations across build and search phases
extern long long distance_count;

struct Object {
    std::vector<int> coordinates;
    int id;
};

struct BKTNode {
    Object pivot;
    std::unordered_map<int, std::unique_ptr<BKTNode>> children;

    BKTNode(Object p) : pivot(p) {}
};

// Discrete distance metric: Manhattan Distance (L1 Norm)
inline int discrete_distance(const Object& a, const Object& b) {
    distance_count++; // Track every distance metric invocation
    int dist = 0;
    size_t dim = a.coordinates.size();
    for (size_t i = 0; i < dim; ++i) {
        dist += std::abs(a.coordinates[i] - b.coordinates[i]);
    }
    return dist;
}

#endif // DATASET_H