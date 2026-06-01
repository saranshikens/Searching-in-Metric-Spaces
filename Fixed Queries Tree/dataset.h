#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <unordered_map>
#include <cmath>
#include <iostream>

// Global counter to explicitly track metric distance computations
extern long long distance_count;

struct Object {
    std::vector<int> features;
};

// Manhattan Distance (L1 Norm) to yield clean discrete integer steps (i >= 0)
inline int discrete_distance(const Object& a, const Object& b) {
    distance_count++;
    int dist = 0;
    for (size_t i = 0; i < a.features.size(); ++i) {
        dist += std::abs(a.features[i] - b.features[i]);
    }
    return dist;
}

struct FQTNode {
    bool is_leaf;
    int level;
    Object level_pivot;                         // The shared pivot chosen for this specific tree depth
    std::vector<Object> objects;                // Occupied ONLY if is_leaf == true
    std::unordered_map<int, FQTNode*> children; // Distance branches mapping integer distances to subtrees

    FQTNode() : is_leaf(false), level(0) {}
};

#endif