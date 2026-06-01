#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>
#include <iostream>

// Global distance tracking counter
extern long long distance_count;

struct Object {
    int id;
    std::vector<double> coords;
};

// Continuous Metric: Euclidean Distance (L2 Norm)
inline double euclidean_distance(const Object& a, const Object& b) {
    distance_count++;
    double sum = 0.0;
    for (size_t i = 0; i < a.coords.size(); ++i) {
        double diff = a.coords[i] - b.coords[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Wrapper structure to preserve precomputed historical pivot distances for VPs pruning
struct DistanceObject {
    Object obj;
    std::vector<double> ancestor_distances; // Stores d(p_i, o) for all ancestors
};

struct VPTNode {
    bool is_leaf;
    
    // Internal Node fields
    Object pivot;
    double median;
    VPTNode* left;
    VPTNode* right;
    
    // Leaf Node fields
    std::vector<DistanceObject> leaf_objects;

    VPTNode() : is_leaf(false), median(0.0), left(nullptr), right(nullptr) {}
    ~VPTNode() {
        delete left;
        delete right;
    }
};

#endif