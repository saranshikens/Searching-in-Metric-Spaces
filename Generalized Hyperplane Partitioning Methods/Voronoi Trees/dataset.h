#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

// Global counter to track distance metric evaluations
extern long long distance_count;

// Inline function to compute Euclidean distance and increment counter
inline double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    distance_count++;
    double sum = 0.0;
    size_t dim = a.size();
    for (size_t i = 0; i < dim; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Voronoi Tree Node Structure
struct VTNode {
    bool is_leaf;
    std::vector<std::vector<double>> leaf_objects; // Active objects stored if leaf node
    std::vector<std::vector<double>> pivots;       // Holds 2 or 3 routing pivots
    std::vector<double> covering_radii;           // Associated maximum covering radius per pivot
    std::vector<VTNode*> children;                 // Subtrees corresponding to each pivot partition

    VTNode() : is_leaf(false) {}
    ~VTNode() {
        for (auto child : children) {
            delete child;
        }
    }
};

#endif