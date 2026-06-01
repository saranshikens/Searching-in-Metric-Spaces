#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>
#include <iostream>

// Global counter tracking absolute metric distance evaluations
extern long long distance_count;

// Continuous Euclidean distance function
inline double continuous_distance(const std::vector<double>& a, const std::vector<double>& b) {
    distance_count++;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// GNAT Node definition supporting flexible multi-branching partitions
struct GNATNode {
    bool is_leaf;
    std::vector<std::vector<double>> pivots;       // Contains m pivots
    std::vector<std::vector<double>> table_min;    // m x m min range matrix: rij_l
    std::vector<std::vector<double>> table_max;    // m x m max range matrix: rij_h
    std::vector<GNATNode*> children;               // m child subtrees
    std::vector<std::vector<double>> leaf_points;  // Datapoints stored if leaf

    GNATNode() : is_leaf(false) {}
    ~GNATNode() {
        for (auto child : children) {
            if (child) delete child;
        }
    }
};

#endif