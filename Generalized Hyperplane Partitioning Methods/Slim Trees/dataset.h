#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>
#include <iostream>

// Global metrics tracker for absolute distance function queries
extern long long distance_count;

struct Object {
    int id;
    std::vector<double> coordinates;
};

// Euclidean distance calculator that automatically increments global counters
inline double euclidean_distance(const Object& a, const Object& b) {
    distance_count++;
    double sum = 0.0;
    size_t dim = a.coordinates.size();
    for (size_t i = 0; i < dim; ++i) {
        double diff = a.coordinates[i] - b.coordinates[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Slim Tree Node Structure
struct STNode {
    bool is_leaf;
    
    // Internal node properties: pivots and covering radii for child branches
    Object left_pivot;
    Object right_pivot;
    double left_radius;
    double right_radius;
    
    STNode* left;
    STNode* right;
    
    // Leaf node properties: local container for bounded elements
    std::vector<Object> objects;
    
    STNode() : is_leaf(true), left(nullptr), right(nullptr), left_radius(0.0), right_radius(0.0) {}
    ~STNode() {
        delete left;
        delete right;
    }
};

#endif // DATASET_H