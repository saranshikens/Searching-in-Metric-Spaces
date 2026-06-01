#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>
#include <iostream>

struct Point {
    int id;
    std::vector<double> coords;
};

// Global distance computation counter
extern size_t distance_count;

// Continuous metric space distance function (L2 Norm)
inline double euclidean_distance(const Point& a, const Point& b) {
    distance_count++;
    double sum = 0.0;
    for (size_t i = 0; i < a.coords.size(); ++i) {
        double diff = a.coords[i] - b.coords[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

struct MVPTNode {
    bool is_leaf;
    
    // Internal node properties (Two pivots per node)
    Point p1;
    Point p2;
    double dm1;         // Median split for p1
    double dm2_left;    // Median split for p2 over p1's left branch
    double dm2_right;   // Median split for p2 over p1's right branch
    std::vector<MVPTNode*> children; // 4-way fanout allocation
    
    // Leaf node properties
    std::vector<Point> leaf_objects;
    // Precomputed distances from each leaf point to its ancestor path pivots
    std::vector<std::vector<double>> leaf_distances; 

    MVPTNode(bool leaf) : is_leaf(leaf), dm1(0), dm2_left(0), dm2_right(0) {
        children.resize(4, nullptr);
    }
    
    ~MVPTNode() {
        for (auto child : children) {
            if (child) delete child;
        }
    }
};

#endif