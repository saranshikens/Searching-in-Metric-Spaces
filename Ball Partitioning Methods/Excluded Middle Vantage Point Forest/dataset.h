#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>
#include <iostream>

// Global counter for absolute metric tracking
extern long long distance_count;

struct Point {
    std::vector<double> coords;
    int id;
};

// Continuous Euclidean Distance function
inline double continuous_distance(const Point& a, const Point& b) {
    distance_count++;
    double sum = 0;
    for (size_t i = 0; i < a.coords.size(); ++i) {
        double diff = a.coords[i] - b.coords[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Tree Node carrying the vantage point and middle metric splits
struct VPFNode {
    Point pivot;
    double dm;
    bool is_leaf;
    std::vector<Point> leaf_points; 
    VPFNode* left;
    VPFNode* right;

    VPFNode() : dm(0.0), is_leaf(false), left(nullptr), right(nullptr) {}
};

struct VPFTree {
    VPFNode* root;
    VPFTree() : root(nullptr) {}
};

#endif