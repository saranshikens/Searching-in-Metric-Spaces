#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>

// Global metrics tracker for distance computations
extern long long distance_count;

struct Point {
    std::vector<double> coords;
    int id;
};

// Continuous metric space distance evaluation (Euclidean)
inline double continuous_distance(const Point& a, const Point& b) {
    distance_count++;
    double sum = 0.0;
    for (size_t i = 0; i < a.coords.size(); ++i) {
        double diff = a.coords[i] - b.coords[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

struct BSTNode {
    Point p1;
    Point p2;
    double rc1 = 0.0; // Covering radius for p1's subtree
    double rc2 = 0.0; // Covering radius for p2's subtree
    bool has_p1 = false;
    bool has_p2 = false;
    BSTNode* left = nullptr;
    BSTNode* right = nullptr;
};

#endif