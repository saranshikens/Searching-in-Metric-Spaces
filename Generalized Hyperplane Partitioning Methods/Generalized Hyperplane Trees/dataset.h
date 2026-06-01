#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>

// Global performance counter tracked across execution phases
extern long long distance_count;

struct Point {
    std::vector<double> coords;
};

// Continuous Euclidean distance metric wrapper
inline double distance(const Point& a, const Point& b) {
    distance_count++;
    double sum = 0.0;
    size_t dim = a.coords.size();
    for (size_t i = 0; i < dim; ++i) {
        double diff = a.coords[i] - b.coords[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

struct GHTNode {
    bool is_leaf;
    Point p1;
    Point p2;
    bool has_p1 = false;
    bool has_p2 = false;
    GHTNode* left = nullptr;
    GHTNode* right = nullptr;
    std::vector<Point> objects; // Utilized strictly if is_leaf is true

    // Safe cascading cleanup mapping
    ~GHTNode() {
        delete left;
        delete right;
    }
};

#endif // DATASET_H