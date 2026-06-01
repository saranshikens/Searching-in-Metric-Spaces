#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>
#include <algorithm>

// Global counter to record the precise number of metric distance calculations
extern long long distance_count;

// Represents a continuous multi-dimensional point vector
struct Point {
    std::vector<double> coords;
};

// Symmetrical matrix storing only the lower triangular portion to optimize memory footprint
struct AESAMatrix {
    int n;
    std::vector<double> distances;

    AESAMatrix(int num_objects) {
        n = num_objects;
        long long size = (long long)n * (n - 1) / 2;
        distances.assign(size, 0.0);
    }

    void set_distance(int i, int j, double dist) {
        if (i == j) return;
        if (i < j) std::swap(i, j);
        long long idx = (long long)i * (i - 1) / 2 + j;
        distances[idx] = dist;
    }

    double get_distance(int i, int j) const {
        if (i == j) return 0.0;
        if (i < j) std::swap(i, j);
        long long idx = (long long)i * (i - 1) / 2 + j;
        return distances[idx];
    }
};

// Continuous metric distance function tracking evaluations automatically
inline double euclidean_distance(const Point& p1, const Point& p2) {
    distance_count++;
    double sum = 0.0;
    size_t dim = p1.coords.size();
    for (size_t i = 0; i < dim; ++i) {
        double diff = p1.coords[i] - p2.coords[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

#endif // DATASET_H