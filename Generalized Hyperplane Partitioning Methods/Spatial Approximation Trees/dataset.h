#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>

// Global distance computation counter
extern long long distance_count;

struct Point {
    std::vector<double> coords;
    int id;
};

// Continuous metric distance function (Euclidean distance)
inline double euclidean_distance(const Point& a, const Point& b) {
    distance_count++;
    double sum = 0.0;
    size_t dim = a.coords.size();
    for (size_t i = 0; i < dim; ++i) {
        double diff = a.coords[i] - b.coords[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Data generation declaration
std::vector<Point> generate_dataset(int N, int D, int seed = 42);

#endif // DATASET_H