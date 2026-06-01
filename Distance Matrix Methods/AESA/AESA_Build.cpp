#include "dataset.h"

// Builds the AESA matrix index structure
AESAMatrix build_AESA(const std::vector<Point>& dataset) {
    int n = dataset.size();
    AESAMatrix matrix(n);
    
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double dist = euclidean_distance(dataset[i], dataset[j]);
            matrix.set_distance(i, j, dist);
        }
    }
    return matrix;
}