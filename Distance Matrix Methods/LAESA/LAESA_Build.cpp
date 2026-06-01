#include "dataset.h"
#include <algorithm>
#include <limits>

LAESAIndex build_laesa(const std::vector<Object>& dataset, int k) {
    LAESAIndex index;
    int n = dataset.size();
    index.k_pivots = std::min(k, n);
    index.pivot_indices.reserve(index.k_pivots);
    index.pivots.reserve(index.k_pivots);

    // 1. Pivot Selection: Select pivots maximizing distance from each other
    std::vector<double> min_dist_to_pivots(n, std::numeric_limits<double>::max());
    
    // Choose the first pivot arbitrarily (index 0)
    int next_pivot = 0;
    index.pivot_indices.push_back(next_pivot);
    index.pivots.push_back(dataset[next_pivot]);

    for (int p = 1; p < index.k_pivots; ++p) {
        int active_pivot = index.pivot_indices.back();
        double max_min_val = -1.0;
        int candidate_pivot = -1;

        for (int i = 0; i < n; ++i) {
            double d = continuous_distance(dataset[i], dataset[active_pivot]);
            if (d < min_dist_to_pivots[i]) {
                min_dist_to_pivots[i] = d;
            }
            // Find the object that is farthest from its closest pivot
            if (min_dist_to_pivots[i] > max_min_val) {
                max_min_val = min_dist_to_pivots[i];
                candidate_pivot = i;
            }
        }
        index.pivot_indices.push_back(candidate_pivot);
        index.pivots.push_back(dataset[candidate_pivot]);
    }

    // 2. Compute full N x k matrix containing metric steps to chosen pivots
    index.dist_matrix.assign(n, std::vector<double>(index.k_pivots, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int p = 0; p < index.k_pivots; ++p) {
            index.dist_matrix[i][p] = continuous_distance(dataset[i], index.pivots[p]);
        }
    }

    return index;
}