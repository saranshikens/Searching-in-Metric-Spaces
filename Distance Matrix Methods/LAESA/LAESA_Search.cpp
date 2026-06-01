#include "dataset.h"
#include <cmath>

std::vector<Object> range_search_laesa(const LAESAIndex& index, const std::vector<Object>& dataset, const Object& query, double radius) {
    int n = dataset.size();
    std::vector<bool> discarded(n, false);
    std::vector<double> dist_q_to_pivot(index.k_pivots, 0.0);

    // Phase 1: Eliminate objects using precomputed bounds across all k pivots
    for (int p = 0; p < index.k_pivots; ++p) {
        double d_q_p = continuous_distance(query, index.pivots[p]);
        dist_q_to_pivot[p] = d_q_p;

        for (int i = 0; i < n; ++i) {
            if (!discarded[i]) {
                // Pruning Rule: |d(p, o) - d(q, p)| > r
                if (std::abs(index.dist_matrix[i][p] - d_q_p) > radius) {
                    discarded[i] = true;
                }
            }
        }
    }

    // Phase 2: Direct scan evaluation of remaining non-discarded candidates
    std::vector<Object> results;
    for (int i = 0; i < n; ++i) {
        if (!discarded[i]) {
            double d_q_o = continuous_distance(query, dataset[i]);
            if (d_q_o <= radius) {
                results.push_back(dataset[i]);
            }
        }
    }

    return results;
}