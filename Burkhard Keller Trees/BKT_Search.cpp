#include "dataset.h"
#include <algorithm>

// Evaluates a range query R(q, r) tracking qualifying matches
void range_search_bkt(const std::unique_ptr<BKTNode>& root, const Object& q, int r, std::vector<Object>& results) {
    if (!root) return;

    // Calculate distance from query to the current pivot
    int dist_q_p = discrete_distance(q, root->pivot);

    // If pivot satisfies the query condition, report it
    if (dist_q_p <= r) {
        results.push_back(root->pivot);
    }

    // Applying Equation 3.1: max{d(q,p) - r, 0} <= i <= d(q,p) + r
    int min_i = std::max(dist_q_p - r, 0);
    int max_i = dist_q_p + r;

    // Direct iteration over map elements to prune non-matching branches
    for (const auto& pair : root->children) {
        int i = pair.first;
        if (i >= min_i && i <= max_i) {
            range_search_bkt(pair.second, q, r, results);
        }
    }
}