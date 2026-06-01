#include "dataset.h"

void range_search_gnat(GNATNode* node, const std::vector<double>& q, double r, std::vector<std::vector<double>>& results) {
    if (!node) return;

    // Linear assessment at leaf tier
    if (node->is_leaf) {
        for (const auto& point : node->leaf_points) {
            if (continuous_distance(q, point) <= r) {
                results.push_back(point);
            }
        }
        return;
    }

    size_t m = node->pivots.size();
    std::vector<bool> active(m, true);
    std::vector<double> dist_q_p(m, -1.0);

    for (size_t i = 0; i < m; ++i) {
        if (!active[i]) continue;

        // Evaluate distance between query and active pivot
        dist_q_p[i] = continuous_distance(q, node->pivots[i]);

        if (dist_q_p[i] <= r) {
            results.push_back(node->pivots[i]);
        }

        // Use the recorded distance constraints to eliminate other candidate branches early
        for (size_t j = 0; j < m; ++j) {
            if (!active[j]) continue;
            
            // Core GNAT range pruning rules
            if (dist_q_p[i] - r > node->table_max[i][j] || dist_q_p[i] + r < node->table_min[i][j]) {
                active[j] = false;
            }
        }
    }

    // Traverse surviving subtrees
    for (size_t j = 0; j < m; ++j) {
        if (active[j] && node->children[j] != nullptr) {
            range_search_gnat(node->children[j], q, r, results);
        }
    }
}