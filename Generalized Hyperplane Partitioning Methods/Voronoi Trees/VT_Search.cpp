#include "dataset.h"

void range_search_vt(VTNode* node, const std::vector<double>& query, double r, std::vector<std::vector<double>>& results) {
    if (!node) return;

    if (node->is_leaf) {
        for (const auto& obj : node->leaf_objects) {
            if (euclidean_distance(query, obj) <= r) {
                results.push_back(obj);
            }
        }
        return;
    }

    // Process internal node partitions
    for (size_t i = 0; i < node->pivots.size(); ++i) {
        double d_q_p = euclidean_distance(query, node->pivots[i]);

        // If the routing pivot itself sits inside the target radius, collect it
        if (d_q_p <= r) {
            results.push_back(node->pivots[i]);
        }

        // Structural Pruning Condition: Only traverse branch if query intersects the covering ball
        if (d_q_p - r <= node->covering_radii[i]) {
            range_search_vt(node->children[i], query, r, results);
        }
    }
}