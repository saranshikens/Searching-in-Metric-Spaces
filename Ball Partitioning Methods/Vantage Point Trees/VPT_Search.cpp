#include "dataset.h"
#include <cmath>

void range_search_recursive(VPTNode* node, const Object& q, double r, 
                                    std::vector<double>& current_path_distances, 
                                    std::vector<Object>& results) {
    if (!node) return;

    // Leaf Phase processing
    if (node->is_leaf) {
        for (const auto& dist_obj : node->leaf_objects) {
            bool discarded = false;
            bool directly_included = false;

            // Apply VPs Pruning Conditions against all ancestor pivots on this branch path
            for (size_t i = 0; i < dist_obj.ancestor_distances.size(); ++i) {
                double dq_pi = current_path_distances[i];      // d(q, p_i) computed earlier
                double dpi_o = dist_obj.ancestor_distances[i]; // d(p_i, o) saved at build time

                // Condition 1: If |d(q,p) - d(p,o)| > r, discard without evaluation
                if (std::abs(dq_pi - dpi_o) > r) {
                    discarded = true;
                    break;
                }
                // Condition 2: If d(q,p) + d(p,o) <= r, instantly accept without evaluation
                if (dq_pi + dpi_o <= r) {
                    directly_included = true;
                    break;
                }
            }

            if (discarded) continue;
            if (directly_included) {
                results.push_back(dist_obj.obj);
                continue;
            }

            // Fallback: Compute actual distance if bounds are inconclusive
            if (euclidean_distance(q, dist_obj.obj) <= r) {
                results.push_back(dist_obj.obj);
            }
        }
        return;
    }

    // Internal Phase processing: Compute distance to current vantage point
    double dq_p = euclidean_distance(q, node->pivot);
    if (dq_p <= r) {
        results.push_back(node->pivot);
    }

    // Cache the distance to this pivot for downstream leaf optimizations
    current_path_distances.push_back(dq_p);

    // Ball Partitioning Pruning Checks
    // Enter Left Branch if max{d(q,p) - dm, 0} <= r
    if (dq_p - node->median <= r) {
        range_search_recursive(node->left, q, r, current_path_distances, results);
    }

    // Enter Right Branch if max{dm - d(q,p), 0} <= r
    if (node->median - dq_p <= r) {
        range_search_recursive(node->right, q, r, current_path_distances, results);
    }

    // Unwind cached distance tracking vector slot
    current_path_distances.pop_back();
}

std::vector<Object> range_search_vpt(VPTNode* root, const Object& q, double r) {
    std::vector<Object> results;
    std::vector<double> current_path_distances; // Stores query distances along active stack path
    range_search_recursive(root, q, r, current_path_distances, results);
    return results;
}