#include "dataset.h"
#include <algorithm>

void search_mvpt(MVPTNode* node, const Point& q, double r, 
                 std::vector<double>& path_q_distances, std::vector<Point>& results) {
    if (!node) return;

    if (node->is_leaf) {
        for (size_t i = 0; i < node->leaf_objects.size(); ++i) {
            const auto& o = node->leaf_objects[i];
            bool pruned = false;
            bool directly_included = false;

            // Apply Lemma 2.6.1 filtering bounds via precomputed history
            size_t check_limit = std::min(path_q_distances.size(), node->leaf_distances[i].size());
            for (size_t p_idx = 0; p_idx < check_limit; ++p_idx) {
                double dq_p = path_q_distances[p_idx];
                double dp_o = node->leaf_distances[i][p_idx];

                // Lower Bound Elimination condition
                if (std::abs(dq_p - dp_o) > r) {
                    pruned = true;
                    break;
                }
                // Upper Bound Direct Inclusion condition
                if (dq_p + dp_o <= r) {
                    directly_included = true;
                }
            }

            if (pruned) continue; 

            if (directly_included) {
                results.push_back(o);
                continue;
            }

            // Fallback to absolute distance calculation if bounds overlap
            if (euclidean_distance(q, o) <= r) {
                results.push_back(o);
            }
        }
        return;
    }

    // Evaluate against node pivots
    double dq_p1 = euclidean_distance(q, node->p1);
    if (dq_p1 <= r) results.push_back(node->p1);

    double dq_p2 = euclidean_distance(q, node->p2);
    if (dq_p2 <= r) results.push_back(node->p2);

    // Save path history for downstream leaf pruning
    path_q_distances.push_back(dq_p1);
    path_q_distances.push_back(dq_p2);

    // Structural level-routing conditions
    bool visit_left = std::max(dq_p1 - node->dm1, 0.0) <= r;
    bool visit_right = std::max(node->dm1 - dq_p1, 0.0) <= r;

    if (visit_left) {
        if (std::max(dq_p2 - node->dm2_left, 0.0) <= r) {
            search_mvpt(node->children[0], q, r, path_q_distances, results);
        }
        if (std::max(node->dm2_left - dq_p2, 0.0) <= r) {
            search_mvpt(node->children[1], q, r, path_q_distances, results);
        }
    }

    if (visit_right) {
        if (std::max(dq_p2 - node->dm2_right, 0.0) <= r) {
            search_mvpt(node->children[2], q, r, path_q_distances, results);
        }
        if (std::max(node->dm2_right - dq_p2, 0.0) <= r) {
            search_mvpt(node->children[3], q, r, path_q_distances, results);
        }
    }

    // Backtrack query history cache
    path_q_distances.pop_back();
    path_q_distances.pop_back();
}