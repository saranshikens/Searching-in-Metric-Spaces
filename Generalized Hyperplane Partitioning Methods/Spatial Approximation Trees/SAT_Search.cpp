#include "SAT.h"

void rangeSearchSAT(SATNode* node, const Point& q, double r, double d_q_p, double min_d_ancestors_neighbors, std::vector<Point>& results) {
    if (!node) return;

    // 1. Covering Ball Pruning Constraint
    if (d_q_p > node->rc + r) {
        return;
    }

    // 2. Validate current pivot identity
    if (d_q_p <= r) {
        results.push_back(node->pivot);
    }

    if (node->children.empty()) return;

    // 3. Dynamic search-path optimization
    // Track closest distance among ancestors and their immediate neighbors
    double local_min_d = min_d_ancestors_neighbors;
    if (d_q_p < local_min_d) {
        local_min_d = d_q_p;
    }

    std::vector<double> d_q_children(node->children.size());
    for (size_t i = 0; i < node->children.size(); ++i) {
        d_q_children[i] = euclidean_distance(q, node->children[i]->pivot);
        if (d_q_children[i] < local_min_d) {
            local_min_d = d_q_children[i];
        }
    }

    // 4. Spatial Approximation Hyperplane Pruning Rule
    for (size_t i = 0; i < node->children.size(); ++i) {
        if (d_q_children[i] <= 2 * r + local_min_d) {
            rangeSearchSAT(node->children[i], q, r, d_q_children[i], local_min_d, results);
        }
    }
}