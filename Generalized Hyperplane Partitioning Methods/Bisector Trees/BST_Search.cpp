#include "dataset.h"

void bst_range_search(BSTNode* node, const Point& query, double r, std::vector<Point>& results) {
    if (!node) return;

    double d1 = -1.0;
    double d2 = -1.0;

    // Evaluate against regional pivots
    if (node->has_p1) {
        d1 = continuous_distance(query, node->p1);
        if (d1 <= r) {
            results.push_back(node->p1);
        }
    }

    if (node->has_p2) {
        d2 = continuous_distance(query, node->p2);
        if (d2 <= r) {
            results.push_back(node->p2);
        }
    }

    // Pruning Strategy: Enter branch only if d(q, pi) - r <= rci
    if (node->left && node->has_p1) {
        if (d1 - r <= node->rc1) {
            bst_range_search(node->left, query, r, results);
        }
    }

    if (node->right && node->has_p2) {
        if (d2 - r <= node->rc2) {
            bst_range_search(node->right, query, r, results);
        }
    }
}