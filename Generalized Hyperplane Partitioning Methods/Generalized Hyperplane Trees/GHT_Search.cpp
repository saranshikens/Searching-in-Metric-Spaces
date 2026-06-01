#include "dataset.h"
#include <vector>

// Navigates the GHT tree by executing strict hyperplane pruning inequalities
void range_search_ght(GHTNode* node, const Point& q, double r, std::vector<Point>& results) {
    if (!node) return;

    if (node->is_leaf) {
        for (const auto& obj : node->objects) {
            if (distance(q, obj) <= r) {
                results.push_back(obj);
            }
        }
        return;
    }

    // Evaluate current internal node pivots against the range constraint
    double dq_p1 = distance(q, node->p1);
    double dq_p2 = distance(q, node->p2);

    if (dq_p1 <= r) {
        results.push_back(node->p1);
    }
    if (dq_p2 <= r) {
        results.push_back(node->p2);
    }

    // Left Subtree Condition: Traverse if d(q, p1) - r <= d(q, p2) + r
    if (dq_p1 - r <= dq_p2 + r) {
        range_search_ght(node->left, q, r, results);
    }

    // Right Subtree Condition: Traverse if d(q, p1) + r >= d(q, p2) - r
    if (dq_p1 + r >= dq_p2 - r) {
        range_search_ght(node->right, q, r, results);
    }
}