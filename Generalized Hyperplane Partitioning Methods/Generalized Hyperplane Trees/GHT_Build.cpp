#include "dataset.h"
#include <vector>

// Recursively partitions data using the generalized hyperplane principle
GHTNode* build_ght(std::vector<Point>& points, int leaf_capacity = 4) {
    if (points.empty()) {
        return nullptr;
    }

    GHTNode* node = new GHTNode();

    // Fall back to a leaf layout if the data pool fits or cannot yield two pivots
    if (points.size() <= (size_t)leaf_capacity || points.size() < 2) {
        node->is_leaf = true;
        node->objects = points;
        return node;
    }

    node->is_leaf = false;

    // Pivot selection: Select first element as p1, furthest object from p1 as p2
    node->p1 = points[0];
    node->has_p1 = true;

    size_t p2_idx = 1;
    double max_dist = -1.0;
    for (size_t i = 1; i < points.size(); ++i) {
        double d = distance(node->p1, points[i]);
        if (d > max_dist) {
            max_dist = d;
            p2_idx = i;
        }
    }
    node->p2 = points[p2_idx];
    node->has_p2 = true;

    // Split remaining set relative to the separating hyperplane
    std::vector<Point> left_points;
    std::vector<Point> right_points;

    for (size_t i = 0; i < points.size(); ++i) {
        // Exclude internal node pivots from subtrees to keep space complexity O(N)
        if (i == 0 || i == p2_idx) {
            continue;
        }

        double d1 = distance(node->p1, points[i]);
        double d2 = distance(node->p2, points[i]);

        if (d1 <= d2) {
            left_points.push_back(points[i]);
        } else {
            right_points.push_back(points[i]);
        }
    }

    node->left = build_ght(left_points, leaf_capacity);
    node->right = build_ght(right_points, leaf_capacity);

    return node;
}