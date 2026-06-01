#include "dataset.h"

BSTNode* build_bst_recursive(std::vector<Point>& points) {
    if (points.empty()) return nullptr;

    BSTNode* node = new BSTNode();

    // Leaf condition: Only one point left
    if (points.size() == 1) {
        node->p1 = points[0];
        node->has_p1 = true;
        node->rc1 = 0.0;
        return node;
    }

    // Step 1: Select first two points as regional pivots
    node->p1 = points[0];
    node->p2 = points[1];
    node->has_p1 = true;
    node->has_p2 = true;

    std::vector<Point> left_points;
    std::vector<Point> right_points;
    double max_r1 = 0.0;
    double max_r2 = 0.0;

    // Step 2: Apply hyperplane partitioning for remaining items
    for (size_t i = 2; i < points.size(); ++i) {
        double d1 = continuous_distance(points[i], node->p1);
        double d2 = continuous_distance(points[i], node->p2);

        if (d1 < d2) {
            left_points.push_back(points[i]);
            if (d1 > max_r1) max_r1 = d1;
        } else {
            right_points.push_back(points[i]);
            if (d2 > max_r2) max_r2 = d2;
        }
    }

    // Step 3: Establish covering radii
    node->rc1 = max_r1;
    node->rc2 = max_r2;

    // Step 4: Recursively repartition
    node->left = build_bst_recursive(left_points);
    node->right = build_bst_recursive(right_points);

    return node;
}

void free_bst(BSTNode* node) {
    if (!node) return;
    free_bst(node->left);
    free_bst(node->right);
    delete node;
}