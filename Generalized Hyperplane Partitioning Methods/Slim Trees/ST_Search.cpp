#include "dataset.h"
#include <vector>

void range_search_rec(STNode* node, const Object& query, double r, std::vector<Object>& results) {
    if (!node) return;
    
    if (node->is_leaf) {
        for (const auto& obj : node->objects) {
            if (euclidean_distance(query, obj) <= r) {
                results.push_back(obj);
            }
        }
        return;
    }
    
    // Pruning Rule: Enter a branch only if d(q, pi) - r <= rc_i
    double d_left = euclidean_distance(query, node->left_pivot);
    if (d_left - r <= node->left_radius) {
        range_search_rec(node->left, query, r, results);
    }
    
    double d_right = euclidean_distance(query, node->right_pivot);
    if (d_right - r <= node->right_radius) {
        range_search_rec(node->right, query, r, results);
    }
}

std::vector<Object> slim_tree_range_search(STNode* root, const Object& query, double r) {
    std::vector<Object> results;
    range_search_rec(root, query, r, results);
    return results;
}