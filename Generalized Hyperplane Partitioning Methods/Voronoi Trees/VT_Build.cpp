#include "dataset.h"
#include <algorithm>

const size_t LEAF_CAPACITY = 4; // Maximum elements in a leaf node

VTNode* build_vt_tree(std::vector<std::vector<double>> objects) {
    if (objects.empty()) return nullptr;

    VTNode* node = new VTNode();

    // Base condition: create a leaf node if items fit within capacity
    if (objects.size() <= LEAF_CAPACITY) {
        node->is_leaf = true;
        node->leaf_objects = objects;
        return node;
    }

    // Voronoi Tree Property: Dynamically choose 2 or 3 pivots
    size_t num_pivots = (objects.size() >= 3) ? 3 : 2;

    for (size_t i = 0; i < num_pivots; ++i) {
        node->pivots.push_back(objects[i]);
    }

    // Eliminate routing pivots from downstream data array to prevent node duplication
    std::vector<std::vector<double>> remaining_objects(objects.begin() + num_pivots, objects.end());

    std::vector<std::vector<std::vector<double>>> child_buckets(num_pivots);
    node->covering_radii.assign(num_pivots, 0.0);

    // Group remaining items under their closest matching pivot
    for (const auto& obj : remaining_objects) {
        double min_dist = -1.0;
        size_t best_pivot_idx = 0;

        for (size_t i = 0; i < num_pivots; ++i) {
            double d = euclidean_distance(obj, node->pivots[i]);
            if (min_dist < 0 || d < min_dist) {
                min_dist = d;
                best_pivot_idx = i;
            }
        }

        child_buckets[best_pivot_idx].push_back(obj);
        
        // Track the covering radius (maximum distance to any item in this sub-region)
        if (min_dist > node->covering_radii[best_pivot_idx]) {
            node->covering_radii[best_pivot_idx] = min_dist;
        }
    }

    // Recursively construct child branches
    for (size_t i = 0; i < num_pivots; ++i) {
        node->children.push_back(build_vt_tree(child_buckets[i]));
    }

    return node;
}