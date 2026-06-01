#include "dataset.h"
#include <algorithm>

void range_search_fqt(FQTNode* node, const Object& q, int r, std::vector<Object>& results, std::unordered_map<int, int>& level_dist_cache) {
    if (!node) return;

    if (node->is_leaf) {
        // Collect matches stored within leaf allocations
        for (const auto& obj : node->objects) {
            if (discrete_distance(obj, q) <= r) {
                results.push_back(obj);
            }
        }
        return;
    }

    // FQT optimization: Calculate distance to the level's pivot exactly once per level depth
    if (level_dist_cache.find(node->level) == level_dist_cache.end()) {
        level_dist_cache[node->level] = discrete_distance(q, node->level_pivot);
    }
    int dq_p = level_dist_cache[node->level];

    // Compute search bounds according to Equation 3.1
    int low_bound = std::max(dq_p - r, 0);
    int high_bound = dq_p + r;

    // Prune branches that do not satisfy the metric bounding rule
    for (const auto& child_pair : node->children) {
        int branch_dist = child_pair.first;
        if (branch_dist >= low_bound && branch_dist <= high_bound) {
            range_search_fqt(child_pair.second, q, r, results, level_dist_cache);
        }
    }
}