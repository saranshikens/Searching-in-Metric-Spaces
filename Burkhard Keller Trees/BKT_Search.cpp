#include "dataset.h"
#include <algorithm>

void range_search_bkt(const std::unique_ptr<BKTNode>& root, const Object& q, int r, std::vector<Object>& results) {
    if (!root) return;

    int dist_q_p = discrete_distance(q, root->pivot);

    if (dist_q_p <= r) {
        results.push_back(root->pivot);
    }

    int min_i = std::max(dist_q_p - r, 0);
    int max_i = dist_q_p + r;

    for (const auto& pair : root->children) {
        int i = pair.first;
        if (i >= min_i && i <= max_i) {
            range_search_bkt(pair.second, q, r, results);
        }
    }
}