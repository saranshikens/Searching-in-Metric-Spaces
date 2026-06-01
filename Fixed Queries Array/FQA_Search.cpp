#include "dataset.h"

// Binary search helper to find the start of a distance value at a given pivot level
int find_lower_bound(const std::vector<FQAElement>& fqa, int start, int end, int level, int value) {
    int low = start, high = end, ans = end + 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (fqa[mid].dists[level] >= value) {
            ans = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return ans;
}

// Binary search helper to find the end of a distance value at a given pivot level
int find_upper_bound(const std::vector<FQAElement>& fqa, int start, int end, int level, int value) {
    int low = start, high = end, ans = start - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (fqa[mid].dists[level] <= value) {
            ans = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return ans;
}

void range_search_FQA_recursive(const std::vector<FQAElement>& fqa, int level, int start, int end,
                                const Vector& q, int r, const std::vector<int>& q_pivot_dists, int h,
                                std::vector<Vector>& results) {
    if (start > end) return;

    // If all h pivot constraints pass, verify candidate targets in the narrowed subrange
    if (level == h) {
        for (int i = start; i <= end; ++i) {
            if (discrete_distance(q, fqa[i].obj) <= r) {
                results.push_back(fqa[i].obj);
            }
        }
        return;
    }

    // Grab precomputed distance to the current level's fixed pivot
    int q_p_dist = q_pivot_dists[level];
    int min_val = std::max(0, q_p_dist - r);
    int max_val = q_p_dist + r;

    // Binary search subranges matching valid triangular inequality parameters
    for (int v = min_val; v <= max_val; ++v) {
        int sub_start = find_lower_bound(fqa, start, end, level, v);
        int sub_end = find_upper_bound(fqa, start, end, level, v);

        if (sub_start <= sub_end) {
            range_search_FQA_recursive(fqa, level + 1, sub_start, sub_end, q, r, q_pivot_dists, h, results);
        }
    }
}

std::vector<Vector> range_search_FQA(const std::vector<FQAElement>& fqa, const Vector& q, int r,
                                     const std::vector<Vector>& pivots, int h) {
    std::vector<Vector> results;
    if (fqa.empty()) return results;

    // Precompute distances to all h pivots: guarantees exactly O(h) filtering distance computations
    std::vector<int> q_pivot_dists(h);
    for (int j = 0; j < h; ++j) {
        q_pivot_dists[j] = discrete_distance(q, pivots[j]);
    }

    range_search_FQA_recursive(fqa, 0, 0, fqa.size() - 1, q, r, q_pivot_dists, h, results);
    return results;
}