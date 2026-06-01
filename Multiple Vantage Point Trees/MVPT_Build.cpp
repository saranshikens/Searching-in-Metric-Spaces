#include "dataset.h"
#include <algorithm>

double find_median(std::vector<double>& distances) {
    if (distances.empty()) return 0.0;
    size_t n = distances.size();
    std::sort(distances.begin(), distances.end());
    if (n % 2 == 1) {
        return distances[n / 2];
    } else {
        return (distances[n / 2 - 1] + distances[n / 2]) / 2.0;
    }
}

MVPTNode* build_mvpt(std::vector<Point>& points, std::vector<Point>& path_pivots, size_t leaf_capacity = 4) {
    if (points.empty()) return nullptr;

    if (points.size() <= leaf_capacity) {
        MVPTNode* leaf = new MVPTNode(true);
        leaf->leaf_objects = points;
        leaf->leaf_distances.resize(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            for (const auto& pivot : path_pivots) {
                leaf->leaf_distances[i].push_back(euclidean_distance(points[i], pivot));
            }
        }
        return leaf;
    }

    MVPTNode* node = new MVPTNode(false);

    // Extract and assign Pivot 1
    node->p1 = points.back();
    points.pop_back();
    path_pivots.push_back(node->p1);

    if (points.empty()) {
        points.push_back(node->p1);
        path_pivots.pop_back();
        MVPTNode* single_leaf = new MVPTNode(true);
        single_leaf->leaf_objects = points;
        single_leaf->leaf_distances.resize(1);
        return single_leaf;
    }

    std::vector<double> dists_p1;
    for (const auto& pt : points) {
        dists_p1.push_back(euclidean_distance(pt, node->p1));
    }

    std::vector<double> dists_p1_copy = dists_p1;
    node->dm1 = find_median(dists_p1_copy);

    std::vector<Point> S_left, S_right;
    for (size_t i = 0; i < points.size(); ++i) {
        if (dists_p1[i] <= node->dm1) S_left.push_back(points[i]);
        else S_right.push_back(points[i]);
    }

    // Extract and assign Pivot 2
    if (!S_left.empty()) {
        node->p2 = S_left.back();
        S_left.pop_back();
    } else {
        node->p2 = S_right.back();
        S_right.pop_back();
    }
    path_pivots.push_back(node->p2);

    // Sub-partition S_left using shared Pivot 2
    std::vector<Point> S_00, S_01;
    if (!S_left.empty()) {
        std::vector<double> dists_p2_left;
        for (const auto& pt : S_left) dists_p2_left.push_back(euclidean_distance(pt, node->p2));
        std::vector<double> dists_p2_left_copy = dists_p2_left;
        node->dm2_left = find_median(dists_p2_left_copy);
        for (size_t i = 0; i < S_left.size(); ++i) {
            if (dists_p2_left[i] <= node->dm2_left) S_00.push_back(S_left[i]);
            else S_01.push_back(S_left[i]);
        }
    }

    // Sub-partition S_right using shared Pivot 2
    std::vector<Point> S_10, S_11;
    if (!S_right.empty()) {
        std::vector<double> dists_p2_right;
        for (const auto& pt : S_right) dists_p2_right.push_back(euclidean_distance(pt, node->p2));
        std::vector<double> dists_p2_right_copy = dists_p2_right;
        node->dm2_right = find_median(dists_p2_right_copy);
        for (size_t i = 0; i < S_right.size(); ++i) {
            if (dists_p2_right[i] <= node->dm2_right) S_10.push_back(S_right[i]);
            else S_11.push_back(S_right[i]);
        }
    }

    // Build out the 4 branches
    node->children[0] = build_mvpt(S_00, path_pivots, leaf_capacity);
    node->children[1] = build_mvpt(S_01, path_pivots, leaf_capacity);
    node->children[2] = build_mvpt(S_10, path_pivots, leaf_capacity);
    node->children[3] = build_mvpt(S_11, path_pivots, leaf_capacity);

    path_pivots.pop_back(); // Pop p2
    path_pivots.pop_back(); // Pop p1

    return node;
}