#include "dataset.h"
#include <algorithm>

VPTNode* build_vpt_recursive(std::vector<DistanceObject>& objects, int leaf_capacity) {
    if (objects.empty()) return nullptr;

    VPTNode* node = new VPTNode();

    // Base Case: Leaf Capacity reached
    if (objects.size() <= (size_t)leaf_capacity) {
        node->is_leaf = true;
        node->leaf_objects = objects;
        return node;
    }

    // Pick the first object as the Vantage Point (Pivot)
    DistanceObject pivot_dist_obj = objects.front();
    node->pivot = pivot_dist_obj.obj;

    // Remove the selected pivot from the pool passed to children
    objects.erase(objects.begin());

    if (objects.empty()) {
        node->is_leaf = true;
        node->leaf_objects.push_back(pivot_dist_obj);
        return node;
    }

    // Compute distances from the vantage point to all remaining sub-elements
    std::vector<double> distances;
    distances.reserve(objects.size());
    for (size_t i = 0; i < objects.size(); ++i) {
        double d = euclidean_distance(node->pivot, objects[i].obj);
        distances.push_back(d);
        
        // VPs Tree Strategy: Remember this distance for ancestral lookups during searches
        objects[i].ancestor_distances.push_back(d);
    }

    // Select the Median split distance
    std::vector<double> sorted_distances = distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    double median = sorted_distances[sorted_distances.size() / 2];
    node->median = median;

    // Split objects into balanced S1 (inside/on ball) and S2 (outside ball) groups
    std::vector<DistanceObject> left_objects;
    std::vector<DistanceObject> right_objects;

    for (size_t i = 0; i < objects.size(); ++i) {
        if (distances[i] <= median) {
            left_objects.push_back(objects[i]);
        } else {
            right_objects.push_back(objects[i]);
        }
    }

    node->left = build_vpt_recursive(left_objects, leaf_capacity);
    node->right = build_vpt_recursive(right_objects, leaf_capacity);

    return node;
}

VPTNode* build_vpt(const std::vector<Object>& dataset, int leaf_capacity) {
    std::vector<DistanceObject> dist_objects;
    dist_objects.reserve(dataset.size());
    for (const auto& obj : dataset) {
        dist_objects.push_back({obj, {}});
    }
    return build_vpt_recursive(dist_objects, leaf_capacity);
}