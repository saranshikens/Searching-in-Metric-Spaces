#include "dataset.h"
#include <algorithm>

// Recursive building of an individual tree within the forest
VPFNode* build_individual_node(std::vector<Point>& points, double rho, std::vector<Point>& global_exclusion) {
    if (points.empty()) return nullptr;

    VPFNode* node = new VPFNode();

    // Enforce leaf capacity constraint
    if (points.size() <= 4) {
        node->is_leaf = true;
        node->leaf_points = points;
        return node;
    }

    // Designate the last point as the current vantage point (pivot)
    node->pivot = points.back();
    points.pop_back();

    if (points.empty()) {
        node->is_leaf = true;
        node->leaf_points.push_back(node->pivot);
        return node;
    }

    // Evaluate distances to calculate the true median
    std::vector<double> distances;
    distances.reserve(points.size());
    for (const auto& p : points) {
        distances.push_back(continuous_distance(node->pivot, p));
    }

    std::vector<double> sorted_distances = distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    node->dm = sorted_distances[sorted_distances.size() / 2];

    // Partition using Excluded Middle conditions (Equation 3.2)
    std::vector<Point> S0, S1;
    for (size_t i = 0; i < points.size(); ++i) {
        double d = distances[i];
        if (d <= node->dm - rho) {
            S0.push_back(points[i]);         // Left Branch
        } else if (d > node->dm + rho) {
            S1.push_back(points[i]);         // Right Branch
        } else {
            global_exclusion.push_back(points[i]); // Expelled to the global exclusion set
        }
    }

    node->left = build_individual_node(S0, rho, global_exclusion);
    node->right = build_individual_node(S1, rho, global_exclusion);

    return node;
}

// Chaining iteration that aggregates exclusion sets into sequential tree records
std::vector<VPFTree> build_vpf_forest(std::vector<Point> dataset, double rho) {
    std::vector<VPFTree> forest;
    std::vector<Point> current_pool = dataset;

    while (!current_pool.empty()) {
        std::vector<Point> next_exclusion_pool;
        VPFTree tree;
        tree.root = build_individual_node(current_pool, rho, next_exclusion_pool);
        forest.push_back(tree);
        
        // If the pool did not shrink (safety check for extreme parameters), break to avoid loops
        if (next_exclusion_pool.size() == current_pool.size()) {
            break; 
        }
        current_pool = next_exclusion_pool;
    }
    return forest;
}