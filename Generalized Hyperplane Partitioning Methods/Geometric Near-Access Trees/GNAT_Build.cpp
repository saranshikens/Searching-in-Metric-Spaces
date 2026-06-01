#include "dataset.h"
#include <algorithm>
#include <limits>

GNATNode* build_gnat(std::vector<std::vector<double>> points, size_t m) {
    if (points.empty()) return nullptr;

    GNATNode* node = new GNATNode();

    // If remaining points are fewer than or equal to m, form a leaf node
    if (points.size() <= m) {
        node->is_leaf = true;
        node->leaf_points = points;
        return node;
    }

    // Select the first m points as structural pivots
    for (size_t i = 0; i < m; ++i) {
        node->pivots.push_back(points[i]);
    }

    // Prepare subsets S_0 to S_{m-1}
    std::vector<std::vector<std::vector<double>>> subsets(m);
    
    // Partition remaining elements into their respective closest Dirichlet cells
    for (size_t i = m; i < points.size(); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        size_t best_pivot = 0;
        for (size_t j = 0; j < m; ++j) {
            double d = continuous_distance(points[i], node->pivots[j]);
            if (d < min_dist) {
                min_dist = d;
                best_pivot = j;
            }
        }
        subsets[best_pivot].push_back(points[i]);
    }

    // Initialize m x m tables
    node->table_min.assign(m, std::vector<double>(m, std::numeric_limits<double>::max()));
    node->table_max.assign(m, std::vector<double>(m, -1.0));

    // Calculate minimum and maximum distances between pivot p_i and subset S_j U {p_j}
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            // Process the sub-pivot itself
            double d_pj = continuous_distance(node->pivots[i], node->pivots[j]);
            node->table_min[i][j] = std::min(node->table_min[i][j], d_pj);
            node->table_max[i][j] = std::max(node->table_max[i][j], d_pj);

            // Process all elements belonging to cell S_j
            for (const auto& obj : subsets[j]) {
                double d_obj = continuous_distance(node->pivots[i], obj);
                node->table_min[i][j] = std::min(node->table_min[i][j], d_obj);
                node->table_max[i][j] = std::max(node->table_max[i][j], d_obj);
            }
        }
    }

    // Recursively grow children branches
    node->children.resize(m, nullptr);
    for (size_t j = 0; j < m; ++j) {
        if (!subsets[j].empty()) {
            node->children[j] = build_gnat(subsets[j], m);
        }
    }

    return node;
}