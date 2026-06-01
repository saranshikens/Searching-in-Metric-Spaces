#include "dataset.h"

void search_vpf_node(VPFNode* node, const Point& q, double r, double rho, 
                     std::vector<Point>& results, bool& search_subsequent_tree) {
    if (!node) return;

    if (node->is_leaf) {
        for (const auto& p : node->leaf_points) {
            if (continuous_distance(q, p) <= r) {
                results.push_back(p);
            }
        }
        return;
    }

    double dq = continuous_distance(q, node->pivot);
    if (dq <= r) {
        results.push_back(node->pivot);
    }

    // Flag the next forest tree for searching if the query ball overlaps the node's exclusion ring
    if (dq - r <= node->dm + rho && dq + r > node->dm - rho) {
        search_subsequent_tree = true;
    }

    // Pruning navigation matching spatial bounds
    if (dq - r <= node->dm - rho) {
        search_vpf_node(node->left, q, r, rho, results, search_subsequent_tree);
    }
    if (dq + r > node->dm + rho) {
        search_vpf_node(node->right, q, r, rho, results, search_subsequent_tree);
    }
}

std::vector<Point> search_vpf_forest(const std::vector<VPFTree>& forest, const Point& q, double r, double rho) {
    std::vector<Point> combined_results;
    
    for (const auto& tree : forest) {
        bool search_next = false;
        search_vpf_node(tree.root, q, r, rho, combined_results, search_next);
        
        // Worst-case performance optimization: If the query completely missed 
        // the exclusion zone of this tree, subsequent forest branches are skipped.
        if (!search_next) {
            break;
        }
    }
    return combined_results;
}