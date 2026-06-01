#include "dataset.h"

void range_search(MTreeNode* node, const Object& q, double r, double d_q_pp, bool is_root, std::vector<Object>& results) {
    if (node->is_leaf) {
        for (const auto& entry : node->leaf_entries) {
            // Pruning Criterion 1 (Leaf level using parent history)
            if (!is_root) {
                if (std::abs(d_q_pp - entry.d_o_op) > r) {
                    continue; 
                }
            }
            
            // Unavoidable direct structural metric evaluation
            double d = continuous_distance(q, entry.o);
            if (d <= r) {
                results.push_back(entry.o);
            }
        }
        return;
    }

    for (const auto& entry : node->internal_entries) {
        // Pruning Criterion 1 (Internal level using parent history bounds)
        if (!is_root) {
            if (std::abs(d_q_pp - entry.d_p_pp) - entry.rc > r) {
                continue; 
            }
        }
        
        // Pruning Criterion 2 (Internal level using computed pivot distance)
        double d_q_p = continuous_distance(q, entry.p);
        if (d_q_p - entry.rc > r) {
            continue; 
        }
        
        // Recursive step into non-pruned path branch
        range_search(entry.ptr, q, r, d_q_p, false, results);
    }
}