#include "dataset.h"
#include <algorithm>

SplitResult split_leaf(MTreeNode* node) {
    size_t p1_idx = 0, p2_idx = 1;
    double max_d = -1.0;
    
    // Heuristic: Pick the two farthest objects as new pivots
    for (size_t i = 0; i < node->leaf_entries.size(); ++i) {
        for (size_t j = i + 1; j < node->leaf_entries.size(); ++j) {
            double d = continuous_distance(node->leaf_entries[i].o, node->leaf_entries[j].o);
            if (d > max_d) {
                max_d = d;
                p1_idx = i;
                p2_idx = j;
            }
        }
    }
    Object pivot1 = node->leaf_entries[p1_idx].o;
    Object pivot2 = node->leaf_entries[p2_idx].o;

    MTreeNode* left_node = new MTreeNode(true);
    MTreeNode* right_node = new MTreeNode(true);
    double rc1 = 0.0, rc2 = 0.0;

    for (const auto& entry : node->leaf_entries) {
        double d1 = continuous_distance(entry.o, pivot1);
        double d2 = continuous_distance(entry.o, pivot2);
        if (d1 <= d2) {
            left_node->leaf_entries.push_back({entry.o, d1});
            if (d1 > rc1) rc1 = d1;
        } else {
            right_node->leaf_entries.push_back({entry.o, d2});
            if (d2 > rc2) rc2 = d2;
        }
    }

    SplitResult sr;
    sr.split_occurred = true;
    sr.p1 = pivot1;
    sr.p2 = pivot2;
    sr.rc1 = rc1;
    sr.rc2 = rc2;
    sr.left_ptr = left_node;
    sr.right_ptr = right_node;
    return sr;
}

SplitResult split_internal(MTreeNode* node) {
    size_t p1_idx = 0, p2_idx = 1;
    double max_d = -1.0;
    
    for (size_t i = 0; i < node->internal_entries.size(); ++i) {
        for (size_t j = i + 1; j < node->internal_entries.size(); ++j) {
            double d = continuous_distance(node->internal_entries[i].p, node->internal_entries[j].p);
            if (d > max_d) {
                max_d = d;
                p1_idx = i;
                p2_idx = j;
            }
        }
    }
    Object pivot1 = node->internal_entries[p1_idx].p;
    Object pivot2 = node->internal_entries[p2_idx].p;

    MTreeNode* left_node = new MTreeNode(false);
    MTreeNode* right_node = new MTreeNode(false);
    double rc1 = 0.0, rc2 = 0.0;

    for (auto& entry : node->internal_entries) {
        double d1 = continuous_distance(entry.p, pivot1);
        double d2 = continuous_distance(entry.p, pivot2);
        if (d1 <= d2) {
            entry.d_p_pp = d1;
            left_node->internal_entries.push_back(entry);
            if (d1 + entry.rc > rc1) rc1 = d1 + entry.rc;
        } else {
            entry.d_p_pp = d2;
            right_node->internal_entries.push_back(entry);
            if (d2 + entry.rc > rc2) rc2 = d2 + entry.rc;
        }
    }

    SplitResult sr;
    sr.split_occurred = true;
    sr.p1 = pivot1;
    sr.p2 = pivot2;
    sr.rc1 = rc1;
    sr.rc2 = rc2;
    sr.left_ptr = left_node;
    sr.right_ptr = right_node;
    return sr;
}

SplitResult insert_node(MTreeNode* node, const Object& o_n, const Object& curr_p) {
    if (node->is_leaf) {
        double d = curr_p.empty() ? 0.0 : continuous_distance(o_n, curr_p);
        node->leaf_entries.push_back({o_n, d});
        if (node->leaf_entries.size() > 4) {
            return split_leaf(node);
        }
        return SplitResult{false};
    }

    int best_idx = -1;
    double min_enlargement = 1e18;
    double min_dist = 1e18;

    // Follow the text heuristic rules for choosing subtrees
    for (size_t i = 0; i < node->internal_entries.size(); ++i) {
        double d = continuous_distance(o_n, node->internal_entries[i].p);
        double enlargement = std::max(0.0, d - node->internal_entries[i].rc);
        
        if (best_idx == -1) {
            best_idx = i;
            min_enlargement = enlargement;
            min_dist = d;
        } else {
            if (enlargement == 0.0 && min_enlargement == 0.0) {
                if (d < min_dist) {
                    min_dist = d;
                    best_idx = i;
                }
            } else if (enlargement == 0.0 && min_enlargement > 0.0) {
                best_idx = i;
                min_enlargement = 0.0;
                min_dist = d;
            } else if (enlargement > 0.0 && min_enlargement > 0.0) {
                if (enlargement < min_enlargement) {
                    min_enlargement = enlargement;
                    min_dist = d;
                    best_idx = i;
                }
            }
        }
    }

    MTreeNode* child = node->internal_entries[best_idx].ptr;
    Object child_p = node->internal_entries[best_idx].p;
    
    SplitResult sr = insert_node(child, o_n, child_p);

    if (sr.split_occurred) {
        node->internal_entries.erase(node->internal_entries.begin() + best_idx);
        
        double d1 = curr_p.empty() ? 0.0 : continuous_distance(sr.p1, curr_p);
        double d2 = curr_p.empty() ? 0.0 : continuous_distance(sr.p2, curr_p);
        
        InternalEntry e1{sr.p1, sr.rc1, d1, sr.left_ptr};
        InternalEntry e2{sr.p2, sr.rc2, d2, sr.right_ptr};
        
        node->internal_entries.push_back(e1);
        node->internal_entries.push_back(e2);
        
        delete child;

        if (node->internal_entries.size() > 4) {
            return split_internal(node);
        }
    } else {
        double d = continuous_distance(o_n, node->internal_entries[best_idx].p);
        if (d > node->internal_entries[best_idx].rc) {
            node->internal_entries[best_idx].rc = d;
        }
    }
    return SplitResult{false};
}

void insert_m_tree(MTreeNode*& root, const Object& o_n) {
    if (root == nullptr) {
        root = new MTreeNode(true);
        root->leaf_entries.push_back({o_n, 0.0});
        return;
    }

    Object dummy_p;
    SplitResult sr = insert_node(root, o_n, dummy_p);

    if (sr.split_occurred) {
        MTreeNode* new_root = new MTreeNode(false);
        InternalEntry e1{sr.p1, sr.rc1, 0.0, sr.left_ptr};
        InternalEntry e2{sr.p2, sr.rc2, 0.0, sr.right_ptr};
        new_root->internal_entries.push_back(e1);
        new_root->internal_entries.push_back(e2);
        delete root;
        root = new_root;
    }
}