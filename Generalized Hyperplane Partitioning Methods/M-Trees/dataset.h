#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cmath>
#include <iostream>

// Global distance evaluation counter
extern long long distance_count;

using Object = std::vector<double>;

// Continuous metric distance function (Euclidean)
inline double continuous_distance(const Object& a, const Object& b) {
    distance_count++;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

struct MTreeNode;

// Entry tuple for an internal node: (p, rc, d(p, pp), ptr)
struct InternalEntry {
    Object p;       // Pivot
    double rc;      // Covering radius
    double d_p_pp;  // Distance to parent pivot
    MTreeNode* ptr; // Pointer to child node
};

// Entry tuple for a leaf node: (o, d(o, op))
struct LeafEntry {
    Object o;       // Database object
    double d_o_op;  // Distance to parent object (pivot)
};

struct MTreeNode {
    bool is_leaf;
    std::vector<InternalEntry> internal_entries;
    std::vector<LeafEntry> leaf_entries;

    MTreeNode(bool leaf) : is_leaf(leaf) {}
    
    ~MTreeNode() {
        if (!is_leaf) {
            for (auto& entry : internal_entries) {
                delete entry.ptr;
            }
        }
    }
};

struct SplitResult {
    bool split_occurred = false;
    Object p1, p2;
    double rc1, rc2;
    MTreeNode* left_ptr = nullptr;
    MTreeNode* right_ptr = nullptr;
};

// Pipeline API definitions
void insert_m_tree(MTreeNode*& root, const Object& o_n);
SplitResult insert_node(MTreeNode* node, const Object& o_n, const Object& curr_p);
SplitResult split_leaf(MTreeNode* node);
SplitResult split_internal(MTreeNode* node);

void range_search(MTreeNode* node, const Object& q, double r, double d_q_pp, bool is_root, std::vector<Object>& results);

#endif