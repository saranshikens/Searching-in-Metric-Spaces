#include "dataset.h"
#include <vector>
#include <algorithm>

STNode* build_slim_tree_rec(const std::vector<Object>& objs, int max_leaf_capacity) {
    if (objs.empty()) return nullptr;
    
    // Base Case: Elements fit entirely within maximum leaf limit limits
    if ((int)objs.size() <= max_leaf_capacity) {
        STNode* node = new STNode();
        node->is_leaf = true;
        node->objects = objs;
        return node;
    }
    
    int n = objs.size();
    std::vector<double> min_dist(n, 1e18);
    std::vector<int> parent(n, -1);
    std::vector<bool> in_mst(n, false);
    
    min_dist[0] = 0.0;
    
    struct Edge {
        int u, v;
        double weight;
    };
    std::vector<Edge> mst_edges;
    mst_edges.reserve(n - 1);
    
    // Step 1: Construct the Minimum Spanning Tree (Prim's Algorithm)
    for (int step = 0; step < n; ++step) {
        int u = -1;
        double d_min = 1e18;
        for (int i = 0; i < n; ++i) {
            if (!in_mst[i] && min_dist[i] < d_min) {
                d_min = min_dist[i];
                u = i;
            }
        }
        
        if (u == -1) break;
        in_mst[u] = true;
        
        if (parent[u] != -1) {
            mst_edges.push_back({parent[u], u, d_min});
        }
        
        for (int v = 0; v < n; ++v) {
            if (!in_mst[v]) {
                double d = euclidean_distance(objs[u], objs[v]);
                if (d < min_dist[v]) {
                    min_dist[v] = d;
                    parent[v] = u;
                }
            }
        }
    }
    
    // Step 2 & 3: Locate and delete the longest edge to find components
    int max_edge_idx = 0;
    double max_weight = -1.0;
    for (size_t i = 0; i < mst_edges.size(); ++i) {
        if (mst_edges[i].weight > max_weight) {
            max_weight = mst_edges[i].weight;
            max_edge_idx = i;
        }
    }
    
    std::vector<std::vector<int>> adj(n);
    for (size_t i = 0; i < mst_edges.size(); ++i) {
        if ((int)i == max_edge_idx) continue;
        adj[mst_edges[i].u].push_back(mst_edges[i].v);
        adj[mst_edges[i].v].push_back(mst_edges[i].u);
    }
    
    std::vector<bool> visited(n, false);
    std::vector<int> q;
    q.reserve(n);
    int start_node = mst_edges[max_edge_idx].u;
    q.push_back(start_node);
    visited[start_node] = true;
    
    size_t head = 0;
    while (head < q.size()) {
        int curr = q[head++];
        for (int neighbor : adj[curr]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push_back(neighbor);
            }
        }
    }
    
    std::vector<Object> left_objs, right_objs;
    left_objs.reserve(q.size());
    right_objs.reserve(n - q.size());
    
    for (int i = 0; i < n; ++i) {
        if (visited[i]) {
            left_objs.push_back(objs[i]);
        } else {
            right_objs.push_back(objs[i]);
        }
    }
    
    // Step 4: minMax strategy to find the optimal group center (pivot)
    auto choose_pivot_and_radius = [](const std::vector<Object>& group, Object& pivot, double& radius) {
        int best_p_idx = 0;
        double min_max_d = 1e18;
        int m = group.size();
        if (m == 1) {
            pivot = group[0];
            radius = 0.0;
            return;
        }
        
        for (int i = 0; i < m; ++i) {
            double current_max_d = 0.0;
            for (int j = 0; j < m; ++j) {
                if (i == j) continue;
                double d = euclidean_distance(group[i], group[j]);
                if (d > current_max_d) {
                    current_max_d = d;
                }
            }
            if (current_max_d < min_max_d) {
                min_max_d = current_max_d;
                best_p_idx = i;
            }
        }
        pivot = group[best_p_idx];
        radius = min_max_d;
    };
    
    Object left_p, right_p;
    double left_r = 0.0, right_r = 0.0;
    choose_pivot_and_radius(left_objs, left_p, left_r);
    choose_pivot_and_radius(right_objs, right_p, right_r);
    
    STNode* node = new STNode();
    node->is_leaf = false;
    node->left_pivot = left_p;
    node->right_pivot = right_p;
    node->left_radius = left_r;
    node->right_radius = right_r;
    
    node->left = build_slim_tree_rec(left_objs, max_leaf_capacity);
    node->right = build_slim_tree_rec(right_objs, max_leaf_capacity);
    
    return node;
}

STNode* build_slim_tree(const std::vector<Object>& dataset, int max_leaf_capacity = 4) {
    return build_slim_tree_rec(dataset, max_leaf_capacity);
}