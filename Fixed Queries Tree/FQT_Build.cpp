#include "dataset.h"

// Helper to handle duplicate vector corner cases safely without infinite recursion
bool all_identical(const std::vector<Object>& objects) {
    if (objects.empty()) return true;
    for (size_t i = 1; i < objects.size(); ++i) {
        if (objects[i].features != objects[0].features) return false;
    }
    return true;
}

FQTNode* build_fqt_recursive(std::vector<Object>& current_objects, int level, std::vector<Object>& level_pivots) {
    FQTNode* node = new FQTNode();
    node->level = level;

    // Base Case: Terminal node reached or dataset elements are duplicate values
    if (current_objects.size() <= 1 || all_identical(current_objects)) {
        node->is_leaf = true;
        node->objects = current_objects;
        return node;
    }

    // Assign or retrieve the uniform global pivot tracking this tree depth level
    if (level >= (int)level_pivots.size()) {
        level_pivots.push_back(current_objects[0]); // Arbitrary baseline selector
    }
    node->level_pivot = level_pivots[level];

    // Partition all elements based on distance to the level-shared pivot
    std::unordered_map<int, std::vector<Object>> partitions;
    for (const auto& obj : current_objects) {
        int dist = discrete_distance(obj, node->level_pivot);
        partitions[dist].push_back(obj);
    }

    node->is_leaf = false;
    for (auto& pair : partitions) {
        node->children[pair.first] = build_fqt_recursive(pair.second, level + 1, level_pivots);
    }

    return node;
}

FQTNode* build_fqt(std::vector<Object>& dataset) {
    std::vector<Object> level_pivots;
    return build_fqt_recursive(dataset, 0, level_pivots);
}