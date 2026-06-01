#include "dataset.h"

// Recursively builds the BKT index structure
std::unique_ptr<BKTNode> build_bkt(std::vector<Object>& objects) {
    if (objects.empty()) {
        return nullptr;
    }

    // Pick an arbitrary object as the pivot (using the last item for O(1) pop)
    Object pivot = objects.back();
    objects.pop_back();

    auto root = std::make_unique<BKTNode>(pivot);

    // Partition remaining items into buckets based on discrete distance
    std::unordered_map<int, std::vector<Object>> partitions;
    for (const auto& obj : objects) {
        int dist = discrete_distance(obj, pivot);
        partitions[dist].push_back(obj);
    }

    // Clear objects vector to free space during recursion
    objects.clear();

    // Recursively build children subtrees for every non-empty set Xi
    for (auto& pair : partitions) {
        int distance_value = pair.first;
        root->children[distance_value] = build_bkt(pair.second);
    }

    return root;
}