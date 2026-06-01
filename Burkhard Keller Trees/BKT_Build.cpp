#include "dataset.h"

std::unique_ptr<BKTNode> build_bkt(std::vector<Object>& objects) {
    if (objects.empty()) {
        return nullptr;
    }

    Object pivot = objects.back();
    objects.pop_back();

    auto root = std::make_unique<BKTNode>(pivot);
    std::unordered_map<int, std::vector<Object>> partitions;
    
    for (const auto& obj : objects) {
        int dist = discrete_distance(obj, pivot);
        partitions[dist].push_back(obj);
    }

    objects.clear();

    for (auto& pair : partitions) {
        root->children[pair.first] = build_bkt(pair.second);
    }

    return root;
}