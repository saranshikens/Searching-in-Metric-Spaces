#include "dataset.h"

std::vector<FQAElement> build_FQA(const std::vector<Vector>& dataset, const std::vector<Vector>& pivots, int h) {
    std::vector<FQAElement> fqa(dataset.size());
    
    // 1. Calculate the distance map path to all h pivots for every object
    for (size_t i = 0; i < dataset.size(); ++i) {
        fqa[i].obj = dataset[i];
        fqa[i].dists.resize(h);
        for (int j = 0; j < h; ++j) {
            fqa[i].dists[j] = discrete_distance(dataset[i], pivots[j]);
        }
    }

    // 2. Sort lexicographically by the distance paths (std::vector comparison does this out-of-the-box)
    std::sort(fqa.begin(), fqa.end(), [](const FQAElement& a, const FQAElement& b) {
        return a.dists < b.dists;
    });

    return fqa;
}