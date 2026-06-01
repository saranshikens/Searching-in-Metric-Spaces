#include "SAT.h"
#include <algorithm>

struct PointDistance {
    Point point;
    double dist;
};

SATNode* buildSAT(std::vector<Point>& X) {
    if (X.empty()) return nullptr;

    // Select the first element arbitrarily as the pivot root
    Point p = X[0];
    SATNode* node = new SATNode(p);

    if (X.size() == 1) {
        node->rc = 0.0;
        return node;
    }

    // Measure distances from p to all other items in X \ {p}
    std::vector<PointDistance> S;
    S.reserve(X.size() - 1);
    double max_d = 0.0;

    for (size_t i = 1; i < X.size(); ++i) {
        double d = euclidean_distance(p, X[i]);
        S.push_back({X[i], d});
        if (d > max_d) {
            max_d = d;
        }
    }

    // Core property: covering radius tracks the maximum localized point distance
    node->rc = max_d;

    // Sort items by proximity to pivot to evaluate Navarro's closest-first heuristic
    std::sort(S.begin(), S.end(), [](const PointDistance& a, const PointDistance& b) {
        return a.dist < b.dist;
    });

    // Greedy choice of spatial neighbors N(p)
    std::vector<Point> neighbors;
    for (const auto& item : S) {
        bool is_neighbor = true;
        for (const auto& nb : neighbors) {
            if (item.dist >= euclidean_distance(item.point, nb)) {
                is_neighbor = false;
                break;
            }
        }
        if (is_neighbor) {
            neighbors.push_back(item.point);
        }
    }

    if (neighbors.empty()) return node;

    // Organize sub-buckets; each neighbor initiates its own branch
    std::vector<std::vector<Point>> buckets(neighbors.size());
    for (size_t i = 0; i < neighbors.size(); ++i) {
        buckets[i].push_back(neighbors[i]);
    }

    // Map remaining outer elements to their closest local neighbor
    for (const auto& item : S) {
        bool is_nb = false;
        for (size_t i = 0; i < neighbors.size(); ++i) {
            if (item.point.id == neighbors[i].id) {
                is_nb = true;
                break;
            }
        }
        if (is_nb) continue;

        double min_d = -1.0;
        size_t closest_idx = 0;
        for (size_t i = 0; i < neighbors.size(); ++i) {
            double d = euclidean_distance(item.point, neighbors[i]);
            if (min_d < 0 || d < min_d) {
                min_d = d;
                closest_idx = i;
            }
        }
        buckets[closest_idx].push_back(item.point);
    }

    // Recursively build out the graph approximation downwards
    node->children.reserve(neighbors.size());
    for (size_t i = 0; i < buckets.size(); ++i) {
        SATNode* childNode = buildSAT(buckets[i]);
        if (childNode) {
            node->children.push_back(childNode);
        }
    }

    return node;
}