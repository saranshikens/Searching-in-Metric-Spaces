#include "dataset.h"
#include <limits>

// Executes the approximating and eliminating range search algorithm
std::vector<int> search_AESA(const std::vector<Point>& dataset, const AESAMatrix& matrix, const Point& q, double r) {
    int n = dataset.size();
    std::vector<bool> eliminated(n, false);
    std::vector<bool> visited(n, false);
    std::vector<double> LB(n, 0.0);
    std::vector<int> results;

    while (true) {
        int next_pivot = -1;
        double min_lb = std::numeric_limits<double>::max();

        // Heuristic: Select the unvisited non-eliminated element closest to q via minimum lower-bound 
        for (int i = 0; i < n; ++i) {
            if (!eliminated[i] && !visited[i]) {
                if (LB[i] < min_lb) {
                    min_lb = LB[i];
                    next_pivot = i;
                }
            }
        }

        // Break loop when all remaining valid items have been evaluated or pruned
        if (next_pivot == -1) {
            break;
        }

        // Evaluate true metric distance to query
        double dq_p = euclidean_distance(q, dataset[next_pivot]);
        visited[next_pivot] = true;

        if (dq_p <= r) {
            results.push_back(next_pivot);
        }

        // Eliminate and update lower bounds utilizing precomputed matrix paths
        for (int i = 0; i < n; ++i) {
            if (!eliminated[i] && !visited[i]) {
                double dp_o = matrix.get_distance(next_pivot, i);
                double current_bound = std::abs(dp_o - dq_p);
                
                LB[i] = std::max(LB[i], current_bound);
                if (LB[i] > r) {
                    eliminated[i] = true;
                }
            }
        }
    }

    return results;
}