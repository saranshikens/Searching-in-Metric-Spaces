#ifndef SAT_H
#define SAT_H

#include "dataset.h"
#include <vector>

struct SATNode {
    Point pivot;
    double rc; // Covering radius
    std::vector<SATNode*> children;

    SATNode(const Point& p) : pivot(p), rc(0.0) {}
    
    ~SATNode() {
        for (SATNode* child : children) {
            delete child;
        }
    }
};

// Decoupled structural signatures
SATNode* buildSAT(std::vector<Point>& X);
void rangeSearchSAT(SATNode* node, const Point& q, double r, double d_q_p, double min_d_ancestors_neighbors, std::vector<Point>& results);

#endif // SAT_H