#include <iostream>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
using namespace std;

#define D 10           // Dimension
#define N_MAX 200      // Max points per node
#define M 4            // Number of pivots per node
#define K_MAX 10       // Max k for k-NN

struct Point {
    float coords[D];
};

float L2(Point a, Point b) {
    float s = 0;
    for (int i = 0; i < D; i++)
        s += (a.coords[i] - b.coords[i]) * (a.coords[i] - b.coords[i]);
    return sqrtf(s);
}

struct GNATNode {
    Point pivots[M];
    float rangeLow[M][M];
    float rangeHigh[M][M];
    Point subset[M][N_MAX];
    int subsetSize[M];
    GNATNode* child[M];
    int m; // number of pivots in this node
    bool isLeaf;
    Point leafPoints[N_MAX];
    int leafCount;

    GNATNode() {
        m = 0;
        isLeaf = false;
        leafCount = 0;
        for (int i = 0; i < M; i++) {
            subsetSize[i] = 0;
            child[i] = nullptr;
        }
    }
};

// ------------------ GNAT BUILD ------------------
GNATNode* buildGNAT(Point arr[], int n, int leaf_size = 4) {
    if (n <= 0) return nullptr;
    GNATNode* node = new GNATNode();

    if (n <= leaf_size) {
        node->isLeaf = true;
        node->leafCount = n;
        for (int i = 0; i < n; i++) node->leafPoints[i] = arr[i];
        return node;
    }

    node->m = (n < M) ? n : M;

    // ---- pick m pivots (random for now) ----
    bool chosen[N_MAX] = {false};
    for (int i = 0; i < node->m; i++) {
        int id;
        do { id = rand() % n; } while (chosen[id]);
        chosen[id] = true;
        node->pivots[i] = arr[id];
    }

    // ---- assign each point to nearest pivot ----
    for (int i = 0; i < n; i++) {
        if (chosen[i]) continue;
        float best = L2(arr[i], node->pivots[0]);
        int bestIdx = 0;
        for (int j = 1; j < node->m; j++) {
            float d = L2(arr[i], node->pivots[j]);
            if (d < best) {
                best = d;
                bestIdx = j;
            }
        }
        node->subset[bestIdx][node->subsetSize[bestIdx]++] = arr[i];
    }

    // ---- compute distance range tables ----
    for (int i = 0; i < node->m; i++) {
        for (int j = 0; j < node->m; j++) {
            if (i == j) {
                node->rangeLow[i][j] = 0;
                node->rangeHigh[i][j] = 0;
            } else {
                float minD = numeric_limits<float>::infinity();
                float maxD = 0;
                for (int k = 0; k < node->subsetSize[j]; k++) {
                    float d = L2(node->pivots[i], node->subset[j][k]);
                    if (d < minD) minD = d;
                    if (d > maxD) maxD = d;
                }
                float dpp = L2(node->pivots[i], node->pivots[j]);
                if (dpp < minD) minD = dpp;
                if (dpp > maxD) maxD = dpp;
                node->rangeLow[i][j] = minD;
                node->rangeHigh[i][j] = maxD;
            }
        }
    }

    // ---- recursively build children ----
    for (int i = 0; i < node->m; i++)
        node->child[i] = buildGNAT(node->subset[i], node->subsetSize[i], leaf_size);

    return node;
}

// ------------------ K-NN SEARCH ------------------

void updateBest(Point candidate, float d, Point bestPts[], float bestDist[], int k) {
    int worst = 0;
    for (int i = 1; i < k; i++)
        if (bestDist[i] > bestDist[worst]) worst = i;

    if (d < bestDist[worst]) {
        bestDist[worst] = d;
        bestPts[worst] = candidate;
    }
}

void knnSearch(GNATNode* node, Point q, int k, Point bestPts[], float bestDist[]) {
    if (!node) return;

    if (node->isLeaf) {
        for (int i = 0; i < node->leafCount; i++) {
            float d = L2(q, node->leafPoints[i]);
            updateBest(node->leafPoints[i], d, bestPts, bestDist, k);
        }
        return;
    }

    float distPivot[M];
    for (int i = 0; i < node->m; i++) distPivot[i] = L2(q, node->pivots[i]);

    // Check each pivot
    for (int i = 0; i < node->m; i++)
        updateBest(node->pivots[i], distPivot[i], bestPts, bestDist, k);

    // Check subtrees with pruning
    for (int i = 0; i < node->m; i++) {
        float best = bestDist[0];
        for (int j = 1; j < k; j++)
            if (bestDist[j] < best) best = bestDist[j];

        bool prune = false;
        for (int j = 0; j < node->m; j++) {
            if (distPivot[j] - best > node->rangeHigh[j][i] ||
                distPivot[j] + best < node->rangeLow[j][i]) {
                prune = true;
                break;
            }
        }
        if (!prune) knnSearch(node->child[i], q, k, bestPts, bestDist);
    }
}

// ------------------ MAIN ------------------
int main() {
    srand(time(0));
    Point points[200];
    for (int i = 0; i < 200; i++)
        for (int j = 0; j < D; j++)
            points[i].coords[j] = -10.0f + static_cast<float>(rand()) / RAND_MAX * 20.0f;

    GNATNode* root = buildGNAT(points, 200);

    Point q;
    for (int j = 0; j < D; j++) q.coords[j] = -10.0f + static_cast<float>(rand()) / RAND_MAX * 20.0f;

    int k = 3;
    Point bestPts[K_MAX];
    float bestDist[K_MAX];
    for (int i = 0; i < k; i++) bestDist[i] = numeric_limits<float>::infinity();

    knnSearch(root, q, k, bestPts, bestDist);

    cout << "Query: ";
    for (int j = 0; j < D; j++) cout << q.coords[j] << " ";
    cout << "\n\nTop " << k << " nearest neighbors:\n";
    for (int i = 0; i < k; i++) {
        cout << i + 1 << ": ";
        for (int j = 0; j < D; j++) cout << bestPts[i].coords[j] << " ";
        cout << " | dist = " << bestDist[i] << "\n";
    }
}
