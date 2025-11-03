#include "dataset.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <limits>
#include <cstdlib>
#include <ctime>
using namespace std;
using namespace chrono;

#define D 10 // Dimension
#define N_MAX 200 // Max points per node
#define M 4 // Number of pivots per node
#define K_MAX 10 // Max k for k-NN

struct Point{
    float coords[D];
};

float L2(Point a, Point b){
    float s = 0;
    for(int i=0; i<D; i++){
        s += (a.coords[i]-b.coords[i])*(a.coords[i]-b.coords[i]);
    }
    return sqrtf(s);
}

struct GNATNode{
    Point pivots[M];
    float rangeLow[M][M];
    float rangeHigh[M][M];
    Point subset[M][N_MAX];
    int subsetSize[M];
    GNATNode* child[M];
    int m;
    bool isLeaf;
    Point leafPoints[N_MAX];
    int leafCount;

    GNATNode(){
        m = 0;
        isLeaf = false;
        leafCount = 0;
        for(int i=0; i<M; i++){
            subsetSize[i] = 0;
            child[i] = nullptr;
        }
    }
};

// Farthest-first pivot selection 
void selectPivots(Point arr[], int n, Point pivots[], int& m){
    if(n==0){
        m = 0;
        return;
    }

    m = max(2, min(M, n/20)); // proportional pivot count
    bool chosen[N_MAX] = {false};

    int first = rand()%n;
    pivots[0] = arr[first];
    chosen[first] = true;

    // second pivot: farthest from first
    float maxDist = -1;
    int farthest = -1;
    for(int i=0; i<n; i++){
        if(i==first) continue;
        float d = L2(arr[i], pivots[0]);
        if(d>maxDist){
            maxDist = d;
            farthest = i;
        }
    }
    pivots[1] = arr[farthest];
    chosen[farthest] = true;

    int count = 2;
    // select remaining pivots greedily: farthest from all chosen pivots
    while(count<m){
        float bestMinDist = -1;
        int bestIdx = -1;
        for(int i=0; i<n; i++){
            if(chosen[i]) continue;
            float minDist = numeric_limits<float>::infinity();
            for(int j=0; j<count; j++){
                float d = L2(arr[i], pivots[j]);
                if(d<minDist) minDist = d;
            }
            if(minDist>bestMinDist){
                bestMinDist = minDist;
                bestIdx = i;
            }
        }
        if(bestIdx==-1) break;
        pivots[count++] = arr[bestIdx];
        chosen[bestIdx] = true;
    }
    m = count;
}


GNATNode* buildGNAT(Point arr[], int n, int leaf_size = 4){
    if(n<=0) return nullptr;
    GNATNode* node = new GNATNode();

    if(n<=leaf_size){
        node->isLeaf = true;
        node->leafCount = n;
        for(int i=0; i<n; i++) node->leafPoints[i] = arr[i];
        return node;
    }

    selectPivots(arr, n, node->pivots, node->m);

    // Assign each point to nearest pivot
    for(int i=0; i<n; i++){
        float best = L2(arr[i], node->pivots[0]);
        int bestIdx = 0;
        for(int j=1; j<node->m; j++){
            float d = L2(arr[i], node->pivots[j]);
            if(d<best){
                best = d;
                bestIdx = j;
            }
        }
        node->subset[bestIdx][node->subsetSize[bestIdx]++] = arr[i];
    }

    // Compute range tables
    for(int i=0; i<node->m; i++){
        for(int j=0; j<node->m; j++){
            if(i==j) node->rangeLow[i][j] = node->rangeHigh[i][j] = 0;
            else{
                float minD = numeric_limits<float>::infinity();
                float maxD = 0;
                for(int k=0; k<node->subsetSize[j]; k++){
                    float d = L2(node->pivots[i], node->subset[j][k]);
                    if(d<minD) minD = d;
                    if(d>maxD) maxD = d;
                }
                float dp = L2(node->pivots[i], node->pivots[j]);
                if(dp<minD) minD = dp;
                if(dp>maxD) maxD = dp;
                node->rangeLow[i][j] = minD;
                node->rangeHigh[i][j] = maxD;
            }
        }
    }

    for(int i=0; i<node->m; i++){
        node->child[i] = buildGNAT(node->subset[i], node->subsetSize[i], leaf_size);
    }
        
    return node;
}


void updateBest(Point candidate, float d, Point bestPts[], float bestDist[], int k){
    int worst = 0;
    for(int i=1; i<k; i++){
        if(bestDist[i]>bestDist[worst]) worst = i;
    }
        
    if(d<bestDist[worst]){
        bestDist[worst] = d;
        bestPts[worst] = candidate;
    }
}

void knnSearch(GNATNode* node, Point q, int k, Point bestPts[], float bestDist[]){
    if(!node) return;

    if(node->isLeaf){
        for(int i=0; i<node->leafCount; i++){
            float d = L2(q, node->leafPoints[i]);
            updateBest(node->leafPoints[i], d, bestPts, bestDist, k);
        }
        return;
    }

    float distPivot[M];
    for(int i=0; i<node->m; i++){
        distPivot[i] = L2(q, node->pivots[i]);
    }
        
    for(int i = 0; i<node->m; i++){
        updateBest(node->pivots[i], distPivot[i], bestPts, bestDist, k);
    }
        
    for(int i=0; i<node->m; i++){
        float best = bestDist[0];
        for(int j=1; j<k; j++){
            if(bestDist[j]<best) best = bestDist[j];
        }
            
        bool prune = false;
        for(int j=0; j<node->m; j++){
            if(distPivot[j]-best > node->rangeHigh[j][i] || distPivot[j] + best < node->rangeLow[j][i]){
                prune = true;
                break;
            }
        }
        if(!prune) knnSearch(node->child[i], q, k, bestPts, bestDist);
    }
}


#define ITERATIONS 10   // hyperparameter: number of repetitions for averaging

int main() {
    srand(time(0));

    // --- Generate dataset ---
    Point points[N_MAX];
    for (int i = 0; i < N_MAX; i++) {
        for (int j = 0; j < D_MAX; j++) {
            points[i].coords[j] = DATASET[i][j];
        }
    }

    double total_build_time = 0.0;
    double total_search_time = 0.0;

    int k = 3;
    Point bestPts[K_MAX];
    float bestDist[K_MAX];

    for (int iter = 0; iter < ITERATIONS; iter++) {
        // --- Build tree ---
        auto build_start = chrono::high_resolution_clock::now();
        GNATNode* root = buildGNAT(points, N_MAX);
        auto build_end = chrono::high_resolution_clock::now();
        auto build_time = chrono::duration_cast<chrono::microseconds>(build_end - build_start).count();
        total_build_time += build_time;

        // --- Generate query point ---
        Point q;
        for (int j = 0; j < D; j++) {
            q.coords[j] = -10.0f + static_cast<float>(rand()) / RAND_MAX * 20.0f;
        }

        for (int i = 0; i < k; i++) {
            bestDist[i] = numeric_limits<float>::infinity();
        }

        // --- Perform kNN search ---
        auto search_start = chrono::high_resolution_clock::now();
        knnSearch(root, q, k, bestPts, bestDist);
        auto search_end = chrono::high_resolution_clock::now();
        auto search_time = chrono::duration_cast<chrono::microseconds>(search_end - search_start).count();
        total_search_time += search_time;

        delete root; // free tree
    }

    cout << fixed << setprecision(2);
    cout << "\nAveraged over " << ITERATIONS << " iterations:\n";
    cout << "Average build time: " << (total_build_time / ITERATIONS) << " microseconds\n";
    cout << "Average search time: " << (total_search_time / ITERATIONS) << " microseconds\n";
}


