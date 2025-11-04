#include "dataset.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <limits>
#include <random>
#include <cstdlib>
#include <ctime>
using namespace std;
using namespace chrono;

#define D 10 // Dimension
#define N_MAX 200 // Max points per node
#define M 4 // Number of pivots per node
#define K_MAX 20 // Max k for k-NN

// --- Global statistics ---
long long distCompBuild = 0;
long long distCompSearch = 0;
long long pivotCount = 0;
int metricType = 0;

struct Point{
    float coords[D];
};

// ---------------------- Distance ----------------------
float distance(Point x, Point y) {
    float d = 0;
    if(metricType==0){
        for (int i = 0; i < D_MAX; i++) {
            float diff = x.coords[i] - y.coords[i];
            d += diff * diff;
        }
        return sqrtf(d);
    }
    else if(metricType==1){
        for (int i = 0; i < D_MAX; i++) {
            float diff = abs(x.coords[i] - y.coords[i]);
            d += diff;
        }
        return d;
    }
    else{
        for (int i = 0; i < D_MAX; i++) {
            float diff = x.coords[i] - y.coords[i];
            d += max(diff, d);
        }
        return d;
    }
    return -1;
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
        float d = distance(arr[i], pivots[0]);
        distCompBuild++;
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
                float d = distance(arr[i], pivots[j]);
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
    pivotCount += m;
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
        float best = distance(arr[i], node->pivots[0]);
        distCompBuild++;
        int bestIdx = 0;
        for(int j=1; j<node->m; j++){
            float d = distance(arr[i], node->pivots[j]);
            distCompBuild++;
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
                    float d = distance(node->pivots[i], node->subset[j][k]);
                    distCompBuild++;
                    if(d<minD) minD = d;
                    if(d>maxD) maxD = d;
                }
                float dp = distance(node->pivots[i], node->pivots[j]);
                distCompBuild++;
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

// --- 1-NN search ---
void nnSearch(GNATNode* node, const Point &q, Point &bestPt, float &bestDist) {
    if (!node) return;

    if (node->isLeaf) {
        for (int i = 0; i < node->leafCount; i++) {
            float d = distance(q, node->leafPoints[i]);
            distCompSearch++;
            if (d < bestDist) {
                bestDist = d;
                bestPt = node->leafPoints[i];
            }
        }
        return;
    }

    float distPivot[M];
    for (int i = 0; i < node->m; i++)
        distPivot[i] = distance(q, node->pivots[i]);
        distCompSearch++;

    for (int i = 0; i < node->m; i++) {
        if (distPivot[i] < bestDist) {
            bestDist = distPivot[i];
            bestPt = node->pivots[i];
        }
    }

    for (int i = 0; i < node->m; i++) {
        bool prune = false;
        for (int j = 0; j < node->m; j++) {
            if (distPivot[j] - bestDist > node->rangeHigh[j][i] ||
                distPivot[j] + bestDist < node->rangeLow[j][i]) {
                prune = true;
                break;
            }
        }
        if (!prune)
            nnSearch(node->child[i], q, bestPt, bestDist);
    }
}

// ---------------------- Utility ----------------------
void printPoint(Point p) {
    cout << "(" << fixed << setprecision(2);
    for (int i = 0; i < D_MAX; i++) {
        cout << p.coords[i];
        if (i < D_MAX - 1) cout << ", ";
    }
    cout << ")";
}

// --- Experiment driver ---
#define ITERATIONS 2000

int main() {

    Point points[N_MAX];
    for (int i = 0; i < N_MAX; i++)
        for (int j = 0; j < D; j++)
            points[i].coords[j] = DATASET[i][j];

    mt19937 rng((unsigned)time(0));
    uniform_real_distribution<float> dist(-10.0f, 10.0f);

    double total_build_time = 0.0;
    double total_search_time = 0.0;
    long long totalDistBuild = 0, totalDistSearch = 0, totalPivot = 0;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        distCompBuild = distCompSearch = pivotCount = 0;

        auto build_start = high_resolution_clock::now();
        GNATNode* root = buildGNAT(points, N_MAX);
        auto build_end = high_resolution_clock::now();
        total_build_time += duration_cast<microseconds>(build_end - build_start).count();

        Point q;
        for (int j = 0; j < D; j++)
            q.coords[j] = dist(rng);

        Point bestPt;
        float bestDist = numeric_limits<float>::infinity();

        auto search_start = high_resolution_clock::now();
        nnSearch(root, q, bestPt, bestDist);
        auto search_end = high_resolution_clock::now();
        total_search_time += duration_cast<microseconds>(search_end - search_start).count();

        totalDistBuild += distCompBuild;
        totalDistSearch += distCompSearch;
        totalPivot += pivotCount;

        delete root;
    }

    cout << fixed << setprecision(2);
    cout << "\nAveraged over " << ITERATIONS << " iterations:\n";
    cout << "Average build time: " << (total_build_time / ITERATIONS) << " microseconds\n";
    cout << "Average search time: " << (total_search_time / ITERATIONS) << " microseconds\n";
    cout << "Average distance computations (build): " << (totalDistBuild / ITERATIONS) << "\n";
    cout << "Average distance computations (search): " << (totalDistSearch / ITERATIONS) << "\n";
    cout << "Average pivot count: " << (totalPivot / ITERATIONS) << "\n";

    // Optional: one example run
    GNATNode* root = buildGNAT(points, N_MAX, 4);
    Point q;
    for (int j = 0; j < D_MAX; j++) q.coords[j] = dist(rng);
    Point bestPoint;
    float bestDist = numeric_limits<float>::infinity();
    nnSearch(root, q, bestPoint, bestDist);

    cout << "\nQuery point:\n"; printPoint(q);
    cout << "\nNearest neighbor:\n"; printPoint(bestPoint);
    cout << "\nDistance = " << bestDist << "\n";

    Point bestPointBrute;
    float bestDistBrute = numeric_limits<float>::infinity();
    auto search_start_brute = high_resolution_clock::now();
    for(int i=0; i<N_MAX; i++){
        float dist = distance(q, points[i]);
        if(dist < bestDistBrute){
            bestPointBrute = points[i];
            bestDistBrute = dist;
        }
    }
    auto search_end_brute = high_resolution_clock::now();
    auto totalSearchTimeBrute = duration_cast<microseconds>(search_end_brute - search_start_brute).count();
    cout << "\nACTUAL Nearest neighbor:\n"; printPoint(bestPointBrute);
    cout << "\nACTUAL Distance = " << bestDistBrute << "\n";
    cout << "\nTime taken = " << totalSearchTimeBrute << "microseconds"<<endl;

    delete root;
}




