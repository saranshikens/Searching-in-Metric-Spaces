#include "dataset.h"
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <limits>
#include <iomanip>
using namespace std;
using namespace chrono;

#define D_MAX 10 // Dimension
#define N_MAX 200 // Max points per node
#define K_MAX 20 // Max k for k-NN

// ---------------------- Stats ----------------------
long long distCompsBuild = 0;
long long distCompsSearch = 0;
long long pivotCount = 1;
int metricType = 0; // 0-L2, 1-L1, 2-L_Inf

struct Point{
    float coords[D_MAX];
};

struct TreeNode{
    Point pivotA;
    Point pivotB;
    Point bucket[N_MAX];
    int bucketSize;
    TreeNode* left;
    TreeNode* right;
    bool isLeaf;

    TreeNode(Point a, Point b){
        pivotA = a;
        pivotB = b;
        left = right = nullptr;
        isLeaf = false;
        bucketSize = 0;
    }

    TreeNode(Point arr[], int n){
        for(int i=0; i<n; i++) bucket[i] = arr[i];
        bucketSize = n;
        left = right = nullptr;
        isLeaf = true;
    }
};

// ---------------------- Distance ----------------------
float distance(Point x, Point y, bool duringBuild = false) {
    if (duringBuild) distCompsBuild++;
    else distCompsSearch++;

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

TreeNode* buildGHT(Point arr[], int n, int leaf_size = 4, Point* reusedPivot = nullptr){
    if(n<=0) return nullptr;
    if(n<=leaf_size) return new TreeNode(arr, n);

    // one pivot is reused
    int idA, idB;
    Point pA, pB;
    if(reusedPivot==nullptr){
        idA = rand()%n;
        idB = rand()%n;
        while(idA==idB) idB = rand()%n;
        pA = arr[idA];
        pB = arr[idB];
    }
    else{
        pA = *reusedPivot;
        idB = rand()%n;
        pB = arr[idB];
    }
    pivotCount++;
    
    TreeNode* node = new TreeNode(pA, pB);

    Point leftPartition[N_MAX], rightPartition[N_MAX];
    int leftN = 0, rightN = 0;

    for(int i=0; i<n; i++){
        if(i==idA || i==idB) continue;
        float dA = distance(arr[i], pA);
        float dB = distance(arr[i], pB);
        distCompsBuild += 2;
        if(dA<=dB) leftPartition[leftN++] = arr[i];
        else rightPartition[rightN++] = arr[i];
    }

    if(leftN==0 || rightN==0){
        rightN = 0;
        leftN = 0;
        for(int i=0; i<n; i++){
            if(i==idA || i==idB) continue;
            if(i%2==0) leftPartition[leftN++] = arr[i];
            else rightPartition[rightN++] = arr[i];
        }
        if(leftN+rightN==0) return new TreeNode(arr, n);
    }

    node->left = buildGHT(leftPartition, leftN, leaf_size);
    node->right = buildGHT(rightPartition, rightN, leaf_size);
    return node;
}

void updateBestK(Point bestPoints[], float bestDists[], int k, const Point &cand, float d){
    int worst = 0;
    for(int i=1; i<k; i++){
        if(bestDists[i]>bestDists[worst]) worst = i;
    }
        
    if(d<bestDists[worst]){
        bestDists[worst] = d;
        bestPoints[worst] = cand;
    }
}

float currentRadius(float bestDists[], int k){
    float maxd = bestDists[0];
    for(int i=1; i<k; i++){
         if(bestDists[i]>maxd) maxd = bestDists[i];
    }
       
    return maxd;
}

// ---------------------- Search ----------------------
void search1NN(TreeNode* node, const Point &q, Point &bestPoint, float &bestDist) {
    if (node == nullptr) return;

    if (node->isLeaf) {
        for (int i = 0; i < node->bucketSize; i++) {
            float d = distance(q, node->bucket[i]);
            if (d < bestDist) {
                bestDist = d;
                bestPoint = node->bucket[i];
            }
        }
        return;
    }

    float dA = distance(q, node->pivotA);
    float dB = distance(q, node->pivotB);
    distCompsSearch += 2;

    if (dA < bestDist) { bestDist = dA; bestPoint = node->pivotA; }
    if (dB < bestDist) { bestDist = dB; bestPoint = node->pivotB; }

    bool preferLeft = (dA <= dB);
    if (preferLeft) {
        search1NN(node->left, q, bestPoint, bestDist);
        if (fabs(dA - dB) <= bestDist) search1NN(node->right, q, bestPoint, bestDist);
    } else {
        search1NN(node->right, q, bestPoint, bestDist);
        if (fabs(dA - dB) <= bestDist) search1NN(node->left, q, bestPoint, bestDist);
    }
}


void printPoint(Point p){
    cout<<"("<<setprecision(2);
    for(int i=0; i<D_MAX; i++){
        cout<<p.coords[i];
        if(i<D_MAX-1) cout<<", ";
    }
    cout<<")";
}


#define ITERATIONS 2000   // Hyperparameter: number of repetitions for averaging

int main() {
    Point points[N_MAX];
    for (int i = 0; i < N_MAX; i++)
        for (int j = 0; j < D_MAX; j++)
            points[i].coords[j] = DATASET[i][j];

    mt19937 rng((unsigned)time(0));
    uniform_real_distribution<float> dist(-10.0f, 10.0f);

    double totalBuildTime = 0, totalSearchTime = 0;
    long long totalDistBuild = 0, totalDistSearch = 0, totalPivots = 0;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        distCompsBuild = distCompsSearch = pivotCount = 0;

        auto build_start = high_resolution_clock::now();
        TreeNode* root = buildGHT(points, N_MAX, 4);
        auto build_end = high_resolution_clock::now();
        totalBuildTime += duration_cast<microseconds>(build_end - build_start).count();

        Point q;
        for (int j = 0; j < D_MAX; j++) q.coords[j] = dist(rng);
        Point bestPoint;
        float bestDist = numeric_limits<float>::infinity();

        auto search_start = high_resolution_clock::now();
        search1NN(root, q, bestPoint, bestDist);
        auto search_end = high_resolution_clock::now();
        totalSearchTime += duration_cast<microseconds>(search_end - search_start).count();

        totalDistBuild += distCompsBuild;
        totalDistSearch += distCompsSearch;
        totalPivots += pivotCount;

        delete root;
    }

    cout << fixed << setprecision(2);
    cout << "\nAveraged over " << ITERATIONS << " iterations:\n";
    cout << "Average build time: " << (totalBuildTime / ITERATIONS) << " µs\n";
    cout << "Average 1-NN search time: " << (totalSearchTime / ITERATIONS) << " µs\n";
    cout << "Average build distance computations: " << (totalDistBuild / ITERATIONS) << "\n";
    cout << "Average search distance computations: " << (totalDistSearch / ITERATIONS) << "\n";
    cout << "Average pivots used: " << (totalPivots / ITERATIONS) << "\n";

    // Optional: one example run
    TreeNode* root = buildGHT(points, N_MAX, 4);
    Point q;
    for (int j = 0; j < D_MAX; j++) q.coords[j] = dist(rng);
    Point bestPoint;
    float bestDist = numeric_limits<float>::infinity();
    search1NN(root, q, bestPoint, bestDist);

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

