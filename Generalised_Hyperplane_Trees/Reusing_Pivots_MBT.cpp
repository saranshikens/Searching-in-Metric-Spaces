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
#define K_MAX 10 // Max k for k-NN

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

float L2(Point x, Point y){
    float d = 0;
    for(int i=0; i<D_MAX; i++){
        d += (x.coords[i] - y.coords[i])*(x.coords[i] - y.coords[i]);
    }
    return sqrtf(d);
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
    
    TreeNode* node = new TreeNode(pA, pB);

    Point leftPartition[N_MAX], rightPartition[N_MAX];
    int leftN = 0, rightN = 0;

    for(int i=0; i<n; i++){
        if(i==idA || i==idB) continue;
        float dA = L2(arr[i], pA);
        float dB = L2(arr[i], pB);
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

void search(TreeNode* node, const Point &q, Point bestPoints[], float bestDists[], int k){
    if(node==nullptr) return;

    if(node->isLeaf){
        for(int i=0; i<node->bucketSize; i++){
            float d = L2(q, node->bucket[i]);
            updateBestK(bestPoints, bestDists, k, node->bucket[i], d);
        }
        return;
    }

    float d1 = L2(q, node->pivotA);
    float d2 = L2(q, node->pivotB);

    updateBestK(bestPoints, bestDists, k, node->pivotA, d1);
    updateBestK(bestPoints, bestDists, k, node->pivotB, d2);

    float r = currentRadius(bestDists, k);

    bool visitLeft = (d1 - r <= d2 + r);
    bool visitRight = (d1 + r >= d2 - r);
    bool preferLeft = (d1 <= d2);

    if(preferLeft){
        if(visitLeft) search(node->left, q, bestPoints, bestDists, k);
        r = currentRadius(bestDists, k);
        if(visitRight && (d1 + r >= d2 - r)) search(node->right, q, bestPoints, bestDists, k);
    } 
    else{
        if(visitRight) search(node->right, q, bestPoints, bestDists, k);
        r = currentRadius(bestDists, k);
        if(visitLeft && (d1 - r <= d2 + r)) search(node->left, q, bestPoints, bestDists, k);
    }
}

void searchK(TreeNode* root, const Point &q, Point bestPoints[], float bestDists[], int k) {
    for(int i=0; i<k; i++) bestDists[i] = numeric_limits<float>::infinity();
    search(root, q, bestPoints, bestDists, k);
}

void printPoint(Point p){
    cout<<"("<<setprecision(2);
    for(int i=0; i<D_MAX; i++){
        cout<<p.coords[i];
        if(i<D_MAX-1) cout<<", ";
    }
    cout<<")";
}

/*int main() {
    srand((unsigned)time(0));
    Point points[N_MAX];
    mt19937 rng((unsigned)time(0));
    uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for(int i=0; i<N_MAX; i++){
        for(int j=0; j<D_MAX; j++){
            points[i].coords[j] = dist(rng);
        }
    }      

    TreeNode* root = buildGHT(points, N_MAX, 4);

    Point q;
    for(int j=0; j<D_MAX; j++) q.coords[j] = dist(rng);

    int k = 5;
    Point bestPoints[K_MAX];
    float bestDists[K_MAX];

    searchK(root, q, bestPoints, bestDists, k);

    cout<<"Query point:"<<endl;
    printPoint(q);
    cout<<endl;
    cout<<"\n"<<k<<" nearest neighbors:"<<endl;
    for(int i=0; i<k; i++){
        cout<<i+1<<". ";
        printPoint(bestPoints[i]);
        cout<<"  dist="<<bestDists[i]<<endl;
    }
}*/

#define ITERATIONS 10   // Hyperparameter: number of repetitions for averaging

int main() {
    Point points[N_MAX];
    for (int i = 0; i < N_MAX; i++) {
        for (int j = 0; j < D_MAX; j++) {
            points[i].coords[j] = DATASET[i][j];
        }
    }

    double total_build_time = 0.0;
    double total_search_time = 0.0;

    mt19937 rng((unsigned)time(0));
    uniform_real_distribution<float> dist(-10.0f, 10.0f);

    int k = 5;
    Point bestPoints[K_MAX];
    float bestDists[K_MAX];

    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Build tree
        auto build_start = high_resolution_clock::now();
        TreeNode* root = buildGHT(points, N_MAX, 4);
        auto build_end = high_resolution_clock::now();
        auto build_time =
            duration_cast<microseconds>(build_end - build_start).count();
        total_build_time += build_time;

        // Generate a random query
        Point q;
        for (int j = 0; j < D_MAX; j++) q.coords[j] = dist(rng);

        // Search kNN
        auto search_start = high_resolution_clock::now();
        searchK(root, q, bestPoints, bestDists, k);
        auto search_end = high_resolution_clock::now();
        auto search_time =
            duration_cast<microseconds>(search_end - search_start).count();
        total_search_time += search_time;

        delete root; // free tree between iterations (important!)
    }

    cout << fixed << setprecision(2);
    cout << "\nAveraged over " << ITERATIONS << " iterations:\n";
    cout << "Average build time: " << (total_build_time / ITERATIONS)
         << " microseconds\n";
    cout << "Average search time: " << (total_search_time / ITERATIONS)
         << " microseconds\n";

    // Optional: run one final search to show actual neighbors
    Point q;
    for (int j = 0; j < D_MAX; j++) q.coords[j] = dist(rng);
    TreeNode* root = buildGHT(points, N_MAX, 4);
    searchK(root, q, bestPoints, bestDists, k);

    cout << "\nQuery point:\n";
    printPoint(q);
    cout << "\n\n" << k << " nearest neighbors:\n";
    for (int i = 0; i < k; i++) {
        cout << i + 1 << ". ";
        printPoint(bestPoints[i]);
        cout << "  dist=" << bestDists[i] << "\n";
    }

    delete root;
}
