#include "dataset.h"
#include <iostream>
#include <chrono> // measure build and search time
#include <random> // generate pseudo random float numbers
#include <cmath>
#include <limits>
#include <iomanip> // set precision to 2
using namespace std;
using namespace chrono;

#define D 10 // dimension of data
#define N_MAX 200 // cardinality of dataset
#define ITERATIONS 2000 // average out results over 2000 iterations


// --------------------Global Counters---------------------
int computationsBuild = 0; // distance computations in building the GHT
int computationsSearch = 0; // distance computations in searching for the neighbour
int pivotCount = 0; // pivots in the GHT
// 0 - L2 distance
// 1 - L1 distance
// 2 - L_inf distance 
int metricType = 0; 


// ---------------------- Structures ----------------------
struct Point{
    float coords[D];
};

struct TreeNode{
    Point pivotA;
    Point pivotB;
    Point bucket[N_MAX]; // contains points in the partition corresponding to the TreeNode
    int bucketSize;
    TreeNode* left;
    TreeNode* right;
    bool isLeaf;

    TreeNode(Point a, Point b){ // constructor for internal nodes
        pivotA = a;
        pivotB = b;
        left = nullptr;
        right = nullptr;
        isLeaf = false;
        bucketSize = 0;
    }

    TreeNode(Point arr[], int n){ // constructor for leaf nodes
        for(int i=0; i<n; i++){
            bucket[i] = arr[i];
        } 
        bucketSize = n;
        left = right = nullptr;
        isLeaf = true;
    }
};


// ---------------------- Distance ----------------------
float distance(Point x, Point y){
    float d = 0;
    if(metricType==0){ // L2 distance
        for(int i=0; i<D; i++){
            float diff = x.coords[i]-y.coords[i];
            d += diff*diff;
        }
        return sqrtf(d);
    }
    else if(metricType==1){ // L1 distance
        for(int i=0; i<D; i++){
            float diff = abs(x.coords[i]-y.coords[i]);
            d += diff;
        }
        return d;
    }
    else{ // L_inf distance
        for(int i=0; i<D; i++){
            float diff = x.coords[i]-y.coords[i];
            d += max(diff, d);
        }
        return d;
    }
    return -1;
}


// ---------------------- Build ----------------------
TreeNode* buildGHT(Point arr[], int n, int leaf_size=4){ // paritioning will stop when the partition size reaches 4
    if(n<=0) return nullptr;
    if(n<=leaf_size) return new TreeNode(arr, n);

    // choosing pivots randomly
    int idA = rand()%n;
    int idB = rand()%n;
    while(idA==idB) idB = rand()%n;
    pivotCount += 2; // two pivots used

    Point pA = arr[idA], pB = arr[idB]; // pivots for the current TreeNode
    TreeNode* node = new TreeNode(pA, pB);

    Point leftPartition[N_MAX], rightPartition[N_MAX]; // partition of the dataset due to the pivots
    int leftN = 0, rightN = 0; // track index of the last elements in the partition arrays

    // partitioning the dataset
    for(int i=0; i<n; i++){
        if(i==idA || i==idB) continue; // skip the pivots while partitioning
        float dA = distance(arr[i], pA);
        float dB = distance(arr[i], pB);
        computationsBuild += 2;
        // points nearer to pA go to left paritition, rest go to right
        if(dA<=dB) leftPartition[leftN++] = arr[i]; 
        else rightPartition[rightN++] = arr[i];
    }

    /*if(leftN==0 || rightN==0){
        leftN = 0;
        rightN = 0;
        for(int i=0; i<n; i++){
            if(i==idA || i==idB) continue;
            if(i%2==0) leftPartition[leftN++] = arr[i];
            else rightPartition[rightN++] = arr[i];
        }
        if(leftN+rightN==0) return new TreeNode(arr, n);
    }*/
    if(leftN+rightN==0) return new TreeNode(arr, n); // if both partitions are empty (very rare), just return a leaf node

    // recursively build the tree
    node->left = buildGHT(leftPartition, leftN, leaf_size);
    node->right = buildGHT(rightPartition, rightN, leaf_size);
    return node;
}


// ---------------------- Search ----------------------
void search(TreeNode* node, const Point &q, Point &bestPoint, float &bestDist){
    if(node==nullptr) return;

    // if a leaf is encountered, simply explore the bucket for the nearest neighbor
    if(node->isLeaf){
        for(int i=0; i<node->bucketSize; i++){
            float d = distance(q, node->bucket[i]);
            if(d<bestDist){
                bestDist = d;
                bestPoint = node->bucket[i];
            }
        }
        return;
    }

    float dA = distance(q, node->pivotA);
    float dB = distance(q, node->pivotB);
    computationsSearch += 2;

    // tracking the nearest neighbour
    if(dA<bestDist){
        bestDist = dA;
        bestPoint = node->pivotA;
    }
    if(dB<bestDist){
        bestDist = dB;
        bestPoint = node->pivotB;
    }

    bool goLeft = (dA<=dB);
    if(goLeft){
        // equivalent to d(q,p1) - r <= d(q,p2) + r
        if(dA-bestDist <= dB+bestDist) search(node->left, q, bestPoint, bestDist);
        // check if right subtree could still have closer point
        if(dB-bestDist <= dA+bestDist) search(node->right, q, bestPoint, bestDist);
    } 
    else{
        if(dB-bestDist <= dA+bestDist) search(node->right, q, bestPoint, bestDist);
        if(dA-bestDist <= dB+bestDist) search(node->left, q, bestPoint, bestDist);
    }
}


void printPoint(Point p){
    cout<<"("<<fixed<<setprecision(2);
    for(int i=0; i<D; i++){
        cout<<p.coords[i];
        if(i<D-1) cout<<", ";
    }
    cout<<")";
}


int main(){
    // "importing" the dataset
    Point points[N_MAX];
    for(int i=0; i<N_MAX; i++){
        for(int j=0; j<D; j++){
            points[i].coords[j] = DATASET[i][j];
        }    
    }
        
    // generate pseudo-random float values
    mt19937 rng((unsigned)time(0));
    uniform_real_distribution<float> dist(-10.0f, 10.0f);

    double totalBuildTime = 0, totalSearchTime = 0;
    int totalDistBuild = 0, totalDistSearch = 0, totalPivots = 0;

    for(int iter=0; iter<ITERATIONS; iter++){
        computationsBuild = 0;
        computationsSearch = 0;
        pivotCount = 0;

        // measure time (in microseconds) to build the GHT
        auto build_start = high_resolution_clock::now();
        TreeNode* root = buildGHT(points, N_MAX, 4);
        auto build_end = high_resolution_clock::now();
        totalBuildTime += duration_cast<microseconds>(build_end - build_start).count();

        // generate the query point (need not be an element of the dataset)
        Point q;
        for (int j = 0; j < D; j++) q.coords[j] = dist(rng);
        Point bestPoint;
        float bestDist = numeric_limits<float>::infinity();

        // measure time (in microseconds) to search the GHT
        auto search_start = high_resolution_clock::now();
        search(root, q, bestPoint, bestDist);
        auto search_end = high_resolution_clock::now();
        totalSearchTime += duration_cast<microseconds>(search_end - search_start).count();

        totalDistBuild += computationsBuild
    ;
        totalDistSearch += computationsSearch;
        totalPivots += pivotCount;

        delete root;
    }

    cout<<fixed<<setprecision(2);
    cout<<"\nAveraged over "<<ITERATIONS<<" iterations:"<<endl;
    cout<<"Average build time: "<<(totalBuildTime/ITERATIONS)<<" microseconds"<<endl;
    cout<<"Average search time: "<<(totalSearchTime/ITERATIONS)<<" microseconds"<<endl;
    cout<<"Average distance computations in building: "<<(totalDistBuild/ITERATIONS)<<endl;
    cout<<"Average distance computations in searching: "<<(totalDistSearch/ITERATIONS)<<endl;
    cout<<"Average pivots used: "<<(totalPivots/ITERATIONS)<<endl;

    // a demo run
    TreeNode* root = buildGHT(points, N_MAX, 4);
    Point q;
    for(int j=0; j<D; j++) q.coords[j] = dist(rng);
    Point bestPoint;
    float bestDist = numeric_limits<float>::infinity();
    search(root, q, bestPoint, bestDist);

    cout<<"\nQuery point:"<<endl;
    printPoint(q);
    cout<<"\nNearest neighbor:"<<endl;
    printPoint(bestPoint);
    cout<<"\nDistance = "<<bestDist<<endl;

    Point bestPointBrute;
    float bestDistBrute = numeric_limits<float>::infinity();
    
    auto search_start_brute = high_resolution_clock::now();
    for(int i=0; i<N_MAX; i++){
        float dist = distance(q, points[i]);
        if(dist<bestDistBrute){
            bestPointBrute = points[i];
            bestDistBrute = dist;
        }
    }
    auto search_end_brute = high_resolution_clock::now();
    auto totalSearchTimeBrute = duration_cast<microseconds>(search_end_brute - search_start_brute).count();
    
    cout<<"\nActual Nearest neighbor:"<<endl;
    printPoint(bestPointBrute);
    cout<<"\nActual Distance = "<<bestDistBrute<<endl;
    cout<<"Time taken to brute force:"<<totalSearchTimeBrute<<" microseconds"<<endl;

    delete root;
}
