#include "dataset.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <chrono> // measure build and search time
#include <random> // generate pseudo random float numbers
#include <iomanip> // set precision to 2

using namespace std;
using namespace chrono;

#define D 50 // dimension of data
#define N_MAX 2000 // cardinality of dataset
#define M 12 // no of pivots per internal node
#define ITERATIONS 2000 // average out results over 2000 iterations

// --------------------Global Counters---------------------
int computationsBuild = 0; // distance computations in building the GHT
int computationsSearch = 0; // distance computations in searching for the neighbour
int pivotCount = 0; // pivots in the GHT
// 0 - L2 distance
// 1 - L1 distance
// 2 - L_inf distance 
int metricType = 2; 

 
// ---------------------- Structures ----------------------
struct Point{
    float coords[D];
};

struct GNATNode{
    Point pivots[M];
    float rangeLow[M][M];
    float rangeHigh[M][M];
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
            child[i] = nullptr;
        }
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
            float diff = fabs(x.coords[i]-y.coords[i]);
            d += diff;
        }
        return d;
    }
    else{ // L_inf distance
        for(int i=0; i<D; i++){
            float diff = fabs(x.coords[i]-y.coords[i]);
            d = max(diff, d);
        }
        return d;
    }
    return -1;
}


// ---------------------- Build ----------------------
GNATNode* buildGNAT(Point arr[], int n, int leaf_size = 4) {
    if(n<=0) return nullptr;
    GNATNode* node = new GNATNode();

    if(n<=leaf_size){
        node->isLeaf = true;
        node->leafCount = n;
        for(int i=0; i<n; i++){
            node->leafPoints[i] = arr[i];
        }  
        return node;
    }

    node->m = (n<M)?n:M;
    pivotCount += node->m;

    // pick m pivots randomly
    bool* chosen = new bool[n];
    for(int i=0; i<n; i++) chosen[i] = false;
    for(int i=0; i<node->m; i++){
        int id;
        do{
            id = rand() % n;
        }while(chosen[id]);
        chosen[id] = true;
        node->pivots[i] = arr[id];
    }

    // assign each point to nearest pivot
    Point* subset = new Point[M * N_MAX];
    int* subsetSize = new int[M];
    for(int i=0; i<M; i++){
        subsetSize[i] = 0;
    }

    for(int i=0; i<n; i++){
        if(chosen[i]) continue;
        float best = distance(arr[i], node->pivots[0]);
        computationsBuild++;
        int bestIdx = 0;
        for(int j=1; j<node->m; j++){
            float d = distance(arr[i], node->pivots[j]);
            computationsBuild++;
            if(d<best){
                best = d;
                bestIdx = j;
            }
        }
        subset[bestIdx*N_MAX+subsetSize[bestIdx]++] = arr[i];
    }

    // compute distance ranges between pivots and subsets
    for(int i=0; i<node->m; i++){
        for(int j=0; j<node->m; j++){
            if(i==j){
                node->rangeLow[i][j] = 0;
                node->rangeHigh[i][j] = 0;
            } 
            else{
                float minD = numeric_limits<float>::infinity();
                float maxD = 0;
                for(int k=0; k<subsetSize[j]; k++){
                    float d = distance(node->pivots[i], subset[j*N_MAX+k]);
                    computationsBuild++;
                    if(d<minD) minD = d;
                    if(d>maxD) maxD = d;
                }
                float dpp = distance(node->pivots[i], node->pivots[j]);
                computationsBuild++;
                if(dpp<minD) minD = dpp;
                if(dpp>maxD) maxD = dpp;
                node->rangeLow[i][j] = minD;
                node->rangeHigh[i][j] = maxD;
            }
        }
    }

    // recursively build children
    for(int i=0; i<node->m; i++){
        node->child[i] = buildGNAT(subset+i*N_MAX, subsetSize[i], leaf_size);
    }
    delete []subset;
    delete []subsetSize;
    delete []chosen;

    return node;
}

// ---------------------- Search ----------------------
void search(GNATNode* node, const Point &q, Point &bestPt, float &bestDist) {
    if (!node) return;

    if (node->isLeaf) {
        for (int i = 0; i < node->leafCount; i++) {
            float d = distance(q, node->leafPoints[i]);
            computationsSearch++;
            if (d < bestDist) {
                bestDist = d;
                bestPt = node->leafPoints[i];
            }
        }
        return;
    }

    float* distPivot = new float[M];
    for (int i = 0; i < node->m; i++){
        distPivot[i] = distance(q, node->pivots[i]);
        computationsSearch++;
    }
        

    for (int i = 0; i < node->m; i++) {
        if (distPivot[i] < bestDist) {
            bestDist = distPivot[i];
            bestPt = node->pivots[i];
        }
    }

    bool* prune = new bool[M];
    for(int i = 0; i < node->m; i++) prune[i] = false;

    for (int i = 0; i < node->m; i++) {
        for (int j = 0; j < node->m; j++) {
            if(i==j) continue;
            if(prune[j]) continue;
            if (distPivot[i] - bestDist > node->rangeHigh[i][j] ||
                distPivot[i] + bestDist < node->rangeLow[i][j]) {
                prune[j] = true;
            }
        }
    }
    for(int i=0; i<node->m; i++){
        if(!prune[i]) search(node->child[i], q, bestPt, bestDist);
    }
    delete []prune;
    delete []distPivot;
}

// ---------------------- Delete/cleanup ----------------------
void deleteGNAT(GNATNode* node){
    if(!node) return;
    if(!node->isLeaf){
        for(int i=0; i<node->m; i++){
            if(node->child[i]){
                deleteGNAT(node->child[i]);
                node->child[i] = nullptr;
            }
        }
    }
    delete node;
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

    mt19937 rng((unsigned)time(0));
    uniform_real_distribution<float> dist(-10.0f, 10.0f);

    double totalBuildTime = 0.0;
    double totalSearchTime = 0.0;
    long long totalDistBuild = 0, totalDistSearch = 0, totalPivots = 0;

    for(int iter=0; iter<ITERATIONS; iter++){
        computationsBuild = computationsSearch = pivotCount = 0;

        auto build_start = high_resolution_clock::now();
        GNATNode* root = buildGNAT(points, N_MAX);
        auto build_end = high_resolution_clock::now();
        totalBuildTime += duration_cast<microseconds>(build_end - build_start).count();

        Point q;
        for(int j=0; j<D; j++) q.coords[j] = dist(rng);

        Point bestPt;
        float bestDist = numeric_limits<float>::infinity();

        auto search_start = high_resolution_clock::now();
        search(root, q, bestPt, bestDist);
        auto search_end = high_resolution_clock::now();
        totalSearchTime += duration_cast<microseconds>(search_end - search_start).count();

        totalDistBuild += computationsBuild;
        totalDistSearch += computationsSearch;
        totalPivots += pivotCount;

        deleteGNAT(root);
    }

    cout<<fixed<<setprecision(2);
    cout<<"\nAveraged over "<<ITERATIONS<<" iterations:"<<endl;
    cout<<"Average build time: "<<(totalBuildTime/ITERATIONS)<<" microseconds"<<endl;
    cout<<"Average search time: "<<(totalSearchTime/ITERATIONS)<<" microseconds"<<endl;
    cout<<"Average distance computations in building: "<<(totalDistBuild/ITERATIONS)<<endl;
    cout<<"Average distance computations in searching: "<<(totalDistSearch/ITERATIONS)<<endl;
    cout<<"Average pivots used: "<<(totalPivots/ITERATIONS)<<endl;

    GNATNode* root = buildGNAT(points, N_MAX, 4);
    Point q;
    for (int j = 0; j < D; j++) q.coords[j] = dist(rng);
    Point bestPoint;
    float bestDist = numeric_limits<float>::infinity();
    search(root, q, bestPoint, bestDist);

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
    cout << "\nTime taken = " << totalSearchTimeBrute << " microseconds"<<endl;
}


