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
int metricType = 1; 


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
TreeNode* buildGHT(Point arr[], int n, int leaf_size=4){ // partitioning will stop when the partition size reaches 4
    if(n<=0) return nullptr;
    if(n<=leaf_size) return new TreeNode(arr, n);

    // choosing pivots randomly
    int idA = rand()%n;
    int idB = rand()%n;
    while(idA==idB) idB = rand()%n;
    pivotCount += 2; // two pivots used

    Point pA = arr[idA], pB = arr[idB]; // pivots for the current TreeNode
    TreeNode* node = new TreeNode(pA, pB);

    // partition of the dataset due to the pivots
    Point* leftPartition = new Point[N_MAX];
    Point* rightPartition = new Point[N_MAX]; 
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

    if(leftN+rightN==0) return new TreeNode(arr, n); // if both partitions are empty (very rare), just return a leaf node

    // recursively build the tree
    node->left = buildGHT(leftPartition, leftN, leaf_size);
    node->right = buildGHT(rightPartition, rightN, leaf_size);
    delete []leftPartition;
    delete []rightPartition;
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

    // equivalent to d(q,p1) - r <= d(q,p2) + r
    if(dA-bestDist <= dB+bestDist) search(node->left, q, bestPoint, bestDist);
    if(dB-bestDist <= dA+bestDist) search(node->right, q, bestPoint, bestDist);
}


// ---------------------- Insert ----------------------
void insertNode(TreeNode*& node, Point* q, int leaf_size = 4){
    // Case 1: empty tree
    if(node==nullptr){
        node = new TreeNode(q, 1); // single-point leaf
        return;
    }

    // Case 2: node is a leaf → insert here
    if(node->isLeaf){
        node->bucket[node->bucketSize++] = *q;   // add point

        // Check overflow → split leaf
        if(node->bucketSize > leaf_size){

            // choose pivots randomly 
            int n = node->bucketSize;
            int idA = rand()%n;
            int idB = rand()%n;
            while(idA==idB) idB = rand()%n;

            Point pA = node->bucket[idA];
            Point pB = node->bucket[idB];
            pivotCount += 2;

            // create new internal node 
            TreeNode* newNode = new TreeNode(pA, pB);

            // partition points into left/right
            Point* leftPartition  = new Point[n];
            Point* rightPartition = new Point[n];
            int leftN = 0, rightN = 0;

            for(int i=0; i<n; i++){
                if(i==idA || i==idB) continue;

                float dA = distance(node->bucket[i], pA);
                float dB = distance(node->bucket[i], pB);
                computationsBuild += 2;

                if(dA<=dB) leftPartition[leftN++] = node->bucket[i];
                else rightPartition[rightN++] = node->bucket[i];
            }

            // recursively build subtrees 
            newNode->left  = buildGHT(leftPartition, leftN, leaf_size);
            newNode->right = buildGHT(rightPartition, rightN, leaf_size);

            delete [] leftPartition;
            delete [] rightPartition;

            // Replace leaf with new internal node
            delete node;
            node = newNode;
        }
        return;
    }

    // Case 3: node is internal → descend according to pivots
    float dA = distance(*q, node->pivotA);
    float dB = distance(*q, node->pivotB);
    computationsSearch += 2; 

    if(dA<=dB) insertNode(node->left, q, leaf_size);
    else insertNode(node->right, q, leaf_size);
}


// Recursively delete tree (important to avoid memory leaks)
void deleteTree(TreeNode* node){
    if(node==nullptr) return;
    if(!node->isLeaf){
        deleteTree(node->left);
        deleteTree(node->right);
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
        
    // generate pseudo-random float values
    mt19937 rng((unsigned)time(0));
    uniform_real_distribution<float> dist(-10.0f, 10.0f);


    TreeNode* root = buildGHT(points, N_MAX, 4);
    // -------------------- Test insertion correctness --------------------
    cout<<"\n=== INSERTION TEST START ===\n";

    // pick a query point q
    Point q2;
    for(int j=0; j<D; j++) q2.coords[j] = dist(rng);

    // find NN BEFORE insertion
    Point nn_before;
    float dist_before = numeric_limits<float>::infinity();
    search(root, q2, nn_before, dist_before);

    cout<<"\nQuery point q2:\n";
    printPoint(q2);
    cout<<"\nNN before insertion:\n";
    printPoint(nn_before);
    cout<<"\nDistance before = "<<dist_before<<endl;

    // ---------- Create point extremely close to q2 ----------
    Point newPoint;
    float eps = 1e-3f;
    for(int j=0; j<D; j++)
        newPoint.coords[j] = q2.coords[j] + eps;     // new point is q + eps

    cout<<"\nInserting new point (q2 + eps):\n";
    printPoint(newPoint);
    cout<<endl;

    // INSERT INTO TREE
    insertNode(root, &newPoint, 4);

    // ---------- Find NN AFTER insertion ----------
    Point nn_after;
    float dist_after = numeric_limits<float>::infinity();
    search(root, q2, nn_after, dist_after);

    cout << "\nNN after insertion:\n";
    printPoint(nn_after);
    cout << "\nDistance after = " << dist_after << endl;

    // ---------- Brute force check ----------
    float brute_before = numeric_limits<float>::infinity();
    float brute_after  = numeric_limits<float>::infinity();
    Point brute_before_pt, brute_after_pt;

    for(int i=0; i<N_MAX; i++){
        float d = distance(q2, points[i]);
        if(d<brute_before){
            brute_before = d;
            brute_before_pt = points[i];
        }
    }
    float d_new = distance(q2, newPoint);

    if(d_new < brute_before){
        brute_after = d_new;
        brute_after_pt = newPoint;
    } 
    else{
        brute_after = brute_before;
        brute_after_pt = brute_before_pt;
    }

    cout<<"\nBrute-force NN before insertion:\n";
    printPoint(brute_before_pt);
    cout<<"\nDistance = "<<brute_before<<endl;

    cout<<"\nBrute-force NN after insertion:\n";
    printPoint(brute_after_pt);
    cout<<"\nDistance = "<<brute_after << endl;

    cout<<"\n=== INSERTION TEST END ===\n";

    deleteTree(root);
}