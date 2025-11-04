#include <iostream>
#include <random>
#include <iomanip>
using namespace std;

#define N_MAX 200
#define D_MAX 50

int main() {
    mt19937 rng((unsigned)time(0));
    uniform_real_distribution<float> dist(-10.0f, 10.0f);

    cout << "#pragma once\n";
    cout << "#define N_MAX " << N_MAX << "\n";
    cout << "#define D_MAX " << D_MAX << "\n";
    cout << "float DATASET[" << N_MAX << "][" << D_MAX << "] = {\n";

    for(int i=0; i<N_MAX; i++){
        cout << "  {";
        for(int j=0; j<D_MAX; j++){
            cout << fixed << setprecision(3) << dist(rng);
            if(j < D_MAX-1) cout << ", ";
        }
        cout << "}";
        if(i < N_MAX-1) cout << ",";
        cout << "\n";
    }
    cout << "};\n";
}
