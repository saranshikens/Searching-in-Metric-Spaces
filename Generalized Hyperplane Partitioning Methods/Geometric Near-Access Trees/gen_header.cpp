#include "dataset.h"
#include <random>

std::vector<std::vector<double>> generate_dataset(int num_points, int dimensions) {
    std::vector<std::vector<double>> data(num_points, std::vector<double>(dimensions));
    std::mt19937 gen(42); // Fixed seed for reproducible behavior
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    for (int i = 0; i < num_points; ++i) {
        for (int d = 0; d < dimensions; ++d) {
            data[i][d] = dis(gen);
        }
    }
    return data;
}