#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <numeric>

namespace py = pybind11;

// Example: A heavy mathematical calculation for volatility
// This runs 10-50x faster in C++ than a Python loop
std::vector<double> calculate_volatility(const std::vector<double>& data, int window) {
    std::vector<double> result;
    if (data.size() < window) return result;

    for (size_t i = 0; i <= data.size() - window; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < window; ++j) {
            sum += data[i + j];
        }
        double mean = sum / window;

        double sq_sum = 0.0;
        for (size_t j = 0; j < window; ++j) {
            sq_sum += std::pow(data[i + j] - mean, 2);
        }
        result.push_back(std::sqrt(sq_sum / window));
    }
    return result;
}

// This block exposes the C++ function to Python
PYBIND11_MODULE(nexus_cpp, m) {
    m.doc() = "Nexus AI C++ Optimization Module";
    m.def("calculate_volatility", &calculate_volatility, "Computes rolling volatility efficiently");
}