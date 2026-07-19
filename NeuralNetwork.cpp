#include "NeuralNetwork.h"

#include <stdexcept>
#include <cmath>
#include <algorithm>

NeuralNetwork::NeuralNetwork(std::size_t input_size, std::size_t hidden_size,
                              std::size_t output_size)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      output_size_(output_size),
      weights_input_hidden_(input_size, hidden_size),
      bias_hidden_(1, hidden_size),
      weights_hidden_output_(hidden_size, output_size),
      bias_output_(1, output_size) {

    double hidden_range = std::sqrt(2.0 / static_cast<double>(input_size));
    double output_range = std::sqrt(2.0 / static_cast<double>(hidden_size));

    weights_input_hidden_.randomize(-hidden_range, hidden_range);
    weights_hidden_output_.randomize(-output_range, output_range);

    bias_hidden_.fill(0.0);
    bias_output_.fill(0.0);
}

Matrix NeuralNetwork::relu(const Matrix& z) {
    Matrix result(z.rows(), z.cols());
    for (std::size_t i = 0; i < z.rows(); ++i) {
        for (std::size_t j = 0; j < z.cols(); ++j) {
            result.at(i, j) = std::max(0.0, z.at(i, j));
        }
    }
    return result;
}

Matrix NeuralNetwork::softmax(const Matrix& z) {
    Matrix result(z.rows(), z.cols());

    for (std::size_t i = 0; i < z.rows(); ++i) {
        double max_val = z.at(i, 0);
        for (std::size_t j = 1; j < z.cols(); ++j) {
            max_val = std::max(max_val, z.at(i, j));
        }

        double sum = 0.0;
        for (std::size_t j = 0; j < z.cols(); ++j) {
            double exp_val = std::exp(z.at(i, j) - max_val);
            result.at(i, j) = exp_val;
            sum += exp_val;
        }

        for (std::size_t j = 0; j < z.cols(); ++j) {
            result.at(i, j) /= sum;
        }
    }

    return result;
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    if (input.rows() != 1 || input.cols() != input_size_) {
        throw std::invalid_argument(
            "NeuralNetwork::forward: ожидается матрица 1x" +
            std::to_string(input_size_));
    }

    Matrix hidden_z = input.dot(weights_input_hidden_).add(bias_hidden_);
    Matrix hidden_a = relu(hidden_z);

    Matrix output_z = hidden_a.dot(weights_hidden_output_).add(bias_output_);
    Matrix output_a = softmax(output_z);

    return output_a;
}
