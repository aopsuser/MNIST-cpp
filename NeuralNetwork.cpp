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
      bias_output_(1, output_size),
      last_input_(1, input_size),
      last_hidden_z_(1, hidden_size),
      last_hidden_a_(1, hidden_size),
      last_output_a_(1, output_size) {

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

Matrix NeuralNetwork::relu_derivative(const Matrix& z) {
    Matrix result(z.rows(), z.cols());
    for (std::size_t i = 0; i < z.rows(); ++i) {
        for (std::size_t j = 0; j < z.cols(); ++j) {
            result.at(i, j) = z.at(i, j) > 0.0 ? 1.0 : 0.0;
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

Matrix NeuralNetwork::one_hot(int label, std::size_t num_classes) {
    Matrix result(1, num_classes);
    result.fill(0.0);
    result.at(0, static_cast<std::size_t>(label)) = 1.0;
    return result;
}

double NeuralNetwork::cross_entropy_loss(const Matrix& probabilities, int label) {
    double p = probabilities.at(0, static_cast<std::size_t>(label));
    double epsilon = 1e-12;
    return -std::log(std::max(p, epsilon));
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    if (input.rows() != 1 || input.cols() != input_size_) {
        throw std::invalid_argument(
            "NeuralNetwork::forward: ожидается матрица 1x" +
            std::to_string(input_size_));
    }

    last_input_ = input;
    last_hidden_z_ = input.dot(weights_input_hidden_).add(bias_hidden_);
    last_hidden_a_ = relu(last_hidden_z_);

    Matrix output_z = last_hidden_a_.dot(weights_hidden_output_).add(bias_output_);
    last_output_a_ = softmax(output_z);

    return last_output_a_;
}

void NeuralNetwork::backward(int label, double learning_rate) {
    Matrix y = one_hot(label, output_size_);
    Matrix dz_output = last_output_a_.subtract(y);

    Matrix hidden_a_transposed = last_hidden_a_.transpose();
    Matrix dw_output = hidden_a_transposed.dot(dz_output);
    Matrix db_output = dz_output;

    Matrix weights_hidden_output_transposed = weights_hidden_output_.transpose();
    Matrix da_hidden = dz_output.dot(weights_hidden_output_transposed);

    Matrix relu_grad = relu_derivative(last_hidden_z_);
    Matrix dz_hidden(1, hidden_size_);
    for (std::size_t j = 0; j < hidden_size_; ++j) {
        dz_hidden.at(0, j) = da_hidden.at(0, j) * relu_grad.at(0, j);
    }

    Matrix input_transposed = last_input_.transpose();
    Matrix dw_hidden = input_transposed.dot(dz_hidden);
    Matrix db_hidden = dz_hidden;

    weights_hidden_output_ = weights_hidden_output_.subtract(dw_output.multiply(learning_rate));
    bias_output_ = bias_output_.subtract(db_output.multiply(learning_rate));

    weights_input_hidden_ = weights_input_hidden_.subtract(dw_hidden.multiply(learning_rate));
    bias_hidden_ = bias_hidden_.subtract(db_hidden.multiply(learning_rate));
}

double NeuralNetwork::train_sample(const Matrix& input, int label, double learning_rate) {
    Matrix probabilities = forward(input);
    double loss = cross_entropy_loss(probabilities, label);
    backward(label, learning_rate);
    return loss;
}
