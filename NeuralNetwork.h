#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Matrix.h"

class NeuralNetwork {
public:
    NeuralNetwork(std::size_t input_size, std::size_t hidden_size,
                  std::size_t output_size);

    Matrix forward(const Matrix& input);

    static Matrix relu(const Matrix& z);
    static Matrix softmax(const Matrix& z);

private:
    std::size_t input_size_;
    std::size_t hidden_size_;
    std::size_t output_size_;

    Matrix weights_input_hidden_;
    Matrix bias_hidden_;

    Matrix weights_hidden_output_;
    Matrix bias_output_;
};

#endif
