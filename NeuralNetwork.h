#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Matrix.h"

class NeuralNetwork {
public:
    NeuralNetwork(std::size_t input_size, std::size_t hidden_size,
                  std::size_t output_size);

    Matrix forward(const Matrix& input);
    double train_sample(const Matrix& input, int label, double learning_rate);

    static Matrix relu(const Matrix& z);
    static Matrix relu_derivative(const Matrix& z);
    static Matrix softmax(const Matrix& z);
    static double cross_entropy_loss(const Matrix& probabilities, int label);
    static Matrix one_hot(int label, std::size_t num_classes);

private:
    std::size_t input_size_;
    std::size_t hidden_size_;
    std::size_t output_size_;

    Matrix weights_input_hidden_;
    Matrix bias_hidden_;

    Matrix weights_hidden_output_;
    Matrix bias_output_;

    Matrix last_input_;
    Matrix last_hidden_z_;
    Matrix last_hidden_a_;
    Matrix last_output_a_;

    void backward(int label, double learning_rate);
};

#endif

