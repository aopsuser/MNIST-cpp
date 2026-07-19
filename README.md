# MNIST Neural Network from Scratch in C++

A fully connected feedforward neural network built entirely from scratch in plain C++ without any external machine learning or linear algebra libraries (no TensorFlow, PyTorch, or Eigen). 

This project was created to deeply understand the underlying math behind Deep Learning, including matrix operations, forward propagation, and the backpropagation algorithm.

## Features
* **Custom Math Engine:** A `Matrix` class implementing core linear algebra operations (dot product, transpose, scalar operations) with optimized cache-friendly loops.
* **Custom DataLoader:** Parses raw binary IDX files from the MNIST dataset directly in C++.
* **Training Pipeline:** Implements Stochastic Gradient Descent (SGD), Cross-Entropy Loss, Softmax, and ReLU activation functions.
* **Evaluation:** Built-in train/test split, dataset shuffling per epoch, and accuracy metrics.
* **Unit Testing:** Includes 36 comprehensive unit tests for matrix math and network components.

## How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/ТВОЙ_НИК/mnist-cpp-from-scratch.git](https://github.com/ТВОЙ_НИК/mnist-cpp-from-scratch.git)
Download the MNIST dataset files (train-images-idx3-ubyte, train-labels-idx1-ubyte) and place them in the data/ folder.

Compile the project using G++ (C++17 required):

Bash
g++ -std=c++17 -Wall -Wextra -O2 -o neural_net main.cpp Matrix.cpp MnistLoader.cpp NeuralNetwork.cpp
Run the executable:

Bash
./neural_net
Results
The network achieves approximately 95-97% accuracy on the MNIST test set after just a few epochs of training.
