#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include <cstdint>

#include "Matrix.h"

struct MnistDataset {
    std::vector<Matrix> images;
    std::vector<int> labels;
};

std::vector<Matrix> load_mnist_images(const std::string& path);
std::vector<int> load_mnist_labels(const std::string& path);
MnistDataset load_mnist_dataset(const std::string& images_path,
                                 const std::string& labels_path);

void print_mnist_image(const Matrix& image, int label);

#endif
