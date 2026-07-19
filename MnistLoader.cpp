#include "MnistLoader.h"

#include <fstream>
#include <stdexcept>
#include <iostream>

namespace {

uint32_t read_uint32_be(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    if (!file) {
        throw std::runtime_error("MNIST: не удалось прочитать заголовок файла");
    }
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           static_cast<uint32_t>(bytes[3]);
}

}

std::vector<Matrix> load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("MNIST: не удалось открыть файл " + path);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 0x00000803) {
        throw std::runtime_error("MNIST: неверная магическая константа в " + path);
    }

    uint32_t num_images = read_uint32_be(file);
    uint32_t num_rows = read_uint32_be(file);
    uint32_t num_cols = read_uint32_be(file);

    std::vector<Matrix> images;
    images.reserve(num_images);

    std::vector<unsigned char> buffer(num_rows * num_cols);

    for (uint32_t i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        if (!file) {
            throw std::runtime_error("MNIST: неожиданный конец файла " + path);
        }

        Matrix image(num_rows, num_cols);
        for (uint32_t r = 0; r < num_rows; ++r) {
            for (uint32_t c = 0; c < num_cols; ++c) {
                unsigned char pixel = buffer[r * num_cols + c];
                image.at(r, c) = static_cast<double>(pixel) / 255.0;
            }
        }
        images.push_back(std::move(image));
    }

    return images;
}

std::vector<int> load_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("MNIST: не удалось открыть файл " + path);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 0x00000801) {
        throw std::runtime_error("MNIST: неверная магическая константа в " + path);
    }

    uint32_t num_labels = read_uint32_be(file);

    std::vector<int> labels;
    labels.reserve(num_labels);

    for (uint32_t i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        if (!file) {
            throw std::runtime_error("MNIST: неожиданный конец файла " + path);
        }
        labels.push_back(static_cast<int>(label));
    }

    return labels;
}

MnistDataset load_mnist_dataset(const std::string& images_path,
                                 const std::string& labels_path) {
    MnistDataset dataset;
    dataset.images = load_mnist_images(images_path);
    dataset.labels = load_mnist_labels(labels_path);

    if (dataset.images.size() != dataset.labels.size()) {
        throw std::runtime_error(
            "MNIST: количество изображений не совпадает с количеством лейблов");
    }

    return dataset;
}

void print_mnist_image(const Matrix& image, int label) {
    static const std::string ramp = " .:-=+*#%@";

    std::cout << "Label: " << label << "\n";
    for (std::size_t r = 0; r < image.rows(); ++r) {
        for (std::size_t c = 0; c < image.cols(); ++c) {
            double value = image.at(r, c);
            std::size_t idx = static_cast<std::size_t>(value * (ramp.size() - 1));
            std::cout << ramp[idx] << ramp[idx];
        }
        std::cout << "\n";
    }
}
