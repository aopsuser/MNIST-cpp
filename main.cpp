#include "Matrix.h"
#include "MnistLoader.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

namespace {

int tests_run = 0;
int tests_passed = 0;

void expect_true(bool condition, const std::string& test_name) {
    ++tests_run;
    if (condition) {
        ++tests_passed;
        std::cout << "[PASS] " << test_name << "\n";
    } else {
        std::cout << "[FAIL] " << test_name << "\n";
    }
}

void expect_matrix_equal(const Matrix& actual, const Matrix& expected,
                          const std::string& test_name, double eps = 1e-9) {
    expect_true(actual.equals(expected, eps), test_name);
}

template <typename Func>
void expect_throws(Func f, const std::string& test_name) {
    ++tests_run;
    try {
        f();
        std::cout << "[FAIL] " << test_name << " (исключение не было брошено)\n";
    } catch (const std::exception&) {
        ++tests_passed;
        std::cout << "[PASS] " << test_name << "\n";
    }
}

void test_construction_and_default_values() {
    Matrix m(2, 3);
    expect_true(m.rows() == 2 && m.cols() == 3,
                "Конструктор: правильные размеры матрицы");

    bool all_zero = true;
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            if (m.at(i, j) != 0.0) all_zero = false;
        }
    }
    expect_true(all_zero, "Конструктор: матрица по умолчанию заполнена нулями");

    expect_throws([]() { Matrix bad(0, 3); },
                   "Конструктор: 0 строк должен бросать исключение");
    expect_throws([]() { Matrix bad(3, 0); },
                   "Конструктор: 0 столбцов должен бросать исключение");
}

void test_element_access() {
    Matrix m(2, 2);
    m.at(0, 0) = 1.0;
    m.at(0, 1) = 2.0;
    m.at(1, 0) = 3.0;
    m.at(1, 1) = 4.0;

    expect_true(m.at(0, 0) == 1.0 && m.at(1, 1) == 4.0,
                "at(): запись и чтение элементов работают корректно");

    expect_throws([&m]() { m.at(5, 0); },
                   "at(): выход за границы по строке бросает исключение");
    expect_throws([&m]() { m.at(0, 5); },
                   "at(): выход за границы по столбцу бросает исключение");
}

void test_fill_and_randomize() {
    Matrix m(3, 3);
    m.fill(7.5);
    bool all_correct = true;
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j)
            if (m.at(i, j) != 7.5) all_correct = false;
    expect_true(all_correct, "fill(): все элементы установлены в заданное значение");

    Matrix r1(4, 4);
    Matrix r2(4, 4);
    r1.randomize(-1.0, 1.0, 42, true);
    r2.randomize(-1.0, 1.0, 42, true);
    expect_matrix_equal(r1, r2,
        "randomize(): одинаковое зерно даёт одинаковый результат");

    bool in_range = true;
    for (std::size_t i = 0; i < r1.rows(); ++i)
        for (std::size_t j = 0; j < r1.cols(); ++j)
            if (r1.at(i, j) < -1.0 || r1.at(i, j) > 1.0) in_range = false;
    expect_true(in_range, "randomize(): значения находятся в диапазоне [-1, 1]");

    Matrix r3(4, 4);
    r3.randomize(-1.0, 1.0, 123, true);
    expect_true(!r1.equals(r3),
        "randomize(): разные зёрна дают разные матрицы");
}

void test_addition() {
    Matrix a(2, 2);
    a.at(0, 0) = 1; a.at(0, 1) = 2;
    a.at(1, 0) = 3; a.at(1, 1) = 4;

    Matrix b(2, 2);
    b.at(0, 0) = 5; b.at(0, 1) = 6;
    b.at(1, 0) = 7; b.at(1, 1) = 8;

    Matrix expected(2, 2);
    expected.at(0, 0) = 6;  expected.at(0, 1) = 8;
    expected.at(1, 0) = 10; expected.at(1, 1) = 12;

    expect_matrix_equal(a.add(b), expected, "add(): корректное поэлементное сложение");
    expect_matrix_equal(a + b, expected, "operator+: соответствует add()");

    Matrix wrong_size(3, 3);
    expect_throws([&]() { a.add(wrong_size); },
                   "add(): несовпадающие размеры бросают исключение");
}

void test_subtraction() {
    Matrix a(2, 2);
    a.at(0, 0) = 5; a.at(0, 1) = 6;
    a.at(1, 0) = 7; a.at(1, 1) = 8;

    Matrix b(2, 2);
    b.at(0, 0) = 1; b.at(0, 1) = 2;
    b.at(1, 0) = 3; b.at(1, 1) = 4;

    Matrix expected(2, 2);
    expected.at(0, 0) = 4; expected.at(0, 1) = 4;
    expected.at(1, 0) = 4; expected.at(1, 1) = 4;

    expect_matrix_equal(a.subtract(b), expected,
        "subtract(): корректное поэлементное вычитание");
    expect_matrix_equal(a - b, expected, "operator-: соответствует subtract()");

    Matrix wrong_size(1, 2);
    expect_throws([&]() { a.subtract(wrong_size); },
                   "subtract(): несовпадающие размеры бросают исключение");
}

void test_scalar_multiplication() {
    Matrix a(2, 2);
    a.at(0, 0) = 1; a.at(0, 1) = -2;
    a.at(1, 0) = 3; a.at(1, 1) = 4;

    Matrix expected(2, 2);
    expected.at(0, 0) = 2;  expected.at(0, 1) = -4;
    expected.at(1, 0) = 6;  expected.at(1, 1) = 8;

    expect_matrix_equal(a.multiply(2.0), expected,
        "multiply(scalar): корректное умножение на скаляр");
    expect_matrix_equal(a * 2.0, expected, "operator*(scalar): соответствует multiply()");

    Matrix zeros(2, 2);
    expect_matrix_equal(a.multiply(0.0), zeros,
        "multiply(0): умножение на 0 даёт нулевую матрицу");
}

void test_dot_product() {
    Matrix a(2, 3);
    a.at(0, 0) = 1; a.at(0, 1) = 2; a.at(0, 2) = 3;
    a.at(1, 0) = 4; a.at(1, 1) = 5; a.at(1, 2) = 6;

    Matrix b(3, 2);
    b.at(0, 0) = 7;  b.at(0, 1) = 8;
    b.at(1, 0) = 9;  b.at(1, 1) = 10;
    b.at(2, 0) = 11; b.at(2, 1) = 12;

    Matrix expected(2, 2);
    expected.at(0, 0) = 58;  expected.at(0, 1) = 64;
    expected.at(1, 0) = 139; expected.at(1, 1) = 154;

    expect_matrix_equal(a.dot(b), expected,
        "dot(): корректное матричное умножение (прямоугольные матрицы)");
    expect_matrix_equal(a * b, expected, "operator*(Matrix): соответствует dot()");

    Matrix identity(3, 3);
    identity.at(0, 0) = 1; identity.at(1, 1) = 1; identity.at(2, 2) = 1;
    expect_matrix_equal(a.dot(identity), a,
        "dot(): умножение на единичную матрицу не изменяет исходную");

    Matrix incompatible(2, 2);
    expect_throws([&]() { a.dot(incompatible); },
                   "dot(): несовместимые размеры бросают исключение");
}

void test_transpose() {
    Matrix a(2, 3);
    a.at(0, 0) = 1; a.at(0, 1) = 2; a.at(0, 2) = 3;
    a.at(1, 0) = 4; a.at(1, 1) = 5; a.at(1, 2) = 6;

    Matrix expected(3, 2);
    expected.at(0, 0) = 1; expected.at(0, 1) = 4;
    expected.at(1, 0) = 2; expected.at(1, 1) = 5;
    expected.at(2, 0) = 3; expected.at(2, 1) = 6;

    Matrix result = a.transpose();
    expect_true(result.rows() == 3 && result.cols() == 2,
                "transpose(): размеры результата корректны");
    expect_matrix_equal(result, expected, "transpose(): значения переставлены верно");
}

void test_relu() {
    Matrix z(1, 4);
    z.at(0, 0) = -2.0; z.at(0, 1) = 0.0; z.at(0, 2) = 3.5; z.at(0, 3) = -0.001;

    Matrix expected(1, 4);
    expected.at(0, 0) = 0.0; expected.at(0, 1) = 0.0;
    expected.at(0, 2) = 3.5; expected.at(0, 3) = 0.0;

    expect_matrix_equal(NeuralNetwork::relu(z), expected,
        "relu(): отрицательные значения обнуляются, положительные не меняются");
}

void test_softmax() {
    Matrix z(1, 3);
    z.at(0, 0) = 1.0; z.at(0, 1) = 1.0; z.at(0, 2) = 1.0;

    Matrix result = NeuralNetwork::softmax(z);

    double sum = result.at(0, 0) + result.at(0, 1) + result.at(0, 2);
    expect_true(std::abs(sum - 1.0) < 1e-9,
        "softmax(): сумма вероятностей равна 1");

    Matrix expected_uniform(1, 3);
    expected_uniform.fill(1.0 / 3.0);
    expect_matrix_equal(result, expected_uniform,
        "softmax(): равные входы дают равные вероятности");

    Matrix z2(1, 3);
    z2.at(0, 0) = 1000.0; z2.at(0, 1) = 1000.0; z2.at(0, 2) = 1000.0;
    Matrix result2 = NeuralNetwork::softmax(z2);
    bool no_nan = true;
    for (std::size_t i = 0; i < result2.cols(); ++i) {
        if (std::isnan(result2.at(0, i))) no_nan = false;
    }
    expect_true(no_nan, "softmax(): большие значения не дают NaN (стабильность)");
}

void test_one_hot() {
    Matrix expected(1, 5);
    expected.fill(0.0);
    expected.at(0, 2) = 1.0;

    expect_matrix_equal(NeuralNetwork::one_hot(2, 5), expected,
        "one_hot(): единица стоит в позиции лейбла, остальное нули");
}

void test_cross_entropy_loss() {
    Matrix probs(1, 3);
    probs.at(0, 0) = 0.7; probs.at(0, 1) = 0.2; probs.at(0, 2) = 0.1;

    double loss = NeuralNetwork::cross_entropy_loss(probs, 0);
    double expected = -std::log(0.7);
    expect_true(std::abs(loss - expected) < 1e-9,
        "cross_entropy_loss(): значение совпадает с -log(p_label)");

    double high_loss = NeuralNetwork::cross_entropy_loss(probs, 2);
    expect_true(high_loss > loss,
        "cross_entropy_loss(): низкая вероятность правильного класса даёт больший loss");
}

void test_train_sample_reduces_loss() {
    NeuralNetwork network(10, 8, 3);

    Matrix input(1, 10);
    input.randomize(0.0, 1.0, 7, true);
    int label = 1;

    double first_loss = network.train_sample(input, label, 0.1);
    double last_loss = first_loss;
    for (int i = 0; i < 50; ++i) {
        last_loss = network.train_sample(input, label, 0.1);
    }

    expect_true(last_loss < first_loss,
        "train_sample(): loss уменьшается после многократного обучения на одном примере");
}

void test_evaluate_accuracy() {
    NeuralNetwork network(4, 6, 2);

    Matrix sample_a(1, 4);
    sample_a.at(0, 0) = 1.0; sample_a.at(0, 1) = 0.0;
    sample_a.at(0, 2) = 0.0; sample_a.at(0, 3) = 0.0;

    Matrix sample_b(1, 4);
    sample_b.at(0, 0) = 0.0; sample_b.at(0, 1) = 0.0;
    sample_b.at(0, 2) = 0.0; sample_b.at(0, 3) = 1.0;

    std::vector<Matrix> images = {sample_a, sample_b};
    std::vector<int> labels = {0, 1};

    for (int i = 0; i < 300; ++i) {
        network.train_sample(sample_a, 0, 0.2);
        network.train_sample(sample_b, 1, 0.2);
    }

    double accuracy = network.evaluate_accuracy(images, labels);
    expect_true(accuracy == 100.0,
        "evaluate_accuracy(): 100% после обучения на разделимых образцах");

    std::vector<int> mismatched_labels = {0};
    expect_throws([&]() { network.evaluate_accuracy(images, mismatched_labels); },
        "evaluate_accuracy(): несовпадение размеров images/labels бросает исключение");
}

void run_matrix_tests() {
    std::cout << "=== Тесты математического движка Matrix ===\n\n";

    test_construction_and_default_values();
    test_element_access();
    test_fill_and_randomize();
    test_addition();
    test_subtraction();
    test_scalar_multiplication();
    test_dot_product();
    test_transpose();
    test_relu();
    test_softmax();
    test_one_hot();
    test_cross_entropy_loss();
    test_train_sample_reduces_loss();
    test_evaluate_accuracy();

    std::cout << "\n=== Итог: " << tests_passed << " / " << tests_run
              << " тестов пройдено ===\n";
}

Matrix flatten_image(const Matrix& image) {
    Matrix flat(1, image.rows() * image.cols());
    for (std::size_t r = 0; r < image.rows(); ++r) {
        for (std::size_t c = 0; c < image.cols(); ++c) {
            flat.at(0, r * image.cols() + c) = image.at(r, c);
        }
    }
    return flat;
}

void run_forward_pass_demo(const Matrix& image, int label) {
    std::cout << "\n=== Прямой проход (forward pass) со случайными весами ===\n\n";

    NeuralNetwork network(784, 128, 10);

    Matrix input = flatten_image(image);
    Matrix output = network.forward(input);

    std::cout << "Истинный лейбл: " << label << "\n\n";
    std::cout << "Вероятности по классам:\n";

    double sum = 0.0;
    int predicted = 0;
    double max_prob = output.at(0, 0);

    for (std::size_t i = 0; i < output.cols(); ++i) {
        double p = output.at(0, i);
        sum += p;
        if (p > max_prob) {
            max_prob = p;
            predicted = static_cast<int>(i);
        }
        std::cout << "  " << i << ": " << p << "\n";
    }

    std::cout << "\nСумма вероятностей: " << sum << " (должна быть ~1.0)\n";
    std::cout << "Предсказанный класс: " << predicted << "\n";
}

void run_training_demo(const MnistDataset& dataset) {
    std::cout << "\n=== Обучение сети (backpropagation + SGD) ===\n\n";

    std::vector<Matrix> flattened_images;
    flattened_images.reserve(dataset.images.size());
    for (const Matrix& image : dataset.images) {
        flattened_images.push_back(flatten_image(image));
    }

    std::size_t total = flattened_images.size();
    std::size_t train_size = static_cast<std::size_t>(total * 0.8);

    std::vector<std::size_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 shuffle_rng(42);
    std::shuffle(indices.begin(), indices.end(), shuffle_rng);

    std::vector<Matrix> train_images;
    std::vector<int> train_labels;
    std::vector<Matrix> test_images;
    std::vector<int> test_labels;

    train_images.reserve(train_size);
    train_labels.reserve(train_size);
    test_images.reserve(total - train_size);
    test_labels.reserve(total - train_size);

    for (std::size_t i = 0; i < total; ++i) {
        std::size_t idx = indices[i];
        if (i < train_size) {
            train_images.push_back(flattened_images[idx]);
            train_labels.push_back(dataset.labels[idx]);
        } else {
            test_images.push_back(flattened_images[idx]);
            test_labels.push_back(dataset.labels[idx]);
        }
    }

    std::cout << "Обучающая выборка: " << train_images.size() << " образцов\n";
    std::cout << "Тестовая выборка: " << test_images.size() << " образцов\n\n";

    NeuralNetwork network(784, 128, 10);

    int num_epochs = 3;
    double learning_rate = 0.05;

    std::vector<std::size_t> train_order(train_images.size());
    std::iota(train_order.begin(), train_order.end(), 0);

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        std::shuffle(train_order.begin(), train_order.end(), shuffle_rng);

        double running_loss = 0.0;
        std::size_t step_in_window = 0;

        for (std::size_t step = 0; step < train_order.size(); ++step) {
            std::size_t idx = train_order[step];
            double loss = network.train_sample(train_images[idx], train_labels[idx], learning_rate);
            running_loss += loss;
            ++step_in_window;

            if ((step + 1) % 200 == 0) {
                double avg_loss = running_loss / static_cast<double>(step_in_window);
                std::cout << "Эпоха " << epoch << " | Шаг " << (step + 1) << " / "
                          << train_order.size() << " | Loss: " << avg_loss << "\n";
                running_loss = 0.0;
                step_in_window = 0;
            }
        }

        double train_accuracy = network.evaluate_accuracy(train_images, train_labels);
        double test_accuracy = network.evaluate_accuracy(test_images, test_labels);

        std::cout << "\n--- Конец эпохи " << epoch << " ---\n";
        std::cout << "Точность на обучающей выборке: " << train_accuracy << "%\n";
        std::cout << "Точность на тестовой выборке: " << test_accuracy << "%\n\n";
    }

    std::cout << "=== Обучение завершено ===\n\n";

    const Matrix& first_test_image = test_images.front();
    int first_test_label = test_labels.front();
    Matrix final_probabilities = network.forward(first_test_image);

    std::cout << "Проверка на первом изображении тестовой выборки (лейбл "
              << first_test_label << "):\n";
    for (std::size_t i = 0; i < final_probabilities.cols(); ++i) {
        std::cout << "  " << i << ": " << final_probabilities.at(0, i) << "\n";
    }
    std::cout << "Предсказанный класс: "
              << NeuralNetwork::predicted_class(final_probabilities) << "\n";
}

void run_mnist_demo() {
    const std::string images_path = "data/train-images-idx3-ubyte";
    const std::string labels_path = "data/train-labels-idx1-ubyte";

    std::cout << "\n=== Загрузка датасета MNIST ===\n\n";

    try {
        MnistDataset dataset = load_mnist_dataset(images_path, labels_path);

        std::cout << "Загружено изображений: " << dataset.images.size() << "\n";
        std::cout << "Загружено лейблов: " << dataset.labels.size() << "\n\n";

        const Matrix& first_image = dataset.images.front();
        int first_label = dataset.labels.front();

        std::cout << "Размер первого изображения: " << first_image.rows()
                  << "x" << first_image.cols() << "\n\n";

        print_mnist_image(first_image, first_label);

        run_forward_pass_demo(first_image, first_label);

        run_training_demo(dataset);
    } catch (const std::exception& e) {
        std::cout << "Не удалось загрузить датасет: " << e.what() << "\n";
        std::cout << "Убедитесь, что файлы " << images_path << " и "
                  << labels_path << " существуют.\n";
    }
}

}

int main() {
    run_matrix_tests();
    run_mnist_demo();
    return (tests_passed == tests_run) ? 0 : 1;
}
