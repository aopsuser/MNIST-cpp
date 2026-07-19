/**
 * main.cpp
 *
 * Простые юнит-тесты для класса Matrix — базового математического
 * движка будущей нейросети для распознавания MNIST.
 *
 * Тесты не используют сторонних фреймворков: каждая проверка — это
 * вызов простой функции expect_true / expect_matrix_equal, которая
 * печатает результат (PASS/FAIL) и ведёт общий счётчик.
 */

#include "Matrix.h"

#include <iostream>
#include <string>
#include <stdexcept>

namespace {

int tests_run = 0;
int tests_passed = 0;

/// Проверяет булево условие и печатает результат теста.
void expect_true(bool condition, const std::string& test_name) {
    ++tests_run;
    if (condition) {
        ++tests_passed;
        std::cout << "[PASS] " << test_name << "\n";
    } else {
        std::cout << "[FAIL] " << test_name << "\n";
    }
}

/// Проверяет, что две матрицы совпадают с точностью eps.
void expect_matrix_equal(const Matrix& actual, const Matrix& expected,
                          const std::string& test_name, double eps = 1e-9) {
    expect_true(actual.equals(expected, eps), test_name);
}

/// Проверяет, что вызов f() бросает std::exception (например, при
/// несовместимых размерах матриц).
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

// ------------------------------------------------------------------
// Тесты
// ------------------------------------------------------------------

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

    // randomize с фиксированным зерном должен быть воспроизводим.
    Matrix r1(4, 4);
    Matrix r2(4, 4);
    r1.randomize(-1.0, 1.0, /*seed=*/42, /*use_seed=*/true);
    r2.randomize(-1.0, 1.0, /*seed=*/42, /*use_seed=*/true);
    expect_matrix_equal(r1, r2,
        "randomize(): одинаковое зерно даёт одинаковый результат");

    // Все значения должны лежать в заданном диапазоне.
    bool in_range = true;
    for (std::size_t i = 0; i < r1.rows(); ++i)
        for (std::size_t j = 0; j < r1.cols(); ++j)
            if (r1.at(i, j) < -1.0 || r1.at(i, j) > 1.0) in_range = false;
    expect_true(in_range, "randomize(): значения находятся в диапазоне [-1, 1]");

    // Разные зёрна должны (почти наверняка) давать разный результат.
    Matrix r3(4, 4);
    r3.randomize(-1.0, 1.0, /*seed=*/123, /*use_seed=*/true);
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
    // [1 2 3]   [ 7  8]
    // [4 5 6] * [ 9 10]
    //           [11 12]
    Matrix a(2, 3);
    a.at(0, 0) = 1; a.at(0, 1) = 2; a.at(0, 2) = 3;
    a.at(1, 0) = 4; a.at(1, 1) = 5; a.at(1, 2) = 6;

    Matrix b(3, 2);
    b.at(0, 0) = 7;  b.at(0, 1) = 8;
    b.at(1, 0) = 9;  b.at(1, 1) = 10;
    b.at(2, 0) = 11; b.at(2, 1) = 12;

    // Ожидаемый результат вычислен вручную:
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    Matrix expected(2, 2);
    expected.at(0, 0) = 58;  expected.at(0, 1) = 64;
    expected.at(1, 0) = 139; expected.at(1, 1) = 154;

    expect_matrix_equal(a.dot(b), expected,
        "dot(): корректное матричное умножение (прямоугольные матрицы)");
    expect_matrix_equal(a * b, expected, "operator*(Matrix): соответствует dot()");

    // Умножение на единичную матрицу не должно менять значения.
    Matrix identity(3, 3);
    identity.at(0, 0) = 1; identity.at(1, 1) = 1; identity.at(2, 2) = 1;
    expect_matrix_equal(a.dot(identity), a,
        "dot(): умножение на единичную матрицу не изменяет исходную");

    // Несовместимые размеры должны бросать исключение.
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

} // namespace

int main() {
    std::cout << "=== Тесты математического движка Matrix ===\n\n";

    test_construction_and_default_values();
    test_element_access();
    test_fill_and_randomize();
    test_addition();
    test_subtraction();
    test_scalar_multiplication();
    test_dot_product();
    test_transpose();

    std::cout << "\n=== Итог: " << tests_passed << " / " << tests_run
              << " тестов пройдено ===\n";

    // Небольшая демонстрация вывода матрицы (пригодится для отладки весов).
    std::cout << "\nПример матрицы весов 3x3, инициализированной случайно:\n";
    Matrix weights(3, 3);
    weights.randomize(-0.5, 0.5, 42, true);
    weights.print();

    return (tests_passed == tests_run) ? 0 : 1;
}
