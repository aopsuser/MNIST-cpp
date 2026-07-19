#include "Matrix.h"

#include <stdexcept>
#include <random>
#include <iomanip>

// ------------------------------------------------------------------
// Конструктор
// ------------------------------------------------------------------

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(
            "Matrix: количество строк и столбцов должно быть больше нуля");
    }
}

// ------------------------------------------------------------------
// Доступ к элементам
// ------------------------------------------------------------------

double& Matrix::at(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix::at: индекс вне границ матрицы");
    }
    return data_[index(row, col)];
}

double Matrix::at(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix::at: индекс вне границ матрицы");
    }
    return data_[index(row, col)];
}

// ------------------------------------------------------------------
// Инициализация
// ------------------------------------------------------------------

void Matrix::randomize(double min_val, double max_val,
                        unsigned int seed, bool use_seed) {
    // std::random_device даёт недетерминированное зерно по умолчанию;
    // если пользователь явно передал seed, используем его — это полезно
    // для воспроизводимых юнит-тестов.
    std::mt19937 gen;
    if (use_seed) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }

    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (double& value : data_) {
        value = dist(gen);
    }
}

void Matrix::fill(double value) {
    for (double& v : data_) {
        v = value;
    }
}

// ------------------------------------------------------------------
// Арифметические операции
// ------------------------------------------------------------------

Matrix Matrix::add(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Matrix::add: размеры матриц должны совпадать");
    }

    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Matrix::subtract: размеры матриц должны совпадать");
    }

    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::multiply(double scalar) const {
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Matrix Matrix::dot(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument(
            "Matrix::dot: число столбцов левой матрицы должно совпадать "
            "с числом строк правой матрицы");
    }

    Matrix result(rows_, other.cols_);

    // Классическое тройное вложенное умножение O(n^3).
    // Порядок циклов (i, k, j) выбран для лучшей локальности кэша:
    // внутренний цикл по j идёт последовательно и по data_, и по
    // other.data_, и по result.data_.
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < cols_; ++k) {
            double a_ik = data_[index(i, k)];
            if (a_ik == 0.0) {
                continue; // небольшая оптимизация для разреженных матриц
            }
            for (std::size_t j = 0; j < other.cols_; ++j) {
                result.data_[result.index(i, j)] +=
                    a_ik * other.data_[other.index(k, j)];
            }
        }
    }

    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result.at(j, i) = at(i, j);
        }
    }
    return result;
}

// ------------------------------------------------------------------
// Сравнение
// ------------------------------------------------------------------

bool Matrix::equals(const Matrix& other, double eps) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        return false;
    }
    for (std::size_t i = 0; i < data_.size(); ++i) {
        double diff = data_[i] - other.data_[i];
        if (diff < 0) diff = -diff;
        if (diff > eps) {
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------
// Отладочный вывод
// ------------------------------------------------------------------

void Matrix::print(std::ostream& os) const {
    os << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < rows_; ++i) {
        os << "[ ";
        for (std::size_t j = 0; j < cols_; ++j) {
            os << std::setw(9) << at(i, j) << " ";
        }
        os << "]\n";
    }
    os << std::defaultfloat;
}
