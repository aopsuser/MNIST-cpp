#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstddef>
#include <iostream>

/**
 * @class Matrix
 * @brief Простой класс плотной матрицы вещественных чисел для математического
 *        движка нейросети.
 *
 * Матрица хранится построчно (row-major) в одномерном std::vector<double>.
 * Индексация двумерная: (row, col), где row in [0, rows), col in [0, cols).
 *
 * Класс покрывает базовые операции, необходимые для прямого и обратного
 * распространения в полносвязной сети:
 *   - создание матрицы заданного размера (нули по умолчанию);
 *   - заполнение случайными числами (инициализация весов);
 *   - поэлементное сложение и вычитание;
 *   - умножение на скаляр;
 *   - матричное умножение (dot product).
 */
class Matrix {
public:
    // ------------------------------------------------------------------
    // Конструкторы
    // ------------------------------------------------------------------

    /**
     * @brief Создаёт матрицу размера rows x cols, заполненную нулями.
     * @param rows Количество строк (> 0).
     * @param cols Количество столбцов (> 0).
     * @throws std::invalid_argument если rows == 0 или cols == 0.
     */
    Matrix(std::size_t rows, std::size_t cols);

    // ------------------------------------------------------------------
    // Доступ к размерам и элементам
    // ------------------------------------------------------------------

    /// Возвращает количество строк матрицы.
    std::size_t rows() const noexcept { return rows_; }

    /// Возвращает количество столбцов матрицы.
    std::size_t cols() const noexcept { return cols_; }

    /**
     * @brief Доступ к элементу (row, col) с проверкой границ (чтение/запись).
     * @throws std::out_of_range если индексы выходят за пределы матрицы.
     */
    double& at(std::size_t row, std::size_t col);

    /**
     * @brief Доступ к элементу (row, col) с проверкой границ (только чтение).
     * @throws std::out_of_range если индексы выходят за пределы матрицы.
     */
    double at(std::size_t row, std::size_t col) const;

    // ------------------------------------------------------------------
    // Инициализация
    // ------------------------------------------------------------------

    /**
     * @brief Заполняет матрицу случайными числами из равномерного
     *        распределения [min_val, max_val]. Используется для
     *        инициализации весов сети.
     * @param min_val Нижняя граница диапазона.
     * @param max_val Верхняя граница диапазона.
     * @param seed    Опциональное зерно генератора для воспроизводимости.
     *                Если не задано, используется недетерминированный источник.
     */
    void randomize(double min_val = -1.0, double max_val = 1.0,
                   unsigned int seed = 0, bool use_seed = false);

    /// Заполняет всю матрицу одним и тем же значением.
    void fill(double value);

    // ------------------------------------------------------------------
    // Арифметические операции
    // ------------------------------------------------------------------

    /**
     * @brief Поэлементное сложение двух матриц одинакового размера.
     * @throws std::invalid_argument при несовпадении размеров.
     */
    Matrix add(const Matrix& other) const;

    /**
     * @brief Поэлементное вычитание: this - other.
     * @throws std::invalid_argument при несовпадении размеров.
     */
    Matrix subtract(const Matrix& other) const;

    /**
     * @brief Умножение всех элементов матрицы на скаляр.
     */
    Matrix multiply(double scalar) const;

    /**
     * @brief Матричное умножение (dot product): (this) * (other).
     *
     * Число столбцов this должно совпадать с числом строк other.
     * Результат имеет размер (this.rows() x other.cols()).
     *
     * @throws std::invalid_argument при несовместимых размерах.
     */
    Matrix dot(const Matrix& other) const;

    /**
     * @brief Транспонирование матрицы.
     */
    Matrix transpose() const;

    // ------------------------------------------------------------------
    // Операторы (удобные обёртки над методами выше)
    // ------------------------------------------------------------------

    Matrix operator+(const Matrix& other) const { return add(other); }
    Matrix operator-(const Matrix& other) const { return subtract(other); }
    Matrix operator*(double scalar) const { return multiply(scalar); }
    Matrix operator*(const Matrix& other) const { return dot(other); }

    /// Сравнение на равенство с абсолютной точностью eps (для тестов).
    bool equals(const Matrix& other, double eps = 1e-9) const;

    // ------------------------------------------------------------------
    // Отладочный вывод
    // ------------------------------------------------------------------

    /// Печатает матрицу построчно в поток вывода (по умолчанию std::cout).
    void print(std::ostream& os = std::cout) const;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_; // построчное (row-major) хранение

    /// Переводит 2D-индекс (row, col) в линейный индекс в data_.
    std::size_t index(std::size_t row, std::size_t col) const noexcept {
        return row * cols_ + col;
    }
};

#endif // MATRIX_H
