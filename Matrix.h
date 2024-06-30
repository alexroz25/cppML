#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>

template <typename T> class Matrix {
public:
    uint64_t numRows;
    uint64_t numCols;
    std::vector<T> matrix;

    Matrix(uint64_t rows, uint64_t cols) : numRows(rows), numCols(cols) {
        matrix.resize(rows * cols, 0);
    }

    Matrix(uint64_t rows, uint64_t cols, std::vector<T> input) : numRows(rows), numCols(cols), matrix(input) {}

    T& at(uint64_t row, uint64_t col) {
        assert(row < numRows && col < numCols);
        return matrix[row * numCols + col];
    }

    Matrix<T> transpose() {
        Matrix out(numCols, numRows);
        for (uint64_t r = 0; r < numRows; ++r)
            for (uint64_t c = 0; c < numCols; ++c) out.at(c, r) = this->at(r, c);
        
        return out;
    }

    Matrix<T> add(Matrix<T>& other) {
        assert(numRows == other.numRows && numCols == other.numCols);

        Matrix out = *this;
        for (uint64_t i = 0; i < numRows * numCols; ++i) {
            out.matrix[i] += other.matrix[i];
        }

        return out;
    }

    Matrix<T> add_to_each_row(std::vector<T>& other) {
        assert(other.size() == numCols);
        
        Matrix out = *this;
        for (uint64_t r = 0; r < numRows; ++r) {
            for (uint64_t c = 0; c < numCols; ++c) {
                out.at(r, c) += other[c];
            }
        }

        return out;
    }

    Matrix<T> subtract(Matrix<T>& other) {
        assert(numRows == other.numRows && numCols == other.numCols);

        Matrix out = *this;
        for (uint64_t i = 0; i < numRows * numCols; ++i) {
            out.matrix[i] -= other.matrix[i];
        }

        return out;
    }

    Matrix<T> multiply(Matrix<T>& other) {
        assert(numCols == other.numRows);

        uint64_t cols = other.numCols;
        Matrix out(numRows, cols);
        for (uint64_t r = 0; r < numRows; ++r) {
            for (uint64_t c = 0; c < cols; ++c) {
                T temp = 0;
                for (uint64_t k = 0; k < numCols; ++k) temp += this->at(r, k) * other.at(k, c);
                out.at(r, c) = temp;
            }
        }

        return out;
    }

    Matrix<T> multiply(double scalar) {
        Matrix out = *this;
        for (T& elt : out.matrix) {
            elt *= scalar;
        }
        return out;
    }

    void print() {
        for (uint64_t r = 0; r < numRows; ++r) {
            for (uint64_t c = 0; c < numCols; ++c) std::cout << this->at(r, c) << '\t';
            std::cout << '\n';
        }
    }

    void print_dimensions() {
        std::cout << numRows << " x " << numCols << std::endl;
    }

};


#endif // MATRIX_H